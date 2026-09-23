"""The rows the agents are never shown, carved before the graph starts.

`verified_holdout_score` is the column the benchmark half of this project reports, and it only
means anything if the rows behind it were unreachable while the pipeline ran. Every partition that
already exists is reachable: `split_artifact` is written by the profiler, from a strategy intake
chose, over the frame the agents were mounted. A split a node decided cannot grade the node that
decided it, and a split drawn after the fact out of the agents' own holdout is worse -- the
estimator was fit on rows next to it and the feature transform's medians and one-hot levels were
fitted with it in scope.

So the carve happens here, above the tools boundary, before `run_pipeline` is called, and its rows
are written outside the run root. They never enter `$DS_DATASET`, `split_artifact`, or the run's
artifact store.

What that layout does and does not buy, stated plainly: the sandbox has no filesystem namespace, so
a snippet could in principle open the withheld CSV by relative path. What this gives is an
*assertable* property -- no file the run's store holds contains a withheld row, which a test checks
-- and not isolation the code does not have. Real isolation waits on the Docker backend already
parked behind `SandboxPool`.

Order of operations is rename first, carve second. `naming.materialize` claims that the two naming
arms differ in the header line and provably nowhere else, and selecting rows here would break that
claim if it happened first. It is safe in this order because `rename_map` never renames the target,
so the column stratification reads has the same name in both arms.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ds_agents.naming import Naming, materialize
from ds_agents.nodes.profiler import N_FOLDS
from ds_agents.provenance import file_sha256
from ds_agents.runnable import Runnable

# Enough rows on the agents' side to build the profiler's 5 folds and still have a fold worth of
# rows in each. Below this the run would not fail -- it would produce folds of one or two rows and
# a cv_mean that reads like a measurement.
MIN_AGENT_ROWS = 2 * N_FOLDS


class PreparedDataset(BaseModel):
    """What one run is mounted on, and what it will be graded against.

    Carries both sides because the two are only meaningful together: a withheld CSV without the
    agent CSV that produced the split is a set of rows nobody can say were withheld from anything.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_csv: Path = Field(description="Mounted at $DS_DATASET. Post-rename, post-carve.")
    withheld_csv: Path | None = Field(
        description="The graded rows, outside the run root. None where nothing was withheld."
    )
    withheld_rows: list[int] | None = Field(
        description="Positional indices into the materialised source CSV, for reproducibility."
    )
    rename: dict[str, str]
    n_agent_rows: int
    n_withheld_rows: int
    agent_sha256: str
    withheld_fraction: float
    seed: int


def _withhold_rows(labels: list[str], fraction: float, seed: int) -> list[int]:
    """Positions to withhold: a stratified sample of `fraction`, by string label.

    numpy only, deliberately, rather than `train_test_split`. sklearn's stratified shuffling is not
    contracted to be stable across versions, and this is the one partition every headline number is
    measured on -- a holdout that quietly changes identity on a dependency bump would move
    `verified_holdout_score` with nothing in the row to say why. The exact indices for `credit_g`
    at the default seed are pinned by test, so a change here is loud.

    Labels are compared as strings, and a missing label is its own group rather than an error. A
    row whose target is null is unusable to the scorer either way; dropping such rows here would
    make the agent frame silently shorter than the file it was materialised from.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for position, label in enumerate(labels):
        groups[label].append(position)

    rng = np.random.default_rng(seed)
    withheld: list[int] = []
    # Sorted, so the draw does not depend on dict insertion order and therefore on row order.
    for label in sorted(groups):
        positions = groups[label]
        if len(positions) < 2:
            # A class with one row cannot be split without emptying it on one side. Leave it with
            # the agents: a withheld holdout missing a class is a scorer error, and an agent frame
            # missing one is a fixture problem the profiler will report.
            continue
        take = max(1, math.floor(fraction * len(positions)))
        take = min(take, len(positions) - 1)
        shuffled = rng.permutation(np.array(positions))
        withheld.extend(int(p) for p in shuffled[:take])
    return sorted(withheld)


def _split_csv(source: Path, withheld: set[int], agent_path: Path, withheld_path: Path) -> None:
    """Write both sides, header first, preserving row order and every byte of every kept row.

    A pandas round trip would re-format floats and re-quote strings, so the agent frame would
    differ from the materialised source in ways `naming.materialize` went to some trouble to
    prevent. This reads and writes lines.
    """
    with source.open("r", encoding="utf-8", newline="") as handle:
        raw = handle.read()
    header, newline, body = raw.partition("\n")
    if not newline:
        raise SystemExit(f"{source} has no rows under its header, so there is nothing to withhold")

    lines = body.split(newline)
    trailing = ""
    if lines and lines[-1] == "":
        lines.pop()
        trailing = newline

    agent_lines = [line for index, line in enumerate(lines) if index not in withheld]
    withheld_lines = [line for index, line in enumerate(lines) if index in withheld]
    for path, kept in ((agent_path, agent_lines), (withheld_path, withheld_lines)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(header + newline + newline.join(kept) + trailing)


def prepare(
    runnable: Runnable,
    naming: Naming,
    *,
    into: Path,
    withheld_into: Path,
    seed: int,
) -> PreparedDataset:
    """Materialise the CSV the agents see, and carve the rows they do not.

    `withheld_fraction == 0.0` -- every fixture -- returns the materialised path unchanged and
    writes no withheld file at all. That is the whole reason fixtures are unaffected by this
    module: their rows, their split and their committed results rows are exactly what they were.

    Called once per (dataset, naming) per invocation, not once per run. Every replicate in a cell
    must be graded against the same withheld rows or the numbers in that cell are not comparable,
    which is the same once-per-invocation rule `materialize` already follows.
    """
    source, rename = materialize(runnable, naming, into)
    if runnable.withheld_fraction == 0.0:
        return PreparedDataset(
            agent_csv=source,
            withheld_csv=None,
            withheld_rows=None,
            rename=rename,
            n_agent_rows=_row_count(source),
            n_withheld_rows=0,
            agent_sha256=file_sha256(source),
            withheld_fraction=0.0,
            seed=seed,
        )

    target = rename.get(runnable.target, runnable.target)
    labels = _labels(source, target)
    withheld = _withhold_rows(labels, runnable.withheld_fraction, seed)
    withheld_set = set(withheld)
    agent_labels = [label for index, label in enumerate(labels) if index not in withheld_set]

    # Refuse rather than mislabel, which is the rule the profiler already follows for an
    # unimplemented split strategy. A carve that leaves the agents one class or a handful of rows
    # produces a run that completes and means nothing.
    if len(agent_labels) < MIN_AGENT_ROWS:
        raise SystemExit(
            f"{runnable.dataset_id}: withholding {runnable.withheld_fraction:.0%} would leave "
            f"{len(agent_labels)} rows for the agents, below the {MIN_AGENT_ROWS} needed for "
            f"{N_FOLDS} folds."
        )
    if len({label for label in agent_labels if label != ""}) < 2:
        raise SystemExit(
            f"{runnable.dataset_id}: withholding {runnable.withheld_fraction:.0%} would leave the "
            f"agents fewer than two classes of {target!r}."
        )

    withheld_into.mkdir(parents=True, exist_ok=True)
    agent_path = into / source.name
    withheld_path = withheld_into / source.name
    if agent_path == source:
        # `materialize` under `descriptive` hands back the source's own path, which for a benchmark
        # dataset is the shared cache. Writing the carved frame there would corrupt the cache for
        # every later run, so the agent frame goes beside it under `into`.
        agent_path = into / f"{runnable.dataset_id}-agent.csv"
    into.mkdir(parents=True, exist_ok=True)
    _split_csv(source, withheld_set, agent_path, withheld_path)
    (withheld_into / "withheld_rows.json").write_text(
        json.dumps({"source": str(source), "seed": seed, "rows": withheld}, indent=2)
    )

    return PreparedDataset(
        agent_csv=agent_path,
        withheld_csv=withheld_path,
        withheld_rows=withheld,
        rename=rename,
        n_agent_rows=len(labels) - len(withheld),
        n_withheld_rows=len(withheld),
        agent_sha256=file_sha256(agent_path),
        withheld_fraction=runnable.withheld_fraction,
        seed=seed,
    )


def _labels(path: Path, target: str) -> list[str]:
    """The target column, as strings, one per data row. Empty string for a missing value."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or target not in reader.fieldnames:
            raise SystemExit(
                f"{path.name}: target {target!r} is not a column, so the carve cannot stratify. "
                f"Columns: {', '.join(reader.fieldnames or [])}"
            )
        return [(row.get(target) or "") for row in reader]


def _row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)
