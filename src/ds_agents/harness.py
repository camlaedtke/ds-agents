"""The benchmark harness: cells, a plan, and one loop that turns a plan into rows.

A `Cell` is one point in the benchmark grid -- a dataset plus the run conditions `_fixture_state`
needs to reproduce it. `SUBSETS` names groups of cells worth running together. `plan()` expands a
subset into an ordered list of individual runs, replicated and indexed. `run_eval()` walks that
plan, calling a `Runner` for each entry and writing every publishable result through
`cli._append_results_row` -- the same gate `--results` uses, so a harness result and a
`ds-agents run --results` result cannot silently disagree about what counts as a result.

`cli.py` imports this module inside `cmd_eval`, and this module imports `cli` inside its own
functions rather than at module scope, to avoid a circular top-level import between the two.

`random_seed` never appears as a `Cell` field, a `plan()` parameter, or a `run_eval()` argument, and
should not gain one. It fixes the train/test split and estimator seeding on `RunConfig`, not which
model answered; moving it between cells would fold split variance into what this project reports as
model variance. `replicates` reruns the SAME seed instead, so the spread it measures is model
nondeterminism alone. See DECISIONS.md (2026-08-28).
"""

import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ds_agents.holdout import PreparedDataset
from ds_agents.naming import Naming
from ds_agents.provenance import REPO_ROOT
from ds_agents.state import (
    DEFAULT_LOOP_CAP,
    ObjectionClosure,
    ObjectionRouting,
    PipelineState,
    ReviewerPrompt,
)

# Where results land unless a caller says otherwise. Named so the CLI can pass the same default
# explicitly rather than keeping a second copy of the path that could drift from this one.
RESULTS_DIR = REPO_ROOT / "evals" / "results"

# How many runs may fail back to back before the invocation gives up. Named rather than inlined,
# beside `loop_cap` and `max_cost_usd` as the third bound on how much a single invocation can
# consume: a config broken in a way every run hits should cost three runs, not a whole cap.
MAX_CONSECUTIVE_FAILURES = 3

# Keeps a path separator (and anything else shell- or filesystem-unfriendly) out of the results
# filename, and keeps every filename this harness has ever produced grep-consistent.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True)
class Cell:
    """One point in the benchmark grid: a dataset plus the run conditions that reproduce it.

    Every field except `name`, `dataset` and `est_cost_usd` mirrors a keyword `_fixture_state`
    accepts, so `conditions()` can hand them straight to `cli._run_once` with no translation layer.
    `est_cost_usd` is planning-only: it feeds the cost cap check and the printed plan, and is never
    written to a results row, so it can never be mistaken for a measured `cost_usd`.

    Deliberately absent: `forced_drop_release`. It exists on `RunConfig` for one same-commit
    control already run (see DECISIONS.md 2026-08-28, fifth entry), not as a second design a `Cell`
    should offer. `_fixture_state` falls back to its correct default (`withdrawn_only`) instead.
    """

    name: str
    dataset: str
    model: str = "haiku"
    reviewer_model: str | None = None
    naming: Naming = "descriptive"
    reviewer_prompt: ReviewerPrompt = "base"
    loop_cap: int = DEFAULT_LOOP_CAP
    objection_routing: ObjectionRouting = "as_addressed"
    objection_closure: ObjectionClosure = "off"
    est_cost_usd: float = 0.03

    def conditions(self) -> dict[str, Any]:
        """Keyword arguments for `cli._fixture_state` / `cli._run_once`.

        Renamed to `model_name` / `reviewer_model_name` to match `_fixture_state`'s signature;
        `Cell.model` stays short since it's read constantly while defining `SUBSETS` below.
        """
        return {
            "model_name": self.model,
            "reviewer_model_name": self.reviewer_model,
            "naming": self.naming,
            "reviewer_prompt": self.reviewer_prompt,
            "loop_cap": self.loop_cap,
            "objection_routing": self.objection_routing,
            "objection_closure": self.objection_closure,
        }


# Shared by the `ci` and `claims-repro` subsets so "identical conditions" is enforced by the
# object, not retyped.
CLAIMS_OPAQUE_WHICH = Cell(
    name="claims-opaque-which",
    dataset="claims_timing",
    naming="opaque",
    reviewer_prompt="which_column",
    objection_routing="by_category",
    est_cost_usd=0.030,
)

SUBSETS: dict[str, tuple[Cell, ...]] = {
    "toy": (Cell(name="toy-default", dataset="toy", est_cost_usd=0.015),),
    # Regression gate: one cell per fixture. The two claims cells run opaque naming + by_category
    # routing together, the combination that makes the leakage trap findable and droppable.
    # Costs are measured means from evals/results/2026-08-31_ci-baseline.jsonl, rounded up.
    "ci": (
        Cell(name="toy-default", dataset="toy", est_cost_usd=0.015),
        CLAIMS_OPAQUE_WHICH,
        Cell(
            name="reissued-opaque-which",
            dataset="reissued_ids",
            naming="opaque",
            reviewer_prompt="which_column",
            objection_routing="by_category",
            est_cost_usd=0.029,
        ),
    ),
    # Ten runs of the Result 3 cell only, to check it still reproduces at HEAD. Compares against
    # evals/results/2026-08-31_ci-baseline.jsonl; crosses a commit boundary, so eval-diff treats
    # the two sides as separate cells and the pooling has to be stated by hand.
    "claims-repro": (CLAIMS_OPAQUE_WHICH,),
    # Cheapest manifest dataset, run alone to prove the run-and-grade path before paying for the
    # rest. Price is the measured mean from the 2026-08-31 credit-g-smoke invocation, rounded up.
    "bench-smoke": (Cell(name="credit-g-default", dataset="credit_g", est_cost_usd=0.017),),
    # Prices four datasets crossing rows x columns, since wall clock tracks rows and token cost
    # tracks columns: phoneme/amazon are short/tall, jasmine/nomao are few/many columns.
    # Prices are measured means from evals/results/2026-09-01_bench-mid.jsonl, rounded up.
    "bench-mid": (
        Cell(name="phoneme-narrow-short", dataset="phoneme", est_cost_usd=0.011),
        Cell(name="jasmine-wide-short", dataset="jasmine", est_cost_usd=0.042),
        Cell(name="amazon-narrow-tall", dataset="amazon_employee_access", est_cost_usd=0.014),
        Cell(name="nomao-wide-tall", dataset="nomao", est_cost_usd=0.041),
    ),
    # Prices the remaining four datasets and tests whether the bench-mid cost model generalizes
    # out of sample; it under-predicts on all four (see DECISIONS.md 2026-09-02, first entry, for
    # the full analysis). Prices are measured means from evals/results/2026-09-02_bench-tall.jsonl,
    # rounded up.
    "bench-tall": (
        Cell(name="adult-categorical-tall", dataset="adult", est_cost_usd=0.016),
        Cell(name="bank-categorical-tall", dataset="bank_marketing", est_cost_usd=0.018),
        Cell(name="numerai-numeric-tall", dataset="numerai28_6", est_cost_usd=0.018),
        Cell(name="higgs-numeric-tall", dataset="higgs", est_cost_usd=0.025),
    ),
    # All thirteen manifest datasets. Nine prices are measured means; the four smallest
    # (australian, kc1, sylvine, kr_vs_kp) are modelled from the bench-tall refit plus one
    # residual sd, rounded up -- the cap, not the estimate, protects against them being wrong.
    # Full pass: ~52 runs at --replicates 2 --n 2, about $1.10 and 45-70 minutes.
    "full": (
        # measured
        Cell(name="phoneme", dataset="phoneme", est_cost_usd=0.011),
        Cell(name="amazon-employee-access", dataset="amazon_employee_access", est_cost_usd=0.014),
        Cell(name="adult", dataset="adult", est_cost_usd=0.016),
        Cell(name="credit-g", dataset="credit_g", est_cost_usd=0.017),
        Cell(name="bank-marketing", dataset="bank_marketing", est_cost_usd=0.018),
        Cell(name="numerai28-6", dataset="numerai28_6", est_cost_usd=0.018),
        Cell(name="higgs", dataset="higgs", est_cost_usd=0.025),
        Cell(name="nomao", dataset="nomao", est_cost_usd=0.041),
        Cell(name="jasmine", dataset="jasmine", est_cost_usd=0.042),
        # modelled, never run
        Cell(name="australian", dataset="australian", est_cost_usd=0.016),
        Cell(name="sylvine", dataset="sylvine", est_cost_usd=0.018),
        Cell(name="kc1", dataset="kc1", est_cost_usd=0.018),
        Cell(name="kr-vs-kp", dataset="kr_vs_kp", est_cost_usd=0.022),
    ),
}


def _resolve_subset(subset: str) -> tuple[Cell, ...]:
    """`SUBSETS[subset]`, or a `ValueError` listing what would have resolved."""
    if subset in SUBSETS:
        return SUBSETS[subset]
    available = ", ".join(sorted(SUBSETS))
    raise ValueError(f"unknown eval subset {subset!r}. Available subsets: {available}.")


@dataclass(frozen=True)
class PlannedRun:
    """One entry in an executed plan: which cell, which replicate, which index within it, and
    where it fell in the invocation's overall execution order."""

    cell: Cell
    replicate: int  # 1..replicates
    index: int  # 0..n-1 within (cell, replicate)
    seq: int  # 0..len(plan)-1, execution order across the whole invocation


def plan(cells: Sequence[Cell], *, replicates: int, n: int) -> list[PlannedRun]:
    """Expand `cells` into an ordered list of runs: every cell's replicate 1 first (n runs each),
    then every cell's replicate 2, and so on.

    Replicate-major, not cell-major, so that `run_eval`'s cost cap, if it binds partway through a
    plan, drops roughly the same fraction of runs from every cell instead of zeroing out whichever
    cells come later in the list. See DECISIONS.md (2026-08-28) for the ablation this fixed.
    """
    runs: list[PlannedRun] = []
    seq = 0
    for replicate in range(1, replicates + 1):
        for cell in cells:
            for index in range(n):
                runs.append(PlannedRun(cell=cell, replicate=replicate, index=index, seq=seq))
                seq += 1
    return runs


Runner = Callable[[PlannedRun, Path], PipelineState]


@dataclass
class CellTally:
    """Running counts for one cell across an invocation."""

    rows_written: int = 0
    rows_refused: int = 0  # publishable() said no -- a stub run, most likely
    runs_failed: int = 0  # raised before producing a state at all
    spend_usd: float = 0.0  # actual for completed runs, estimated for failed ones -- see `charged`
    charged_estimate_usd: float = 0.0  # the part of spend_usd nobody measured
    leakage_remediated: int = 0  # written rows where results_row()["leakage_remediated"] is True
    errored: int = 0  # written rows where results_row()["errored"] is True


@dataclass
class HarnessReport:
    """What one `run_eval` invocation did."""

    path: Path | None  # None only for a dry run -- nothing was written anywhere
    rows_written: int
    rows_refused: int
    runs_failed: int
    spend_usd: float
    # Portion of spend_usd that is a cell's est_cost_usd charged to a failed run rather than
    # measured, so a spend figure never silently mixes measured and guessed dollars.
    charged_estimate_usd: float
    stopped_early: str | None  # None | "cost cap" | "consecutive failures"
    per_cell: dict[str, CellTally]


def _live_run(artifacts_root: Path, *, transport: str, no_live: bool) -> Runner:
    """The real runner: materializes each (dataset, naming) pair once, then calls `cli._run_once`.

    Caching is keyed on (dataset, naming) rather than done once per cell, since two cells can share
    both and every run that's supposed to see the same bytes has to actually see the same bytes.

    Provenance is read here, once, before any row exists -- reading it per-run would let the
    harness's own output (writing row 0 dirties the tree) change `commit` partway through a plan
    and fragment a cell in eval-diff. See DECISIONS.md (2026-08-31).

    Uses `describe_commit`, not `git_commit`, and prints its own warning rather than
    `git_commit`'s: a null `commit` here means these rows can't be pooled with any committed cell,
    which is worth stopping for on a benchmark run and not on a one-off `ds-agents run`. Not
    behind `provenance._warned`, so a second `_live_run` in one process warns again.
    """
    # Deferred import: see the module docstring for why this cannot be a top-level import.
    from ds_agents import cli
    from ds_agents.holdout import prepare
    from ds_agents.provenance import GIT_ENV_VAR, describe_commit, warn_to_stderr
    from ds_agents.runnable import Runnable, resolve
    from ds_agents.state import RunConfig

    # `describe_commit` rather than `git_commit`, so the reason is printed here, once, before the
    # first run rather than into whatever scrollback the operator is not reading. A null `commit`
    # fragments the cell in eval-diff and is only fixable before the money is spent -- 2026-09-21's
    # `claims-repro` wrote ten rows with no commit and reported success.
    commit, commit_reason = describe_commit()
    if commit is None:
        warn_to_stderr(
            f"provenance: no commit will be recorded on these rows -- {commit_reason}. "
            f"`commit` is an eval-diff condition field, so these rows cannot be pooled with any "
            f"committed cell. Set {GIT_ENV_VAR} to a git that works, or accept it deliberately."
        )

    runnables: dict[str, Runnable] = {}
    prepared_by_key: dict[tuple[str, str], PreparedDataset] = {}

    def runner(planned_run: PlannedRun, root: Path) -> PipelineState:
        cell = planned_run.cell
        # `resolve` rather than `load_fixture`: a Cell's dataset may now name a manifest dataset,
        # and the two registries share one namespace.
        dataset = runnables.setdefault(cell.dataset, resolve(cell.dataset))
        key = (cell.dataset, cell.naming)
        if key not in prepared_by_key:
            # Once per (dataset, naming) per invocation, for the reason `materialize` already was
            # and one more: every replicate in a cell must be graded against the SAME withheld
            # rows, or the runs in it are not measuring the same question and the cell cannot be
            # pooled. `withheld` is a sibling of `input`, so it is outside every run root.
            prepared_by_key[key] = prepare(
                dataset,
                cell.naming,
                into=artifacts_root / "input" / f"{cell.dataset}-{cell.naming}",
                withheld_into=artifacts_root / "withheld" / f"{cell.dataset}-{cell.naming}",
                seed=RunConfig().random_seed,
            )
        return cli._run_once(
            dataset,
            root=root,
            prepared=prepared_by_key[key],
            commit=commit,
            transport=transport,
            no_live=no_live,
            **cell.conditions(),
        )

    return runner


def _print_plan(planned: list[PlannedRun], cells: Sequence[Cell], max_cost_usd: float) -> None:
    estimated = sum(r.cell.est_cost_usd for r in planned)
    print(
        f"plan: {len(planned)} runs across {len(cells)} cells (cells={[c.name for c in cells]})",
        file=sys.stderr,
    )
    for r in planned:
        print(
            f"  seq={r.seq:>4}  cell={r.cell.name:<24} replicate={r.replicate} index={r.index}",
            file=sys.stderr,
        )
    print(f"estimated cost: ${estimated:.4f}  (cap ${max_cost_usd:.2f})", file=sys.stderr)


def _print_report(report: HarnessReport) -> None:
    print(
        f"\ndone: {report.rows_written} rows written, {report.rows_refused} refused, "
        f"{report.runs_failed} runs failed, ${report.spend_usd:.4f} spent"
        + (
            f" (${report.charged_estimate_usd:.4f} of it estimated, from failed runs)"
            if report.charged_estimate_usd
            else ""
        )
        + (f", stopped early: {report.stopped_early}" if report.stopped_early else ""),
        file=sys.stderr,
    )
    for name, tally in report.per_cell.items():
        print(
            f"  {name:<24} written={tally.rows_written} refused={tally.rows_refused} "
            f"failed={tally.runs_failed} spend=${tally.spend_usd:.4f} "
            f"remediated={tally.leakage_remediated} errored={tally.errored}",
            file=sys.stderr,
        )
    if report.path is not None:
        print(f"results: {report.path}", file=sys.stderr)


def run_eval(
    *,
    subset: str,
    name: str,
    replicates: int = 1,
    n: int = 1,
    max_cost_usd: float = 0.50,
    out_dir: Path = RESULTS_DIR,
    artifacts_dir: Path | None = None,
    transport: str = "mcp",
    no_live: bool = False,
    dry_run: bool = False,
    today: date | None = None,
    runner: Runner | None = None,
) -> HarnessReport:
    """Run `subset` and append every publishable result to a dated JSONL file.

    `name` must match `^[a-z0-9][a-z0-9-]*$`, checked up front so a bad name is caught before
    anything is spent rather than surfacing as a write to some unintended path.

    Prints the plan and cost estimate before running anything. `dry_run` returns right after that,
    with `path=None` and every counter at zero, calling `runner` zero times.

    The cost cap is invocation-level, not per-cell: before each run, if `spend + cell.est_cost_usd`
    would exceed `max_cost_usd`, the run does not start. A run that raises is charged its cell's
    estimate instead of a measurement, since the tokens it burned are unrecoverable and guessing
    high is the safe direction; that portion is reported separately as `charged_estimate_usd`.

    Each run is wrapped in its own `try/except Exception` and counted in `runs_failed`; three
    CONSECUTIVE failures abort the invocation so a broken config can't burn the whole cap one
    exception at a time. `SystemExit` is not caught -- `cli._run_once` raises it to mean the run
    never started at all, and whatever stopped one run from starting will stop the next the same
    way.

    Every written row carries the harness's own annotations under `extra=`: `cell`, `replicate`,
    `run_index`, `eval_subset`, `eval_name`. A row `publishable()` refuses is not written but still
    increments `rows_refused`, since it changes a cell's denominator just as a written row changes
    its numerator.
    """
    if not _NAME_RE.match(name):
        raise ValueError(
            f"eval name {name!r} must match {_NAME_RE.pattern!r} (lowercase, digits, hyphens, no "
            "path separators) -- it becomes a filename"
        )

    cells = _resolve_subset(subset)
    planned = plan(cells, replicates=replicates, n=n)
    per_cell = {cell.name: CellTally() for cell in cells}

    _print_plan(planned, cells, max_cost_usd)

    if dry_run:
        print("dry run: no runner called, nothing written", file=sys.stderr)
        return HarnessReport(
            path=None,
            rows_written=0,
            rows_refused=0,
            runs_failed=0,
            spend_usd=0.0,
            charged_estimate_usd=0.0,
            stopped_early=None,
            per_cell=per_cell,
        )

    # Deferred import: see the module docstring.
    from ds_agents import cli

    today = today or date.today()
    path = out_dir / f"{today:%Y-%m-%d}_{name}.jsonl"
    if path.exists():
        print(f"note: {path} already exists, appending to it", file=sys.stderr)

    artifacts_root = artifacts_dir or Path(tempfile.mkdtemp(prefix="ds-agents-eval-"))
    if runner is None:
        runner = _live_run(artifacts_root, transport=transport, no_live=no_live)

    spend = 0.0
    charged_estimate = 0.0
    rows_written = 0
    rows_refused = 0
    runs_failed = 0
    consecutive_failures = 0
    stopped_early: str | None = None

    for planned_run in planned:
        cell = planned_run.cell
        if spend + cell.est_cost_usd > max_cost_usd:
            print(
                f"stopping: cost cap ${max_cost_usd:.2f} would be exceeded by "
                f"seq={planned_run.seq} (cell={cell.name}, spent so far ${spend:.4f})",
                file=sys.stderr,
            )
            stopped_early = "cost cap"
            break

        run_root = artifacts_root / cell.name / f"seq-{planned_run.seq:04d}"
        try:
            state = runner(planned_run, run_root)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - any run failure is counted the same way
            print(
                f"run failed (seq={planned_run.seq}, cell={cell.name}, "
                f"replicate={planned_run.replicate}, index={planned_run.index}): {exc!r}",
                file=sys.stderr,
            )
            runs_failed += 1
            per_cell[cell.name].runs_failed += 1
            # Charged its cell's estimate, not a measurement -- see run_eval's docstring.
            spend += cell.est_cost_usd
            charged_estimate += cell.est_cost_usd
            per_cell[cell.name].spend_usd += cell.est_cost_usd
            per_cell[cell.name].charged_estimate_usd += cell.est_cost_usd
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"stopping: {MAX_CONSECUTIVE_FAILURES} consecutive failures", file=sys.stderr)
                stopped_early = "consecutive failures"
                break
            continue

        consecutive_failures = 0
        spend += state.total_cost_usd
        per_cell[cell.name].spend_usd += state.total_cost_usd

        wrote = cli._append_results_row(
            state,
            path,
            extra={
                "cell": cell.name,
                "replicate": planned_run.replicate,
                "run_index": planned_run.index,
                "eval_subset": subset,
                "eval_name": name,
            },
        )
        if wrote:
            rows_written += 1
            per_cell[cell.name].rows_written += 1
            row = state.results_row()
            if row.get("leakage_remediated"):
                per_cell[cell.name].leakage_remediated += 1
            if row.get("errored"):
                per_cell[cell.name].errored += 1
        else:
            rows_refused += 1
            per_cell[cell.name].rows_refused += 1

    report = HarnessReport(
        path=path,
        rows_written=rows_written,
        rows_refused=rows_refused,
        runs_failed=runs_failed,
        spend_usd=spend,
        charged_estimate_usd=charged_estimate,
        stopped_early=stopped_early,
        per_cell=per_cell,
    )
    _print_report(report)
    return report
