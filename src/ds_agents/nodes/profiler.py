"""profiler: describe the dataset and nominate columns that may encode the answer.

Reads `spec`, `dataset_id`, `config`. Writes `profile`, `split_artifact`.
Tools: `run_python`, `write_artifact`.

The node is deliberately split in two. Mechanical statistics -- dtypes, missing fractions,
cardinality, class balance, each column's association with the target -- are computed by a fixed
snippet through `run_python`. A model adds nothing to `df.nunique()` except cost and a way to be
wrong. What the model does is the part the eval actually measures: looking at the numbers and
saying which of them mean "this column encodes the answer" rather than "this column is a genuinely
strong predictor". The toy fixture contains one of each on purpose, plus an id column whose
association is elevated and meaningless, so the distinction is not free.

`split_artifact` is pinned here and never rewritten. Note what it is and is not: this is the split
the AGENTS see. The harness's independent holdout, the one behind `verified_holdout_score`, is
withheld before the graph ever starts and is not in this manifest.
"""

import json
from typing import Any

from pydantic import ValidationError

from ds_agents.nodes._run import NodeRun
from ds_agents.state import (
    ColumnProfile,
    Contract,
    LeakageCandidate,
    PipelineError,
    PipelineState,
    ProfileReport,
)
from ds_agents.tools.llm import StructuredModel
from ds_agents.tools.protocol import ToolError, Tools

PROFILE_TIMEOUT_S = 60
HOLDOUT_FRACTION = 0.2
N_FOLDS = 5

# `temporal` and `grouped` are valid on `TaskSpec` but the snippet below does not implement them.
# Running a random split and labelling the manifest "grouped" would put the same entity on both
# sides of the partition while the file claims otherwise, which is precisely the contamination
# category that is supposed to be falsifiable against this manifest.
SUPPORTED_SPLIT_STRATEGIES = frozenset({"random", "stratified"})

PROFILE_SNIPPET = '''
import json, os
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

TARGET = {target!r}
df = pd.read_csv(os.environ["DS_DATASET"])


def discretize(series):
    """Coarse bins so mutual information is defined for numeric columns too.

    Missing values become their own label rather than being dropped. They are not noise: a
    column that is missing exactly when the target is one is a leak, and dropping the NaNs
    would make it look like the cleanest column in the table.
    """
    if series.dtype.kind in "ifc" and series.nunique(dropna=True) > 12:
        binned = pd.qcut(series.rank(method="first"), 10, labels=False, duplicates="drop")
        return binned.astype("string").fillna("<MISSING>")
    return series.astype("string").fillna("<MISSING>")


columns = []
for name in df.columns:
    col = df[name]
    columns.append(
        {{
            "name": name,
            "dtype": str(col.dtype),
            "missing_fraction": float(col.isna().mean()),
            "n_unique": int(col.nunique(dropna=True)),
            "sample_values": [str(v) for v in col.dropna().unique()[:5]],
        }}
    )

target_binned = discretize(df[TARGET])
association = {{}}
association_errors = {{}}
for name in df.columns:
    if name == TARGET:
        continue
    try:
        association[name] = round(
            float(normalized_mutual_info_score(target_binned, discretize(df[name]))), 4
        )
    except Exception as exc:
        # Reported, never swallowed. A column whose association silently reads `null` is a
        # column the model cannot flag, which is indistinguishable from a clean one.
        association[name] = None
        association_errors[name] = f"{{type(exc).__name__}}: {{exc}}"

counts = df[TARGET].value_counts(normalize=True, dropna=False)
balance = {{str(k): round(float(v), 4) for k, v in counts.items()}} if len(counts) <= 20 else {{}}

print(json.dumps({{
    "n_rows": int(len(df)),
    "n_columns": int(df.shape[1]),
    "columns": columns,
    "target_balance": balance,
    "target_association": association,
    "target_association_errors": association_errors,
}}))
'''

SPLIT_SNIPPET = """
import json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

TARGET = {target!r}
SEED = {seed}
STRATEGY = {strategy!r}
HOLDOUT_FRACTION = {holdout_fraction}
N_FOLDS = {n_folds}
df = pd.read_csv(os.environ["DS_DATASET"])
rows = np.arange(len(df))
y = df[TARGET]
stratify = y if STRATEGY == "stratified" and y.nunique(dropna=True) <= 20 else None

train_rows, holdout_rows = train_test_split(
    rows, test_size=HOLDOUT_FRACTION, random_state=SEED, stratify=stratify
)
splitter = (
    StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    if stratify is not None
    else KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
)
y_train = y.iloc[train_rows]
folds = [
    {{"train": train_rows[tr].tolist(), "valid": train_rows[va].tolist()}}
    for tr, va in splitter.split(train_rows, y_train if stratify is not None else None)
]

manifest = {{
    "strategy": STRATEGY,
    "seed": SEED,
    "target": TARGET,
    "n_rows": int(len(df)),
    "train": sorted(int(i) for i in train_rows),
    "holdout": sorted(int(i) for i in holdout_rows),
    "folds": folds,
}}
path = os.path.join(os.environ["DS_ARTIFACTS"], "split_manifest.json")
with open(path, "w") as fh:
    json.dump(manifest, fh)
print(path)
"""

PROFILER_SYSTEM = """You are the profiling step of a tabular data science pipeline.

You are given column statistics and, for each column, its normalized mutual information with the \
target. Nominate the columns that may encode the target -- that is, columns whose value would not \
be knowable at prediction time, or that are derived from the answer.

Rules:
- Flag a column only if you can state the number or observation behind it. Guesses are worse than \
silence here, because a false alarm costs as much as a miss.
- High association is not automatically leakage. A column can simply be a strong, legitimate \
predictor. An identifier with one distinct value per row shows elevated association purely \
because it memorises the rows, and predicts nothing.
- You are flagging for review, not deciding. Something else decides what to drop.
- Return an empty list if nothing warrants a flag."""


class LeakageNomination(Contract):
    """What the profiler asks the model for."""

    candidates: list[LeakageCandidate] = []
    notes: str = ""


def _user_message(state: PipelineState, stats: dict[str, Any]) -> str:
    assert state.spec is not None
    facts = {
        "task_description": state.task_description,
        "target": state.spec.target,
        "task_type": state.spec.task_type,
        "n_rows": stats.get("n_rows"),
        "target_balance": stats.get("target_balance", {}),
        "columns": [
            {
                **{k: v for k, v in column.items() if k != "sample_values"},
                "sample_values": column.get("sample_values", []),
                "mutual_info_with_target": stats.get("target_association", {}).get(column["name"]),
            }
            for column in stats.get("columns", [])
            if column["name"] != state.spec.target
        ],
    }
    return json.dumps(facts, indent=2)


def profiler(state: PipelineState, *, tools: Tools, model: StructuredModel) -> dict[str, Any]:
    run = NodeRun("profiler")
    if state.spec is None:
        return run.failure("no spec: intake did not produce one", recoverable=False)

    try:
        stats_run = tools.run_python(
            PROFILE_SNIPPET.format(target=state.spec.target), timeout_s=PROFILE_TIMEOUT_S
        )
    except ToolError as exc:
        return run.failure(f"profiling snippet could not run: {exc}", recoverable=False)
    if not stats_run.ok:
        return run.failure(f"profiling snippet failed: {stats_run.stderr.strip()[-500:]}")
    try:
        stats = json.loads(stats_run.stdout)
    except json.JSONDecodeError as exc:
        return run.failure(f"profiling snippet printed no JSON: {exc}")

    try:
        report = ProfileReport(
            n_rows=stats["n_rows"],
            n_columns=stats["n_columns"],
            columns=[ColumnProfile(**column) for column in stats["columns"]],
            target_balance=stats["target_balance"],
        )
    except (KeyError, ValidationError) as exc:
        return run.failure(f"profiling output does not fit ProfileReport: {exc}")

    # The split is pinned before anything downstream runs, so a later contamination objection is
    # falsifiable against a manifest that existed first.
    split_artifact: str | None = None
    split_fatal = False
    if state.spec.split_strategy not in SUPPORTED_SPLIT_STRATEGIES:
        split_error = (
            f"split_strategy={state.spec.split_strategy!r} is not implemented; refusing to write "
            f"a manifest labelled with a strategy that did not run"
        )
        split_fatal = True
    else:
        try:
            split_run = tools.run_python(
                SPLIT_SNIPPET.format(
                    target=state.spec.target,
                    seed=state.config.random_seed,
                    strategy=state.spec.split_strategy,
                    holdout_fraction=HOLDOUT_FRACTION,
                    n_folds=N_FOLDS,
                ),
                timeout_s=PROFILE_TIMEOUT_S,
            )
        except ToolError as exc:
            split_error = str(exc)
        else:
            split_error = split_run.stderr.strip()[-500:]
            if split_run.ok and split_run.artifacts_written:
                split_artifact = split_run.artifacts_written[0]

    known = {column.name for column in report.columns}
    try:
        nomination = run.record(
            model.generate(
                system=PROFILER_SYSTEM,
                user=_user_message(state, stats),
                schema=LeakageNomination,
            )
        )
        candidates = nomination.candidates
    except Exception as exc:  # noqa: BLE001 - see below
        # Deliberately broad. A real client raises its own timeout, rate-limit, and connection
        # errors, and an uncaught one crashes the node with no PipelineError and no NodeEvent --
        # which makes the cost table read low by exactly the runs that failed and drops the row
        # for the dataset entirely. Recoverable: the mechanical profile is still worth having, and
        # a run that reaches the reviewer with no candidates is an outcome to measure, not an abort.
        candidates = []
        nomination_error = str(exc)
    else:
        nomination_error = ""

    # A candidate naming a column that does not exist cannot be scored against ground truth, and
    # counting it would inflate false alarms with the model's typos rather than its judgement.
    report.leakage_candidates = [
        candidate
        for candidate in candidates
        if candidate.column in known and candidate.column != state.spec.target
    ]

    update: dict[str, Any] = {"profile": report, "node_trace": [run.event()]}
    if split_artifact is not None:
        update["split_artifact"] = split_artifact
    errors: list[PipelineError] = []
    association_errors = stats.get("target_association_errors", {})
    if association_errors:
        errors.append(
            PipelineError(
                node="profiler",
                message=f"association could not be computed for: {sorted(association_errors)}",
            )
        )
    if split_artifact is None:
        # An unimplemented strategy is not recoverable: nothing downstream can produce a
        # trustworthy score without a partition that matches what the spec asked for.
        errors.append(
            PipelineError(
                node="profiler",
                message=f"split manifest not written: {split_error}",
                recoverable=not split_fatal,
            )
        )
    if nomination_error:
        errors.append(
            PipelineError(
                node="profiler",
                message=f"leakage nomination unusable, continuing with none: {nomination_error}",
            )
        )
    if errors:
        update["errors"] = errors
    return update
