"""feature_eng: turn the surviving columns into a fitted, re-appliable feature matrix.

Reads `spec`, `profile`, `split_artifact`. Writes `feature_code_artifact`, `feature_summary`,
`final_features`, `dropped_features`. Tools: `read_artifact` (the split manifest), `run_python`.

Like the profiler, this is split in two. The node computes a FORCED drop set -- the target, any
column that is really just a row id, and any column under an open leakage-shaped reviewer objection
that this node is the one to act on (`config.objection_routing` decides that, via
`open_objections`, not this file) -- before the model is ever called, and the model's own proposed
drops can only add to that set, never remove from it. The model also gets to say which
profiler-flagged columns it is keeping and why. Everything downstream of the drop decision -- which
columns are numeric vs one-hot, what the medians and one-hot levels are, whether the resulting
matrix has any NaN -- is computed by a fixed snippet through `run_python`, fitted on the pinned
TRAIN rows of `split_artifact` only. Fitting on the whole frame would move the agents' own holdout
rows into the training statistic, which is precisely the `contamination` category the reviewer
exists to catch; refusing to run without a split manifest follows the profiler's precedent of
refusing rather than mislabelling.

`final_features`, `dropped_features`, and `FeatureDrop.column` all hold SOURCE column names, never
one-hot expansions like `region=north`. `results_row()` intersects `final_features` with
`planted_leakage_columns`, which are source names too; storing a dummy name here would empty that
intersection and score every run as remediated even with the leak fully present.
"""

import json
from typing import Any, Literal

from pydantic import Field

from ds_agents import split_manifest
from ds_agents.nodes._run import NodeRun
from ds_agents.state import (
    COLUMN_SCOPED_CATEGORIES,
    Contract,
    PipelineError,
    PipelineState,
)
from ds_agents.tools.llm import StructuredModel
from ds_agents.tools.protocol import ToolError, Tools

FEATURE_TIMEOUT_S = 90
MAX_ONE_HOT_LEVELS = 20
# Not 1.0: a near-unique column with a couple of accidental duplicates (a typo'd id, a re-issued
# customer number) is still an identifier in every way that matters, and 1.0 would let it slip
# through as a feature just because it wasn't quite perfectly unique.
ID_DISTINCTNESS_THRESHOLD = 0.98

# The categories whose whole point is "this column is the problem" are `COLUMN_SCOPED_CATEGORIES`,
# imported rather than restated. This file used to keep its own byte-identical copy; the two are now
# load-bearing together, because `objection_routing="by_category"` routes on the state.py set while
# the forced drop below gates on this one. Had they ever diverged, the router would have sent a run
# to feature_eng for an objection `_forced_drops` then skipped -- the same dead end this session
# removed, reintroduced one layer down. All three are folded into a forced FeatureDrop with
# reason="leakage": the schema has no separate bucket for contamination or implausible-importance,
# and none of the alternatives ("constant", "high_missing", "redundant", "other") describe why a
# reviewer objection forces a drop.

FEATURE_SNIPPET = '''
import json, os
import numpy as np
import pandas as pd

{decoder}

TARGET = {target!r}
DROP = {drop!r}
MAX_LEVELS = {max_levels}
SPLIT_MANIFEST = json.loads({split_json!r})

df = pd.read_csv(os.environ["DS_DATASET"])
# Decoded against THIS frame's length, which is what makes a manifest written for a different
# frame an exception rather than a silently short training set. The bounds filter this line used
# to carry is now the decoder's job: see ds_agents/split_manifest.py.
SPLIT = decode_split(SPLIT_MANIFEST, len(df))
train_rows = SPLIT["train"]
train = df.iloc[train_rows]

drop = sorted({{c for c in DROP if c in df.columns}} | {{TARGET}})
sources = [c for c in df.columns if c not in drop]

# Decided from TRAIN ROWS ONLY, same as the medians and one-hot levels below. Deciding this from
# the full frame would let holdout rows alone push a column's cardinality over MAX_LEVELS and get
# it silently skipped from a transform that is otherwise fitted on train -- schema contamination,
# the exact category the reviewer exists to catch.
numeric, categorical, skipped = [], [], []
for name in sources:
    col = train[name]
    if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        numeric.append(name)
    elif col.nunique(dropna=True) <= MAX_LEVELS:
        categorical.append(name)
    else:
        # Not one-hot encodable at this width. Reported, never swallowed: a column that vanishes
        # without appearing in `dropped_features` makes `final_features` disagree with the matrix.
        skipped.append(name)

# Fitted on TRAIN ROWS ONLY. A median over the whole frame moves the agents' own holdout into
# the training statistic, which is the `contamination` category the reviewer exists to catch.
medians = {{}}
for c in numeric:
    m = pd.to_numeric(train[c], errors="coerce").median()
    medians[c] = float(m) if pd.notna(m) else 0.0
levels = {{c: sorted(str(v) for v in train[c].dropna().unique()) for c in categorical}}

order = list(numeric) + [c + "=" + lv for c in categorical for lv in levels[c]]
column_source = {{c: c for c in numeric}}
for c in categorical:
    for lv in levels[c]:
        column_source[c + "=" + lv] = c

# Built by concatenation rather than an f-string: every constant is repr()'d, so the emitted
# module is a literal, brace-free, re-importable file with no data read at import time.
lines = [
    '"""Fitted feature transform emitted by feature_eng. Generated -- do not edit.',
    "",
    "Constants below were fitted on the pinned train rows only. `transform` reads no data at",
    "import time, so re-applying it to a withheld holdout yields the same columns in the same",
    "order. That property is what makes verified_holdout_score comparable to the claimed one.",
    '"""',
    "import pandas as pd",
    "",
    "TARGET = " + repr(TARGET),
    "DROPPED = " + repr(drop),
    "NUMERIC = " + repr(numeric),
    "CATEGORICAL = " + repr(categorical),
    "MEDIANS = " + repr(medians),
    "LEVELS = " + repr(levels),
    "FEATURE_ORDER = " + repr(order),
    "SOURCE_COLUMNS = " + repr(numeric + categorical),
    "COLUMN_SOURCE = " + repr(column_source),
    "",
    "",
    "def transform(df):",
    '    """Raw frame -> model matrix, float64, no NaN, columns exactly FEATURE_ORDER."""',
    "    out = pd.DataFrame(index=df.index)",
    "    for c in NUMERIC:",
    "        col = df[c] if c in df.columns else pd.Series(MEDIANS[c], index=df.index)",
    "        out[c] = pd.to_numeric(col, errors='coerce').astype('float64').fillna(MEDIANS[c])",
    "    for c in CATEGORICAL:",
    "        if c in df.columns:",
    "            col = df[c].astype('string')",
    "        else:",
    "            col = pd.Series(pd.NA, index=df.index, dtype='string')",
    "        for level in LEVELS[c]:",
    "            out[c + '=' + level] = (col == level).fillna(False).astype('float64')",
    "    for name in FEATURE_ORDER:",
    "        if name not in out.columns:",
    "            out[name] = 0.0",
    "    return out[FEATURE_ORDER]",
    "",
]
code = "\\n".join(lines) + "\\n"

# Executed here so a transform that does not run cannot reach the modeler.
ns = {{}}
exec(compile(code, "feature_transform.py", "exec"), ns)
X = ns["transform"](df)

path = os.path.join(os.environ["DS_ARTIFACTS"], "feature_transform.py")
with open(path, "w") as fh:
    fh.write(code)

print(json.dumps({{
    "n_rows": int(len(df)),
    "n_train_rows": int(len(train_rows)),
    "dropped": sorted(set(drop) - {{TARGET}} | set(skipped)),
    "final_features": numeric + categorical,
    "matrix_columns": order,
    "column_source": column_source,
    "numeric": numeric,
    "categorical": categorical,
    "skipped_high_cardinality": skipped,
    "medians": medians,
    "levels": levels,
    "n_matrix_columns": int(X.shape[1]),
    "n_nan_in_matrix": int(X.isna().sum().sum()),
    "columns_match_order": bool(list(X.columns) == order),
    "code_path": path,
}}))
'''

FEATURE_ENG_SYSTEM = """You are the feature engineering step of a tabular data science pipeline.

Some columns have already been removed for you -- the prediction target, columns that are really \
just a row id, and anything currently under an open reviewer objection. See `already_dropped` for \
what and why; nothing you say can undo one of those. You are choosing what else, if anything, to \
drop from what remains, and which profiler-flagged columns you are keeping anyway.

Rules:
- Propose a drop only if you can name the column and state the number or observation behind it. A \
drop with no justification cannot be checked against the profile.
- A profiler leakage candidate is a flag for review, not a verdict. High association is not \
automatically leakage -- a column can simply be a strong, legitimate predictor. If you keep a \
flagged column, say why it is not leakage, referencing the evidence you were given.
- Return empty lists if the remaining columns all look fine."""


class FeatureDrop(Contract):
    column: str
    reason: Literal["leakage", "identifier", "constant", "high_missing", "redundant", "other"]
    justification: str = Field(
        description="The number or observation behind the drop. Attached to a named column so it "
        "is checkable against the profile, unlike a free paragraph."
    )


class KeptColumn(Contract):
    """A column the profiler flagged that the plan keeps anyway."""

    column: str
    why_not_leakage: str


class FeaturePlan(Contract):
    drops: list[FeatureDrop] = Field(default_factory=list)
    kept_despite_flag: list[KeptColumn] = Field(default_factory=list)


def _is_float_dtype(dtype: str) -> bool:
    return "float" in dtype.lower()


def _forced_drops(state: PipelineState) -> list[FeatureDrop]:
    """The drops the node makes on its own, before the model sees anything.

    1. The target.
    2. The id-column class: >=98% distinct AND not a float dtype. Purely statistical -- a column
       named `id` that is a real feature (or a float column with high cardinality, like a price)
       must survive. This is what catches `customer_id` on the toy fixture without name heuristics.
    3. Columns named by a NOT-WITHDRAWN objection in a leakage-shaped category that ACTS on
       feature_eng. Under `objection_routing="by_category"` that includes objections the reviewer
       addressed to `modeler`; `binding_objections` resolves who acts and this node does not, so
       there is exactly one answer to that question in the codebase.

    Note the predicate in 3 is `binding_objections`, not `open_objections`. This function
    recomputes from scratch on every entry, and a `resolved` objection that stopped forcing its
    drop would put the leaked column back the next time the run returned here for any unrelated
    reason -- resolution un-fixing itself. Only `withdrawn` releases a column. See
    `PipelineState.binding_objections`, which is the one place that distinction lives, and note
    that this is its only caller in the graph.
    """
    assert state.spec is not None and state.profile is not None
    target = state.spec.target
    n_rows = state.profile.n_rows

    drops = [FeatureDrop(column=target, reason="other", justification="the prediction target")]

    for column in state.profile.columns:
        if column.name == target:
            continue
        if _is_float_dtype(column.dtype):
            continue
        if n_rows > 0 and column.n_unique >= ID_DISTINCTNESS_THRESHOLD * n_rows:
            drops.append(
                FeatureDrop(
                    column=column.name,
                    reason="identifier",
                    justification=f"{column.n_unique} distinct values in {n_rows} rows",
                )
            )

    seen = {d.column for d in drops}
    for objection in state.binding_objections("feature_eng"):
        if objection.category not in COLUMN_SCOPED_CATEGORIES:
            continue
        for column in objection.columns:
            if column in seen:
                continue
            seen.add(column)
            drops.append(
                FeatureDrop(
                    column=column,
                    reason="leakage",
                    # "not withdrawn" rather than "open": this string reaches both the snippet
                    # the model reads and `dropped_features` on the results row, and a resolved
                    # objection still binds, so calling it open here would be false.
                    justification=(
                        f"reviewer objection {objection.id} ({objection.category}), "
                        f"not withdrawn: {objection.evidence}"
                    ),
                )
            )
    return drops


def _user_message(state: PipelineState, forced: list[FeatureDrop]) -> str:
    assert state.spec is not None and state.profile is not None
    target = state.spec.target
    facts = {
        "task_description": state.task_description,
        "target": target,
        "task_type": state.spec.task_type,
        "metric": state.spec.metric,
        "n_rows": state.profile.n_rows,
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "missing_fraction": column.missing_fraction,
                "n_unique": column.n_unique,
            }
            for column in state.profile.columns
            if column.name != target
        ],
        "profiler_leakage_candidates": [
            {
                "column": candidate.column,
                "reason": candidate.reason,
                "evidence": candidate.evidence,
                "suspicion": candidate.suspicion,
            }
            for candidate in state.profile.leakage_candidates
        ],
        "already_dropped": [
            {"column": drop.column, "why": f"{drop.reason}: {drop.justification}"}
            for drop in forced
        ],
        "open_objections": [
            {
                "category": objection.category,
                "columns": objection.columns,
                "evidence": objection.evidence,
                "severity": objection.severity,
            }
            for objection in state.open_objections("feature_eng")
        ],
    }
    return json.dumps(facts, indent=2)


def _build_summary(
    result: dict[str, Any],
    forced: list[FeatureDrop],
    model_drops: list[FeatureDrop],
    kept: list[KeptColumn],
    split_artifact: str,
) -> str:
    """Assembled from the snippet's JSON and from model fields attached to a named column --
    never from a free-text summary, because `FeaturePlan` deliberately does not have one. A
    model-authored summary could claim a drop that never happened, which is the "system reports
    its own grade" problem in miniature."""
    numeric: list[str] = result.get("numeric", [])
    categorical: list[str] = result.get("categorical", [])
    levels: dict[str, list[str]] = result.get("levels", {})
    n_final = len(result.get("final_features", []))
    n_matrix = result.get("n_matrix_columns", len(result.get("matrix_columns", [])))
    n_train = result.get("n_train_rows", 0)

    reasons = {drop.column: drop for drop in forced + model_drops}
    dropped_names: list[str] = result.get("dropped", [])
    skipped = set(result.get("skipped_high_cardinality", []))

    lines = [
        f"{n_final} source columns -> {n_matrix} matrix columns, fitted on {n_train} train rows "
        f"from split {split_artifact}."
    ]

    if dropped_names:
        parts = []
        for name in dropped_names:
            drop = reasons.get(name)
            if drop is not None:
                parts.append(f"{name} [{drop.reason}: {drop.justification}]")
            elif name in skipped:
                parts.append(f"{name} [high_cardinality: too many levels to one-hot encode]")
            else:
                parts.append(name)
        lines.append(f"Dropped ({len(dropped_names)}): " + "; ".join(parts) + ".")
    else:
        lines.append("Dropped (0): none.")

    if numeric:
        lines.append(
            f"Numeric, median-imputed on train rows ({len(numeric)}): " + ", ".join(numeric) + "."
        )
    if categorical:
        levels_desc = ", ".join(f"{c} ({len(levels.get(c, []))})" for c in categorical)
        lines.append(f"One-hot, levels fitted on train rows ({len(categorical)}): {levels_desc}.")
    if kept:
        kept_desc = "; ".join(f"{k.column} -- {k.why_not_leakage}" for k in kept)
        lines.append(f"Kept despite a profiler flag ({len(kept)}): {kept_desc}.")

    return "\n".join(lines)


def feature_eng(state: PipelineState, *, tools: Tools, model: StructuredModel) -> dict[str, Any]:
    run = NodeRun("feature_eng")
    if state.spec is None:
        return run.failure("no spec: intake did not produce one", recoverable=False)
    if state.profile is None:
        return run.failure("no profile: profiler did not produce one", recoverable=False)
    if state.split_artifact is None:
        return run.failure(
            "no split_artifact: fitting medians and one-hot levels on the full frame would move "
            "the agents' own holdout rows into the training statistic -- exactly the "
            "contamination category the reviewer exists to catch, so refusing rather than "
            "fitting on an unpinned split",
            recoverable=False,
        )

    try:
        split_payload = tools.read_artifact(state.split_artifact)
    except ToolError as exc:
        return run.failure(f"could not read split manifest {state.split_artifact!r}: {exc}")
    if split_payload.truncated:
        # A truncated manifest means `train_rows` below is a subset of the pinned split, not the
        # split itself: medians and one-hot levels would be fitted on fewer rows than the modeler
        # believes, and there is no way to tell from the output alone. The modeler applies the
        # same guard to the feature code artifact it reads; this is the symmetric guard here.
        return run.failure(
            "split manifest artifact was truncated; refusing to fit on a partial split",
            recoverable=False,
        )

    known = {column.name for column in state.profile.columns} - {state.spec.target}
    forced = _forced_drops(state)
    forced_columns = {drop.column for drop in forced}

    try:
        plan = run.record(
            model.generate(
                system=FEATURE_ENG_SYSTEM,
                user=_user_message(state, forced),
                schema=FeaturePlan,
            )
        )
    except Exception as exc:  # noqa: BLE001 - a real client's own timeout/rate-limit/connection
        # errors must not crash the node out of the graph. Recoverable: the forced drops alone are
        # still a usable feature set, and a run that reaches the reviewer with only the mechanical
        # drops is an outcome to measure, not an abort.
        plan = FeaturePlan()
        plan_error = str(exc)
    else:
        plan_error = ""

    # A drop naming a column that does not exist cannot be scored against ground truth, and one
    # naming an already-forced column adds nothing -- it is folded into `reasons` by `forced`
    # taking priority, never removed by the model's say-so.
    model_drops = [
        drop for drop in plan.drops if drop.column in known and drop.column not in forced_columns
    ]
    drop_list = sorted(forced_columns | {drop.column for drop in model_drops})
    # Kept only if it actually survived the drop list -- a model claiming to keep a column that
    # was force-dropped anyway must not show up in the summary as a deliberate keep.
    kept_despite_flag = [
        kept
        for kept in plan.kept_despite_flag
        if kept.column in known and kept.column not in drop_list
    ]

    try:
        feature_run = tools.run_python(
            FEATURE_SNIPPET.format(
                decoder=split_manifest.DECODER_SRC,
                target=state.spec.target,
                drop=drop_list,
                max_levels=MAX_ONE_HOT_LEVELS,
                split_json=split_payload.content,
            ),
            timeout_s=FEATURE_TIMEOUT_S,
        )
    except ToolError as exc:
        return run.failure(f"feature engineering snippet could not run: {exc}")
    if not feature_run.ok:
        return run.failure(
            f"feature engineering snippet failed: {feature_run.stderr.strip()[-500:]}"
        )
    try:
        result = json.loads(feature_run.stdout)
    except json.JSONDecodeError as exc:
        return run.failure(f"feature engineering snippet printed no JSON: {exc}")

    errors: list[PipelineError] = []
    if plan_error:
        errors.append(
            PipelineError(
                node="feature_eng",
                message=f"feature plan unusable, continuing with forced drops only: {plan_error}",
            )
        )

    n_nan = result.get("n_nan_in_matrix", 0)
    if n_nan:
        # Refused, not reported alongside a "successful" write: the modeler would fit on a matrix
        # with NaN in it, which either crashes the candidate fit or silently corrupts it depending
        # on the estimator. No feature_code_artifact means nothing downstream can pick it up.
        errors.append(
            PipelineError(
                node="feature_eng",
                message=(
                    f"feature matrix contains {n_nan} NaN value(s) after the transform; refusing "
                    "to hand it to the modeler"
                ),
            )
        )
        update: dict[str, Any] = {"node_trace": [run.event()], "errors": errors}
        return update

    skipped = result.get("skipped_high_cardinality") or []
    if skipped:
        errors.append(
            PipelineError(
                node="feature_eng",
                message=(
                    f"columns skipped as too high-cardinality to one-hot encode at "
                    f"{MAX_ONE_HOT_LEVELS} levels: {skipped}"
                ),
            )
        )

    if len(feature_run.artifacts_written) != 1:
        # Zero means no feature_transform.py for the modeler to exec; more than one means the
        # snippet wrote something unexpected and indexing [0] would silently pick an arbitrary
        # one. Either way this is not a schema the node was built to hand off.
        return run.failure(
            f"expected exactly one feature_transform.py artifact, got "
            f"{len(feature_run.artifacts_written)}"
        )

    summary = _build_summary(result, forced, model_drops, kept_despite_flag, state.split_artifact)

    update = {
        "feature_code_artifact": feature_run.artifacts_written[0],
        "feature_summary": summary,
        "final_features": result.get("final_features", []),
        "dropped_features": result.get("dropped", []),
        "node_trace": [run.event()],
    }
    if errors:
        update["errors"] = errors
    return update
