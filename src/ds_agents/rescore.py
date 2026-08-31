"""The independent number: the modeler's model, scored on rows it never saw.

`chosen_model.claimed_holdout_score` is a claim. It is measured on `split_artifact`'s holdout,
which the profiler drew, from a strategy intake chose, over the frame the agents were mounted --
every part of it decided inside the graph. `verified_holdout_score` is the same model measured on
the rows `holdout.prepare` carved out before the graph started. The signed difference between them
is `holdout_claim_gap`, and it is the finding this half of the project exists to produce.

**No fitted model is persisted anywhere.** `ModelResult.model_artifact` has been declared and never
written since Phase 1, so scoring the withheld rows means refitting. The recipe comes from
`modeler.CANDIDATE_SPECS` with the same seed substitution the modeler used -- not from
`ModelResult.params`, which is a flat merge across every pipeline step and is lossy by
construction: `logistic_l2`'s `StandardScaler` contributes an empty dict and is invisible in it.

**The refit is a claim too, so it gets a self-check.** Before the withheld rows are touched, the
same pipeline is fit on `split["train"]` and scored on `split["holdout"]` -- the agents' own
holdout -- and compared to what the modeler said it got there. That single comparison validates the
whole chain at once: transform re-application, seed, positive-class resolution, scorer sign, split
parsing. Without it, `verified_holdout_score` is a number computed on some rows. With it, it is the
modeler's model measured on rows it never saw.

**Where this runs.** In a sandbox, through its own `LocalTools` rooted under the run, never
in-process: CLAUDE.md forbids in-process `exec` and the 2026-08-26 entry gives the reason. The
grader is not an agent, so the transport is `local` even when the run used `mcp` -- MCP exists so
agents get the standard surface, and a server subprocess per re-score is cost with no property
attached. A `SandboxError` here is caught rather than propagated: in a node it propagates so a
broken machine is not filed as an agent mistake, but here the run has already produced everything
it will, and losing the row would drop exactly the rows a table needs.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ds_agents.holdout import PreparedDataset
from ds_agents.nodes.modeler import CANDIDATE_SPECS, SEED_SENTINEL, _scoring_for
from ds_agents.state import BaselineStatus, PipelineState, RescoreStatus
from ds_agents.tools.protocol import ToolError, Tools

RESCORE_TIMEOUT_S = 300

# Absolute, and tight on purpose. Both fits run in the same sandbox on identical rows with a pinned
# `random_state` and the same stripped environment, so a disagreement is a bug in the refit and not
# float noise. If a future estimator turns out not to be bit-reproducible across forks, loosen this
# and record the measured spread -- do not silence the check.
REFIT_TOLERANCE = 1e-6


class RescoreInputs(BaseModel):
    """What the grader needs out of the run's own store, read before those tools are closed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_code: str
    split_json: str


class RescoreOutcome(BaseModel):
    """The grader's whole result, including the reason there is no number."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: RescoreStatus
    verified_holdout_score: float | None = None
    refit_agent_holdout_score: float | None = None
    refit_claim_gap: float | None = None
    n_withheld_rows: int | None = None
    n_withheld_positive: int | None = None
    detail: str = Field(default="", description="Free text. Never parsed.")


class BaselineOutcome(BaseModel):
    """The yardstick's whole result, including the reason there is no yardstick."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: BaselineStatus
    zero_score: float | None = None
    unit_score: float | None = None
    recipe: str = ""
    detail: str = Field(default="", description="Free text. Never parsed.")


_SNIPPET_PRELUDE = """
import json, os
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer

TARGET = {target!r}
TASK_TYPE = {task_type!r}
POSITIVE_CLASS = {positive_class!r}
SCORING = {scoring!r}
SIGN = -1.0 if SCORING.startswith("neg_") else 1.0
SEED = {seed}
SPLIT = json.loads({split_json!r})
WITHHELD_PATH = {withheld_path!r}

# The frame the agents were mounted on. Its row ids are what SPLIT indexes.
df = pd.read_csv(os.environ["DS_DATASET"])
withheld = pd.read_csv(WITHHELD_PATH)

# Resolved on the AGENT frame, then applied to both. Resolving it on the withheld rows would flip
# the label mapping whenever a class happens to be absent there, turning roc_auc into 1-auc with
# nothing in the output to explain it.
positive = None
positive_mismatch = None
if TASK_TYPE == "binary":
    classes = sorted(str(v) for v in df[TARGET].dropna().unique())
    if POSITIVE_CLASS is not None and POSITIVE_CLASS in classes:
        positive = POSITIVE_CLASS
    else:
        positive = classes[-1] if classes else None
        if POSITIVE_CLASS is not None:
            positive_mismatch = {{"requested": POSITIVE_CLASS, "used": positive}}


def encode(frame):
    raw = frame[TARGET]
    if TASK_TYPE == "binary":
        return (raw.astype("string") == str(positive)).astype("int64")
    if TASK_TYPE == "multiclass":
        return raw.astype("string")
    return pd.to_numeric(raw, errors="coerce")


y = encode(df)
usable = set(np.flatnonzero(df[TARGET].notna().to_numpy()).tolist())


def keep(ids):
    return [i for i in ids if 0 <= i < len(df) and i in usable]


train_rows = keep(SPLIT["train"])
holdout_rows = keep(SPLIT["holdout"])
scorer = get_scorer(SCORING)

withheld_usable = withheld[withheld[TARGET].notna()]
y_withheld = encode(withheld_usable)
n_positive = int(y_withheld.sum()) if TASK_TYPE == "binary" else None
single_class = TASK_TYPE == "binary" and len(set(y_withheld.tolist())) < 2
"""
"""Everything the grader's two snippets share, and the reason they share it rather than each
carrying a copy.

The positive class is resolved here, once. A copy-pasted second resolution that drifted would flip
one score to `1 - auc` and leave the other alone, and the normalised number built from the two
would then be comparing two different label mappings with nothing in the output to say so. Same
argument for `encode`, for `keep`, and for which rows count as usable.

Ends at `single_class`: everything above is a fact about the data, and everything below it in
either body is a model.
"""

_RESCORE_BODY = '''
import importlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

SENTINEL = {sentinel!r}
SPEC = {spec!r}
FEATURE_CODE = {feature_code!r}

ns = {{}}
exec(compile(FEATURE_CODE, "feature_transform.py", "exec"), ns)
transform = ns["transform"]
SOURCE_COLUMNS = list(ns["SOURCE_COLUMNS"])
X = df[SOURCE_COLUMNS]


def build():
    """Rebuilt from CANDIDATE_SPECS exactly as the modeler builds it. Same steps, same order, same
    seed substitution -- if this drifts, the self-check below is what says so."""
    steps = [("features", FunctionTransformer(transform, validate=False))]
    for name, path, params in SPEC:
        module, cls = path.split(":")
        kwargs = {{k: (SEED if v == SENTINEL else v) for k, v in params.items()}}
        steps.append((name, getattr(importlib.import_module(module), cls)(**kwargs)))
    return Pipeline(steps)


# The self-check FIRST: the agents' own holdout, which the run already reported a number for.
pipe = build()
pipe.fit(X.iloc[train_rows], y.iloc[train_rows])
refit_agent = SIGN * float(scorer(pipe, X.iloc[holdout_rows], y.iloc[holdout_rows]))

# The withheld frame is scored by the SAME fitted pipeline. Refitting for it would score a
# different model from the one the self-check just validated.
verified = None
if not single_class and len(withheld_usable) > 0:
    verified = SIGN * float(scorer(pipe, withheld_usable[SOURCE_COLUMNS], y_withheld))

print(json.dumps({{
    "verified_holdout_score": verified,
    "refit_agent_holdout_score": refit_agent,
    "n_withheld_rows": int(len(withheld_usable)),
    "n_withheld_positive": n_positive,
    "single_class_holdout": bool(single_class),
    "positive_class": None if positive is None else str(positive),
    "positive_class_mismatch": positive_mismatch,
    "n_train_rows": len(train_rows),
    "n_agent_holdout_rows": len(holdout_rows),
}}))
'''

RESCORE_SNIPPET = _SNIPPET_PRELUDE + _RESCORE_BODY


BASELINE_TIMEOUT_S = 900
"""The unit point's own budget, separate from RESCORE_TIMEOUT_S because it is a separate process.

Sharing one budget across both fits would make the SAME code return `ok` on a fast machine and
`snippet_failed` on a slow one, depending on how much of the budget the refit happened to spend
first. Longer than the re-scorer's because a 200-tree forest on 98k rows is the slowest thing the
grader does, and because nothing downstream of it is waiting on a model.
"""

BASELINE_RECIPE = "rf-v1"
"""Version string for the unit point's recipe, written onto every row it produces.

On the ROW rather than only in git, because two runs graded against different yardsticks must not
be pooled into one normalised distribution, and a reader holding a results file cannot see a
commit. Bump it whenever `_RF` changes.
"""

# OURS. Not AMLB's, and the difference is recorded rather than smoothed over. AMLB publishes the
# CONVENTION -- normalise from a constant class-prior predictor to a tuned RandomForest -- but its
# per-dataset numbers live in an object store at openml1.win.tue.nl behind a self-signed
# certificate, so the grid cannot be re-fetched or re-verified from this repo. Every hyperparameter
# below was therefore chosen here, which makes this the one number in the benchmark path that is
# not traceable to something external, and it is labelled that way everywhere it is described.
#
# Fixed, not tuned: a tuned baseline needs a search space and a validation protocol, and each of
# those is another invented choice nobody can cite. `n_estimators=200` rather than the more usual
# 500 is a COST decision and not a statistical one -- thirteen datasets up to 98k rows, and the
# unit point is a yardstick rather than a competitor. `n_jobs=1` because the sandbox child owns its
# process group and is killed with `killpg`; joblib workers inside it make the timeout semantics
# murky for no benefit here.
_RF: dict[str, Any] = {
    "n_estimators": 200,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "n_jobs": 1,
    "random_state": SEED_SENTINEL,
}

BASELINE_SPECS: dict[str, list[tuple[str, str, dict[str, Any]]]] = {
    "binary": [("model", "sklearn.ensemble:RandomForestClassifier", _RF)],
    "multiclass": [("model", "sklearn.ensemble:RandomForestClassifier", _RF)],
    "regression": [("model", "sklearn.ensemble:RandomForestRegressor", _RF)],
}

ZERO_SPECS: dict[str, list[tuple[str, str, dict[str, Any]]]] = {
    "binary": [("model", "sklearn.dummy:DummyClassifier", {"strategy": "prior"})],
    "multiclass": [("model", "sklearn.dummy:DummyClassifier", {"strategy": "prior"})],
    "regression": [("model", "sklearn.dummy:DummyRegressor", {"strategy": "mean"})],
}

_BASELINE_BODY = '''
import importlib, traceback
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SENTINEL = {sentinel!r}
ZERO_SPEC = {zero_spec!r}
UNIT_SPEC = {unit_spec!r}

# NOT the agents' feature_code_artifact. The baseline sees the RAW mounted frame -- every column
# but the target, including any the agents chose to drop and any feature_eng skipped for exceeding
# its one-hot cap. That independence is the whole point: a yardstick that inherits the decisions it
# is measuring cannot say whether those decisions helped. It is also exactly why
# baseline_normalised_score is suppressed on a dataset with a planted leak, because on such a
# dataset the baseline kept the trap.
SOURCE = [c for c in df.columns if c != TARGET]
train_frame = df.iloc[train_rows]

status = "ok"
detail = ""
zero_score = None
unit_score = None
numeric, categorical = [], []


def as_source(frame):
    """Raw frame -> exactly `numeric + categorical`, dtypes settled.

    Applied identically to the train rows and the withheld rows. Casting here rather than inside
    the ColumnTransformer means the fitted encoder never sees a mixed-dtype object column, which is
    where OrdinalEncoder's errors stop being readable.
    """
    out = pd.DataFrame(index=frame.index)
    for c in numeric:
        out[c] = pd.to_numeric(frame[c], errors="coerce").astype("float64")
    for c in categorical:
        # NaN becomes its own level rather than a missing value: on a categorical, "absent" is
        # information, and folding it in with genuinely-unseen levels would merge two facts.
        out[c] = frame[c].astype("string").fillna("__missing__").astype("object")
    return out


def make_encoder():
    return ColumnTransformer(
        transformers=[
            # Median, not mean, for the reason feature_eng gives: one -999 sentinel in an OpenML
            # column moves a mean and does not move a median. keep_empty_features holds the matrix
            # at the same width when a column is all-NaN on the train rows and not on the others.
            ("num", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            (
                # Ordinal and not one-hot precisely because a level present in the withheld rows
                # and absent from train must encode to SOMETHING, and to a value no fitted level
                # occupies -- fitted codes are 0..k-1, so -1 is unoccupied by construction. It
                # imposes a false ordering on a nominal code, which is a known and deliberate
                # property: it makes the unit point a FLOOR rather than a ceiling.
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype="float64",
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build(spec):
    steps = [("encode", make_encoder())]
    for name, path, params in spec:
        module, cls = path.split(":")
        kwargs = {{k: (SEED if v == SENTINEL else v) for k, v in params.items()}}
        steps.append((name, getattr(importlib.import_module(module), cls)(**kwargs)))
    return Pipeline(steps)


if not SOURCE:
    status = "empty_source_matrix"
    detail = "the mounted frame carries no column but the target"
else:
    # Decided on the TRAIN ROWS ONLY, by dtype -- the same rule feature_eng uses, minus its
    # one-hot cap. A high-cardinality INTEGER id column is numeric here and passes through; a
    # high-cardinality STRING column is ordinal-encoded and kept where feature_eng would skip it.
    for name in SOURCE:
        col = train_frame[name]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            numeric.append(name)
        else:
            categorical.append(name)

    y_train = y.iloc[train_rows]
    if TASK_TYPE != "regression" and len(set(y_train.tolist())) < 2:
        status = "single_class_train"
        detail = "the agents' train split carries one class, so neither point can be fit"
    else:
        X_train = as_source(train_frame)
        X_withheld = as_source(withheld_usable)
        try:
            zero = build(ZERO_SPEC)
            zero.fit(X_train, y_train)
            zero_score = SIGN * float(scorer(zero, X_withheld, y_withheld))
        except Exception:
            status = "zero_point_failed"
            detail = traceback.format_exc()[-500:]
        try:
            unit = build(UNIT_SPEC)
            unit.fit(X_train, y_train)
            unit_score = SIGN * float(scorer(unit, X_withheld, y_withheld))
        except Exception:
            # The status this whole separate process exists for. `zero_score` above is KEPT: it was
            # measured, and discarding it would hide that the scale has a floor and no ceiling.
            status = "unit_point_failed"
            detail = traceback.format_exc()[-500:]

print(json.dumps({{
    "status": status,
    "zero_score": zero_score,
    "unit_score": unit_score,
    "n_numeric": len(numeric),
    "n_categorical": len(categorical),
    "n_train_rows": len(train_rows),
    "detail": detail,
}}))
'''

BASELINE_SNIPPET = _SNIPPET_PRELUDE + _BASELINE_BODY


def read_inputs(state: PipelineState, tools: Tools) -> RescoreInputs | RescoreOutcome:
    """Pull the feature transform and the split manifest out of the RUN's store.

    Called while the run's tools are still open, because after `tools.close()` there is nothing to
    read them from. Returns an outcome rather than raising, so every reason the grader cannot run
    lands on the row as a status instead of as an exception nobody recorded.
    """
    if state.feature_code_artifact is None:
        return RescoreOutcome(status="no_feature_code", detail="feature_eng produced no artifact")
    if state.split_artifact is None:
        return RescoreOutcome(status="no_split", detail="profiler produced no split manifest")
    try:
        feature = tools.read_artifact(state.feature_code_artifact)
        split = tools.read_artifact(state.split_artifact)
    except ToolError as exc:
        return RescoreOutcome(status="no_feature_code", detail=f"could not read artifact: {exc}")
    # The same refusal the modeler makes, for the same reason: a truncated transform either raises
    # or execs a shorter-than-real module, and a truncated manifest scores on rows that are not the
    # split. Either would produce a number, which is worse than producing none.
    if feature.truncated:
        return RescoreOutcome(status="no_feature_code", detail="feature code artifact truncated")
    if split.truncated:
        return RescoreOutcome(status="no_split", detail="split manifest truncated")
    return RescoreInputs(feature_code=feature.content, split_json=split.content)


def _preconditions(state: PipelineState, prepared: PreparedDataset) -> RescoreOutcome | None:
    """Every reason there is nothing to grade, each with its own name."""
    if prepared.withheld_csv is None or prepared.withheld_fraction == 0.0:
        return RescoreOutcome(status="no_withheld_holdout")
    if state.spec is None:
        return RescoreOutcome(status="no_spec", detail="intake produced no spec")
    if state.chosen_model is None:
        return RescoreOutcome(status="no_model", detail="no candidate was promoted")
    if not state.final_features:
        return RescoreOutcome(status="empty_matrix", detail="final_features is empty")
    specs = CANDIDATE_SPECS.get(state.spec.task_type, {})
    if state.chosen_model.name not in specs:
        return RescoreOutcome(
            status="unknown_model_spec",
            detail=(
                f"{state.chosen_model.name!r} is not in CANDIDATE_SPECS[{state.spec.task_type!r}]"
            ),
        )
    return None


def rescore(
    state: PipelineState,
    prepared: PreparedDataset,
    inputs: RescoreInputs,
    *,
    root: Path,
) -> RescoreOutcome:
    """Refit the promoted model and score it on the withheld rows.

    `root` is the grader's own directory, and the tools built on it are the grader's own store --
    separate from the run's, so nothing written here is reachable by anything that already ran.
    """
    from ds_agents.tools.local import LocalTools

    precondition = _preconditions(state, prepared)
    if precondition is not None:
        return precondition
    assert state.spec is not None and state.chosen_model is not None
    assert prepared.withheld_csv is not None

    spec = CANDIDATE_SPECS[state.spec.task_type][state.chosen_model.name]
    code = RESCORE_SNIPPET.format(
        **_snippet_kwargs(state, prepared, inputs),
        spec=spec,
        feature_code=inputs.feature_code,
    )

    tools = LocalTools(root, dataset_path=prepared.agent_csv, dataset_id=state.dataset_id)
    try:
        result = tools.run_python(code, timeout_s=RESCORE_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 -- see the module docstring: caught, never propagated
        return RescoreOutcome(status="sandbox_error", detail=f"{type(exc).__name__}: {exc}")
    finally:
        tools.close()

    if not result.ok:
        return RescoreOutcome(
            status="snippet_failed", detail=(result.stderr or result.stdout or "")[-500:]
        )
    try:
        payload: dict[str, Any] = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return RescoreOutcome(
            status="snippet_failed", detail=f"no JSON on stdout: {result.stdout[-500:]}"
        )
    return _outcome(state, payload)


def _outcome(state: PipelineState, payload: dict[str, Any]) -> RescoreOutcome:
    """Turn the snippet's numbers into a status, applying the self-check."""
    assert state.chosen_model is not None
    refit = payload.get("refit_agent_holdout_score")
    claimed = state.chosen_model.claimed_holdout_score
    gap = None if (refit is None or claimed is None) else refit - claimed

    if payload.get("single_class_holdout"):
        return RescoreOutcome(
            status="single_class_holdout",
            refit_agent_holdout_score=refit,
            refit_claim_gap=gap,
            n_withheld_rows=payload.get("n_withheld_rows"),
            n_withheld_positive=payload.get("n_withheld_positive"),
            detail="the withheld rows carry one class, so the metric is undefined on them",
        )

    common = {
        "verified_holdout_score": payload.get("verified_holdout_score"),
        "refit_agent_holdout_score": refit,
        "refit_claim_gap": gap,
        "n_withheld_rows": payload.get("n_withheld_rows"),
        "n_withheld_positive": payload.get("n_withheld_positive"),
    }
    if gap is None or abs(gap) > REFIT_TOLERANCE:
        # The score is KEPT. Deleting it would hide the finding; the status is what says the number
        # must not be pooled, and the detail carries both sides so a reader can see the size of it.
        return RescoreOutcome(
            status="refit_mismatch",
            detail=(
                f"refit scored {refit} on the agents' holdout against a claimed {claimed}; "
                f"the refit is not the model that produced the claim"
            ),
            **common,
        )
    mismatch = payload.get("positive_class_mismatch")
    return RescoreOutcome(
        status="ok",
        detail="" if not mismatch else f"positive_class fell back: {mismatch}",
        **common,
    )


def _snippet_kwargs(
    state: PipelineState,
    prepared: PreparedDataset,
    inputs: RescoreInputs,
) -> dict[str, Any]:
    """The prelude's substitutions, built once so both snippets are given the same facts.

    Extra keys are harmless -- `str.format(**kwargs)` only complains about keys the string asks for
    and the dict lacks -- so each body adds its own on top of these rather than assembling a second
    dict that could disagree about the seed or the scorer.
    """
    assert state.spec is not None
    assert prepared.withheld_csv is not None
    return {
        "target": state.spec.target,
        "task_type": state.spec.task_type,
        "positive_class": state.spec.positive_class,
        "scoring": _scoring_for(state.spec.task_type, state.spec.metric),
        "seed": state.config.random_seed,
        "sentinel": SEED_SENTINEL,
        "split_json": inputs.split_json,
        "withheld_path": str(prepared.withheld_csv),
    }


def _baseline_preconditions(
    prepared: PreparedDataset,
    outcome: RescoreOutcome,
) -> BaselineOutcome | None:
    """Every reason there is no scale to place the run on, each with its own name."""
    if prepared.withheld_csv is None or prepared.withheld_fraction == 0.0:
        return BaselineOutcome(status="no_withheld_holdout")
    # The coupling that genuinely exists, stated rather than hidden: the baseline is fit on the
    # agents' `split["train"]` and scored on the same withheld rows, so nearly every reason the
    # re-scorer could not produce a number is also a reason this cannot. `verified_holdout_score`
    # is the single condition that covers all of them, and the status carries which one it was so a
    # reader never has to join two columns to find out.
    if outcome.verified_holdout_score is None:
        return BaselineOutcome(
            status="rescore_unavailable", detail=f"rescore_status={outcome.status}"
        )
    return None


def baseline_precondition(
    prepared: PreparedDataset,
    outcome: RescoreOutcome,
) -> BaselineOutcome:
    """The reason there is no scale, for a caller that already knows there is no score.

    Exists so the no-feature-code and no-split paths get the SAME precedence as every other path
    -- a fixture reads `no_withheld_holdout` rather than `rescore_unavailable`, because "this
    dataset withholds nothing" is the more specific and more useful fact. Never returns `None`:
    reaching it at all means `verified_holdout_score` is absent.
    """
    precondition = _baseline_preconditions(prepared, outcome)
    assert precondition is not None
    return precondition


def baseline(
    state: PipelineState,
    prepared: PreparedDataset,
    inputs: RescoreInputs,
    outcome: RescoreOutcome,
    *,
    root: Path,
) -> BaselineOutcome:
    """Fit the two reference points and score them on the same withheld rows as the run.

    A SECOND process, with its own store and its own timeout, and that is the design rather than an
    accident of layout. In one process a RandomForest that dies on a wide frame and a refit that
    dies are the same exit code, and `verified_holdout_score` -- which is the number this half of
    the project exists to produce -- would go down with the yardstick. Two processes make
    `unit_point_failed` a fact the OS records rather than one reconstructed from tagged stdout.

    Like `rescore`, a sandbox failure here is caught and never propagated.
    """
    from ds_agents.tools.local import LocalTools

    precondition = _baseline_preconditions(prepared, outcome)
    if precondition is not None:
        return precondition
    assert state.spec is not None

    task_type = state.spec.task_type
    code = BASELINE_SNIPPET.format(
        **_snippet_kwargs(state, prepared, inputs),
        zero_spec=ZERO_SPECS[task_type],
        unit_spec=BASELINE_SPECS[task_type],
    )

    tools = LocalTools(root, dataset_path=prepared.agent_csv, dataset_id=state.dataset_id)
    try:
        result = tools.run_python(code, timeout_s=BASELINE_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 -- see `rescore`: caught, never propagated
        return BaselineOutcome(status="sandbox_error", detail=f"{type(exc).__name__}: {exc}")
    finally:
        tools.close()

    if not result.ok:
        return BaselineOutcome(
            status="snippet_failed", detail=(result.stderr or result.stdout or "")[-500:]
        )
    try:
        payload: dict[str, Any] = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return BaselineOutcome(
            status="snippet_failed", detail=f"no JSON on stdout: {result.stdout[-500:]}"
        )
    return _baseline_outcome(payload)


def _baseline_outcome(payload: dict[str, Any]) -> BaselineOutcome:
    """Turn the snippet's numbers into an outcome.

    The snippet names its own status, because the distinctions it draws -- which of the two fits
    raised, and whether there was a matrix to fit at all -- are only visible from inside it. What
    is decided here is what survives: `unit_point_failed` keeps the zero point, and `recipe` is
    stamped on any row where the unit point was actually attempted.
    """
    status: BaselineStatus = payload.get("status", "snippet_failed")
    attempted = status in ("ok", "unit_point_failed", "zero_point_failed")
    return BaselineOutcome(
        status=status,
        zero_score=payload.get("zero_score"),
        unit_score=payload.get("unit_score"),
        recipe=BASELINE_RECIPE if attempted else "",
        detail=payload.get("detail", ""),
    )


def apply(
    state: PipelineState,
    outcome: RescoreOutcome,
    baseline_outcome: BaselineOutcome | None = None,
) -> PipelineState:
    """Write the grader's result onto the state. Never appends a `PipelineError`."""
    update: dict[str, Any] = {
        "verified_holdout_score": outcome.verified_holdout_score,
        "rescore_status": outcome.status,
        "rescore_detail": outcome.detail,
        "refit_claim_gap": outcome.refit_claim_gap,
        "n_withheld_rows": outcome.n_withheld_rows,
    }
    if baseline_outcome is not None:
        update |= {
            "baseline_zero_score": baseline_outcome.zero_score,
            "baseline_unit_score": baseline_outcome.unit_score,
            "baseline_status": baseline_outcome.status,
            "baseline_detail": baseline_outcome.detail,
            "baseline_recipe": baseline_outcome.recipe,
        }
    return state.model_copy(update=update)
