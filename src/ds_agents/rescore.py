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
from ds_agents.state import PipelineState, RescoreStatus
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


RESCORE_SNIPPET = '''
import importlib, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

TARGET = {target!r}
TASK_TYPE = {task_type!r}
POSITIVE_CLASS = {positive_class!r}
SCORING = {scoring!r}
SIGN = -1.0 if SCORING.startswith("neg_") else 1.0
SEED = {seed}
SENTINEL = {sentinel!r}
SPEC = {spec!r}
SPLIT = json.loads({split_json!r})
FEATURE_CODE = {feature_code!r}
WITHHELD_PATH = {withheld_path!r}

ns = {{}}
exec(compile(FEATURE_CODE, "feature_transform.py", "exec"), ns)
transform = ns["transform"]
SOURCE_COLUMNS = list(ns["SOURCE_COLUMNS"])

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
X = df[SOURCE_COLUMNS]
scorer = get_scorer(SCORING)


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
withheld_usable = withheld[withheld[TARGET].notna()]
y_withheld = encode(withheld_usable)
n_positive = int(y_withheld.sum()) if TASK_TYPE == "binary" else None
verified = None
single_class = TASK_TYPE == "binary" and len(set(y_withheld.tolist())) < 2
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
        target=state.spec.target,
        task_type=state.spec.task_type,
        positive_class=state.spec.positive_class,
        scoring=_scoring_for(state.spec.task_type, state.spec.metric),
        seed=state.config.random_seed,
        sentinel=SEED_SENTINEL,
        spec=spec,
        split_json=inputs.split_json,
        feature_code=inputs.feature_code,
        withheld_path=str(prepared.withheld_csv),
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


def apply(state: PipelineState, outcome: RescoreOutcome) -> PipelineState:
    """Write the grader's result onto the state. Never appends a `PipelineError`."""
    return state.model_copy(
        update={
            "verified_holdout_score": outcome.verified_holdout_score,
            "rescore_status": outcome.status,
            "rescore_detail": outcome.detail,
            "refit_claim_gap": outcome.refit_claim_gap,
            "n_withheld_rows": outcome.n_withheld_rows,
        }
    )
