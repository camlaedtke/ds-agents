"""modeler: fit the declared candidate models on the pinned split and let a model pick one.

Reads `spec`, `feature_code_artifact`, `split_artifact`, `config.random_seed`, `config.run_id`,
`open_objections("modeler")`, `final_features`. Note that under
`config.objection_routing="by_category"` that call excludes column-scoped objections even when the
reviewer addressed them here, deliberately: this node's only lever is which candidate to promote,
so a column complaint in its prompt can only produce a spurious response that reads like
remediation in a trace and is nothing of the kind. Writes `candidates`, `chosen_model`,
`importance_artifact`, `top_importances`. Tools: `read_artifact`, `run_python`, `log_metric`.

Every candidate is fit and scored entirely by code -- cross-validated on the pinned folds, then
refit on the pinned train rows and scored on the agents' holdout, with permutation importance run
on that same holdout with the feature transform inside the estimator pipeline. The model's only
judgement is which name to pick; `claimed_holdout_score` always comes from the snippet's own
number, never from the model's rationale, or a model that hallucinates a score could get it onto
the state. `CANDIDATE_SPECS` is module-level data (not buried in the snippet string) because the
harness refits `chosen_model.name` from it directly, and that refit is only reproducible if the
recipe lives in exactly one place.
"""

import json
from typing import Any

from pydantic import Field

from ds_agents import split_manifest
from ds_agents.nodes._run import NodeRun
from ds_agents.state import Contract, ModelResult, PipelineError, PipelineState
from ds_agents.tools.llm import StructuredModel
from ds_agents.tools.protocol import ToolError, Tools

MODEL_TIMEOUT_S = 240
N_PERMUTATION_REPEATS = 10
TOP_IMPORTANCES = 15

SEED_SENTINEL = "__SEED__"

# Shared between "binary" and "multiclass": same two estimators, same hyperparameters. A single
# dict reused for both keys keeps that fact visible instead of two copies that can drift apart.
_CLASSIFIER_SPECS: dict[str, list[tuple[str, str, dict[str, Any]]]] = {
    "logistic_l2": [
        ("scale", "sklearn.preprocessing:StandardScaler", {}),
        (
            "model",
            "sklearn.linear_model:LogisticRegression",
            {"max_iter": 2000, "C": 1.0, "random_state": SEED_SENTINEL},
        ),
    ],
    "hist_gbdt": [
        (
            "model",
            "sklearn.ensemble:HistGradientBoostingClassifier",
            {"max_iter": 200, "learning_rate": 0.1, "random_state": SEED_SENTINEL},
        ),
    ],
}

CANDIDATE_SPECS: dict[str, dict[str, list[tuple[str, str, dict[str, Any]]]]] = {
    "binary": _CLASSIFIER_SPECS,
    "multiclass": _CLASSIFIER_SPECS,
    "regression": {
        "ridge": [
            ("scale", "sklearn.preprocessing:StandardScaler", {}),
            (
                "model",
                "sklearn.linear_model:Ridge",
                {"alpha": 1.0, "random_state": SEED_SENTINEL},
            ),
        ],
        "hist_gbdt": [
            (
                "model",
                "sklearn.ensemble:HistGradientBoostingRegressor",
                {"max_iter": 200, "learning_rate": 0.1, "random_state": SEED_SENTINEL},
            ),
        ],
    },
}

SCORERS: dict[str, str] = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "f1": "f1",
    "log_loss": "neg_log_loss",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}
# Overrides for multiclass only. Binary and regression use SCORERS unchanged.
MULTICLASS_SCORERS: dict[str, str] = {"roc_auc": "roc_auc_ovr", "f1": "f1_macro"}

MODELER_SYSTEM = """You are the modelling step of a tabular data science pipeline.

Every candidate listed below has already been fit and scored by code: cross-validated on pinned \
folds, then refit on the pinned train rows and scored on a holdout that neither you nor any later \
step gets to rescore. Your only job is to pick one candidate by name from the list you are given.

Rules:
- Every score below is in the metric's own natural units: an rmse of 0.30 is shown as 0.30, never \
as a negative number. `greater_is_better` tells you which direction wins: true means a higher \
score is better, false means a lower score is better (for example rmse or log_loss). Read it \
before comparing numbers -- do not assume "higher is better" applies to every metric.
- Weigh cv_mean and holdout_score together, not just whichever single number is largest.
- A candidate with a `fit_error` could not be fit at all. Never choose it.
- You do not get to report a score. `claimed_holdout_score` is filled in later from the \
candidate's own recorded `holdout_score`, never from your rationale.
- `chosen` must be exactly one of the candidate names you were given, spelled exactly."""


class ModelChoice(Contract):
    """The only judgement the modeler asks a model for.

    It picks among candidates already fitted and already scored by code. The score is never the
    model's to state: `claimed_holdout_score` comes from the snippet, so a model that hallucinates
    a number cannot get it onto the state.
    """

    chosen: str = Field(description="Exactly one candidate name from the list you were given.")
    rationale: str


MODEL_SNIPPET = '''
import importlib, json, os
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

{decoder}

TARGET = {target!r}
TASK_TYPE = {task_type!r}
POSITIVE_CLASS = {positive_class!r}
SCORING = {scoring!r}
# sklearn scorers are ALWAYS greater-is-better: `neg_root_mean_squared_error` returns -0.30 for an
# rmse of 0.30. That convention must not escape this snippet. Every score printed below is flipped
# back into natural metric units, so `claimed_holdout_score` is directly comparable to the
# harness's `verified_holdout_score` and the prompt's stated direction matches the numbers shown.
SIGN = -1.0 if SCORING.startswith("neg_") else 1.0
SEED = {seed}
# Interpolated rather than hard-coded so changing SEED_SENTINEL in the module actually changes
# what this snippet matches on -- a literal "__SEED__" here would silently stop seeding the moment
# the constant changed.
SENTINEL = {sentinel!r}
N_REPEATS = {n_repeats}
TOP_N = {top_n}
SPECS = {specs!r}
SPLIT_MANIFEST = json.loads({split_json!r})
FEATURE_CODE = {feature_code!r}

# exec inside the SANDBOX, which is where agent code belongs. CLAUDE.md's rule against exec is
# about the node process, whose state holds planted_leakage_columns.
ns = {{}}
exec(compile(FEATURE_CODE, "feature_transform.py", "exec"), ns)
transform = ns["transform"]
SOURCE_COLUMNS = list(ns["SOURCE_COLUMNS"])
MATRIX_COLUMNS = list(ns["FEATURE_ORDER"])

df = pd.read_csv(os.environ["DS_DATASET"])
# Decoded against this frame's length. A manifest written for a different frame raises here
# rather than fitting on whichever rows happened to be in range.
SPLIT = decode_split(SPLIT_MANIFEST, len(df))
y_raw = df[TARGET]

# Reported, never swallowed: silently substituting classes[-1] for a positive_class that does not
# match the data (spec says "1", the CSV holds True/False) flips the metric to 1-AUC with no trace
# anywhere -- a wrong number in whichever direction happens to be flattering, with nothing in the
# output to explain it.
positive_mismatch = None
if TASK_TYPE == "binary":
    classes = sorted(str(v) for v in y_raw.dropna().unique())
    if POSITIVE_CLASS is not None and POSITIVE_CLASS in classes:
        positive = POSITIVE_CLASS
    else:
        positive = classes[-1] if classes else None
        if POSITIVE_CLASS is not None:
            positive_mismatch = {{"requested": POSITIVE_CLASS, "used": positive}}
    y = (y_raw.astype("string") == str(positive)).astype("int64")
elif TASK_TYPE == "multiclass":
    positive = None
    y = y_raw.astype("string")
else:
    positive = None
    y = pd.to_numeric(y_raw, errors="coerce")

# Rows with no label cannot be trained on or scored, and dropping them silently would make the
# fold sizes disagree with the pinned manifest without saying so.
usable = set(np.flatnonzero(y_raw.notna().to_numpy()).tolist())


def keep(ids):
    return [i for i in ids if 0 <= i < len(df) and i in usable]


train_rows = keep(SPLIT["train"])
holdout_rows = keep(SPLIT["holdout"])
folds = [(keep(f["train"]), keep(f["valid"])) for f in SPLIT["folds"]]

X = df[SOURCE_COLUMNS]
scorer = get_scorer(SCORING)


def resolve_params(params):
    """Substitute the seed sentinel for the actual seed. Shared by build() and the params
    recorded on each candidate below, so ModelResult.params carries the seed that was actually
    used to fit rather than the placeholder string that named it."""
    return {{k: (SEED if v == SENTINEL else v) for k, v in params.items()}}


def build(spec):
    """The transform lives INSIDE the pipeline, so permutation_importance permutes SOURCE
    columns rather than one-hot expansions, and the reported names match the vocabulary
    planted_leakage_columns and Objection.columns are written in."""
    steps = [("features", FunctionTransformer(transform, validate=False))]
    for name, path, params in spec:
        module, cls = path.split(":")
        kwargs = resolve_params(params)
        steps.append((name, getattr(importlib.import_module(module), cls)(**kwargs)))
    return Pipeline(steps)


candidates = []
for name, spec in SPECS.items():
    # Resolved params, not the raw spec: a raw dict would put the literal sentinel string into
    # the published ModelResult.params instead of the seed that was actually used to fit.
    resolved_params = {{}}
    for _step_name, _path, prm in spec:
        resolved_params.update(resolve_params(prm))
    entry = {{"name": name, "params": resolved_params,
              "cv_scores": [], "cv_mean": None, "holdout_score": None,
              "importances": [], "fit_error": None}}
    try:
        for tr, va in folds:
            # The PINNED folds, not cross_val_score: re-splitting here would break the only
            # partition a later contamination objection can be checked against.
            pipe = build(spec)
            pipe.fit(X.iloc[tr], y.iloc[tr])
            entry["cv_scores"].append(SIGN * float(scorer(pipe, X.iloc[va], y.iloc[va])))
        entry["cv_mean"] = float(np.mean(entry["cv_scores"])) if entry["cv_scores"] else None

        pipe = build(spec)
        pipe.fit(X.iloc[train_rows], y.iloc[train_rows])
        entry["holdout_score"] = SIGN * float(
            scorer(pipe, X.iloc[holdout_rows], y.iloc[holdout_rows])
        )

        # Permuted on the agents' HOLDOUT with the model already fitted on train. Permuting
        # training rows measures memorisation, not usefulness.
        result = permutation_importance(
            pipe, X.iloc[holdout_rows], y.iloc[holdout_rows],
            scoring=SCORING, n_repeats=N_REPEATS, random_state=SEED,
        )
        entry["importances"] = sorted(
            (
                [str(c), float(m), float(s)]
                for c, m, s in zip(SOURCE_COLUMNS, result.importances_mean, result.importances_std)
            ),
            key=lambda row: -row[1],
        )
    except Exception as exc:
        # Reported per candidate. One estimator that will not fit must not cost the whole run.
        entry["fit_error"] = f"{{type(exc).__name__}}: {{exc}}"
    candidates.append(entry)

report = {{
    "method": "permutation_importance",
    "scoring": SCORING,
    "n_repeats": N_REPEATS,
    "evaluated_on": "agent_holdout",
    "random_state": SEED,
    "note": "Source-column importances: the feature transform is inside the estimator pipeline, "
            "so sklearn permuted raw columns, not one-hot expansions.",
    "matrix_columns": MATRIX_COLUMNS,
    "candidates": {{c["name"]: c["importances"] for c in candidates}},
}}
path = os.path.join(os.environ["DS_ARTIFACTS"], "feature_importance.json")
with open(path, "w") as fh:
    json.dump(report, fh, indent=2)

scored = [c for c in candidates if c["cv_mean"] is not None]
best = None
if scored:
    # cv_mean is in natural units by now (see SIGN), so the direction is applied exactly once.
    # Applying it on top of a neg_ scorer would pick the WORST candidate on every lower-is-better
    # metric, and the toy fixture is binary so no test would ever notice.
    best = max(scored, key=lambda c: c["cv_mean"] * (1 if {greater_is_better} else -1))["name"]

print(json.dumps({{
    "metric": SCORING,
    "task_type": TASK_TYPE,
    "positive_class": None if positive is None else str(positive),
    "positive_class_mismatch": positive_mismatch,
    "n_train_rows": len(train_rows),
    "n_holdout_rows": len(holdout_rows),
    "n_folds": len(folds),
    "n_features": len(MATRIX_COLUMNS),
    "source_columns": SOURCE_COLUMNS,
    "matrix_columns": MATRIX_COLUMNS,
    "dropped_missing_target": int(len(df) - len(usable)),
    "candidates": [
        {{**c, "importances": c["importances"][:TOP_N]}} for c in candidates
    ],
    "best_by_cv": best,
    "importance_path": path,
}}))
'''


def _scoring_for(task_type: str, metric: str) -> str:
    if task_type == "multiclass" and metric in MULTICLASS_SCORERS:
        return MULTICLASS_SCORERS[metric]
    return SCORERS[metric]


def _user_message(state: PipelineState, result: dict[str, Any]) -> str:
    assert state.spec is not None
    candidates = [
        {
            "name": c["name"],
            "params": c.get("params", {}),
            "cv_scores": c.get("cv_scores", []),
            "cv_mean": c.get("cv_mean"),
            "holdout_score": c.get("holdout_score"),
            "fit_error": c.get("fit_error"),
            "top_importances": c.get("importances", []),
        }
        for c in result.get("candidates", [])
    ]
    facts = {
        "task_description": state.task_description,
        "target": state.spec.target,
        "task_type": state.spec.task_type,
        "metric": state.spec.metric,
        "greater_is_better": state.spec.greater_is_better,
        "n_train_rows": result.get("n_train_rows"),
        "n_holdout_rows": result.get("n_holdout_rows"),
        "n_features": result.get("n_features"),
        "final_features": state.final_features or [],
        "candidates": candidates,
        "open_objections": [
            {
                "category": o.category,
                "columns": o.columns,
                "evidence": o.evidence,
                "severity": o.severity,
            }
            for o in state.open_objections("modeler")
        ],
    }
    return json.dumps(facts, indent=2)


def modeler(state: PipelineState, *, tools: Tools, model: StructuredModel) -> dict[str, Any]:
    run = NodeRun("modeler")
    if state.spec is None:
        return run.failure("no spec: intake did not produce one", recoverable=False)
    if state.feature_code_artifact is None:
        return run.failure(
            "no feature_code_artifact: feature_eng did not produce one", recoverable=False
        )
    if state.split_artifact is None:
        return run.failure("no split_artifact: profiler did not produce one", recoverable=False)

    try:
        feature_payload = tools.read_artifact(state.feature_code_artifact)
    except ToolError as exc:
        return run.failure(f"could not read feature code artifact: {exc}")
    if feature_payload.truncated:
        # A truncated transform either raises SyntaxError when exec'd or silently execs a
        # shorter-than-real module. Neither is a model to fit on.
        return run.failure(
            "feature code artifact was truncated; refusing to exec a partial transform",
            recoverable=False,
        )

    try:
        split_payload = tools.read_artifact(state.split_artifact)
    except ToolError as exc:
        return run.failure(f"could not read split manifest: {exc}")
    if split_payload.truncated:
        # A truncated manifest is a subset of the pinned split, and the snippet below cannot tell
        # the difference: it would fit and score on fewer rows than `split_artifact` says, and
        # `claimed_holdout_score` would be computed against a holdout that is not the holdout.
        # `feature_eng` has the symmetric guard.
        return run.failure(
            "split manifest artifact was truncated; refusing to fit on a partial split",
            recoverable=False,
        )

    scoring = _scoring_for(state.spec.task_type, state.spec.metric)
    specs = CANDIDATE_SPECS[state.spec.task_type]
    code = MODEL_SNIPPET.format(
        decoder=split_manifest.DECODER_SRC,
        target=state.spec.target,
        task_type=state.spec.task_type,
        positive_class=state.spec.positive_class,
        scoring=scoring,
        seed=state.config.random_seed,
        sentinel=SEED_SENTINEL,
        n_repeats=N_PERMUTATION_REPEATS,
        top_n=TOP_IMPORTANCES,
        specs=specs,
        feature_code=feature_payload.content,
        split_json=split_payload.content,
        greater_is_better=state.spec.greater_is_better,
    )

    try:
        run_result = tools.run_python(code, timeout_s=MODEL_TIMEOUT_S)
    except ToolError as exc:
        return run.failure(f"modeling snippet could not run: {exc}")
    if not run_result.ok:
        return run.failure(f"modeling snippet failed: {run_result.stderr.strip()[-500:]}")
    try:
        result = json.loads(run_result.stdout)
    except json.JSONDecodeError as exc:
        return run.failure(f"modeling snippet printed no JSON: {exc}")

    if len(run_result.artifacts_written) != 1:
        return run.failure(
            f"expected exactly one importance artifact, got {len(run_result.artifacts_written)}"
        )
    importance_artifact = run_result.artifacts_written[0]

    errors: list[PipelineError] = []
    mismatch = result.get("positive_class_mismatch")
    if mismatch:
        # Not fatal -- the run still finishes and scores something -- but a metric flip with no
        # trace anywhere is exactly the kind of wrong number this project exists to catch, so it
        # is recorded even though it does not stop the run.
        errors.append(
            PipelineError(
                node="modeler",
                message=(
                    f"spec.positive_class={mismatch['requested']!r} does not match any observed "
                    f"class; used {mismatch['used']!r} instead, which may flip the metric"
                ),
            )
        )

    raw_candidates: list[dict[str, Any]] = result.get("candidates", [])
    for c in raw_candidates:
        if c.get("fit_error"):
            errors.append(
                PipelineError(
                    node="modeler", message=f"{c['name']} failed to fit: {c['fit_error']}"
                )
            )

    model_results = [
        ModelResult(
            name=c["name"],
            params=c.get("params", {}),
            cv_scores=c.get("cv_scores", []),
            claimed_holdout_score=c.get("holdout_score"),
            fit_error=c.get("fit_error"),
        )
        for c in raw_candidates
    ]

    scored = [c for c in raw_candidates if c.get("cv_mean") is not None]
    chosen_name: str | None = None
    if not scored:
        # No estimator produced a usable fit. Still a row: the reporter needs the errors and the
        # empty candidate list to say what happened, not a crashed graph.
        errors.append(
            PipelineError(
                node="modeler",
                message="every candidate failed to fit; no model can be chosen",
            )
        )
    else:
        valid_names = {c["name"] for c in scored}
        try:
            choice = run.record(
                model.generate(
                    system=MODELER_SYSTEM,
                    user=_user_message(state, result),
                    schema=ModelChoice,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a real client's timeout/rate-limit must not crash the node
            chosen_name = result.get("best_by_cv")
            errors.append(
                PipelineError(
                    node="modeler",
                    message=f"model call failed, falling back to best_by_cv: {exc}",
                )
            )
        else:
            if choice.chosen in valid_names:
                chosen_name = choice.chosen
            else:
                chosen_name = result.get("best_by_cv")
                errors.append(
                    PipelineError(
                        node="modeler",
                        message=f"model chose unknown candidate {choice.chosen!r}; "
                        f"falling back to best_by_cv={chosen_name!r}",
                    )
                )

    candidates_by_name = {c["name"]: c for c in raw_candidates}
    chosen_model: ModelResult | None = None
    top_importances: list[tuple[str, float]] = []
    if chosen_name is not None:
        chosen_model = next((m for m in model_results if m.name == chosen_name), None)
        chosen_raw = candidates_by_name.get(chosen_name)
        if chosen_raw is not None:
            top_importances = [
                (name, mean) for name, mean, _std in chosen_raw.get("importances", [])
            ][:TOP_IMPORTANCES]

    try:
        for c in raw_candidates:
            if c.get("cv_mean") is not None:
                tools.log_metric(state.config.run_id, f"cv_mean.{c['name']}", c["cv_mean"])
        if chosen_model is not None and chosen_model.claimed_holdout_score is not None:
            tools.log_metric(
                state.config.run_id, "claimed_holdout_score", chosen_model.claimed_holdout_score
            )
    except ToolError as exc:
        errors.append(PipelineError(node="modeler", message=f"metric logging failed: {exc}"))

    update: dict[str, Any] = {
        "candidates": model_results,
        "chosen_model": chosen_model,
        "importance_artifact": importance_artifact,
        "top_importances": top_importances,
        "node_trace": [run.event()],
    }
    if errors:
        update["errors"] = errors
    return update
