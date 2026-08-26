"""modeler writes `candidates`, `chosen_model`, `importance_artifact`, `top_importances`.

The test that matters most here is `test_the_claimed_score_comes_from_the_snippet_not_the_model`:
the modeler asks a model to pick a name, never to report a number. If `claimed_holdout_score`
could come from the model's own words, a model that simply claims a great score could get it onto
the state with nothing to check it against.
"""

import json

import pytest
from conftest import FakeTools, ScriptedModel

from ds_agents.nodes.modeler import ModelChoice, modeler
from ds_agents.state import PipelineState, TaskSpec
from ds_agents.tools.protocol import ArtifactMeta, ArtifactPayload, RunResult, ToolError

pytestmark = pytest.mark.fast

FEATURE_CODE = (
    "def transform(df):\n    return df[['tenure_months', 'monthly_charges']].astype('float64')\n"
)

SPLIT_MANIFEST = {
    "strategy": "stratified",
    "seed": 20260822,
    "target": "churned",
    "n_rows": 200,
    "train": list(range(160)),
    "holdout": list(range(160, 200)),
    "folds": [{"train": list(range(128)), "valid": list(range(128, 160))}],
}

SNIPPET_OUT = {
    "metric": "roc_auc",
    "task_type": "binary",
    "positive_class": "1",
    "n_train_rows": 160,
    "n_holdout_rows": 40,
    "n_folds": 5,
    "n_features": 12,
    "source_columns": [
        "tenure_months",
        "monthly_charges",
        "support_tickets_90d",
        "region",
        "plan_tier",
        "account_status_code",
    ],
    "matrix_columns": ["tenure_months", "monthly_charges", "account_status_code=ACTIVE_S1"],
    "dropped_missing_target": 0,
    "candidates": [
        {
            "name": "logistic_l2",
            "params": {"max_iter": 2000, "C": 1.0, "random_state": 20260822},
            "cv_scores": [0.90, 0.91, 0.92, 0.90, 0.93],
            "cv_mean": 0.9121,
            "holdout_score": 0.9833,
            "importances": [
                ["account_status_code", 0.367, 0.02],
                ["support_tickets_90d", 0.034, 0.01],
            ],
            "fit_error": None,
        },
        {
            "name": "hist_gbdt",
            "params": {"max_iter": 200, "learning_rate": 0.1, "random_state": 20260822},
            "cv_scores": [0.86, 0.88, 0.87, 0.87, 0.88],
            "cv_mean": 0.8718,
            "holdout_score": 0.95,
            "importances": [
                ["account_status_code", 0.360, 0.03],
                ["support_tickets_90d", 0.030, 0.01],
            ],
            "fit_error": None,
        },
    ],
    "best_by_cv": "logistic_l2",
    "importance_path": "/artifacts/feature_importance.json",
}


def state(**overrides) -> PipelineState:
    fields = {
        "dataset_id": "toy",
        "task_description": "Predict churned, report roc_auc.",
        "spec": TaskSpec(target="churned", task_type="binary", metric="roc_auc"),
        "feature_code_artifact": "art-009-feature-code",
        "split_artifact": "art-007-split",
        "final_features": [
            "tenure_months",
            "monthly_charges",
            "support_tickets_90d",
            "region",
            "plan_tier",
            "account_status_code",
        ],
    }
    fields.update(overrides)
    return PipelineState(**fields)


def feature_code_payload(truncated: bool = False) -> ArtifactPayload:
    return ArtifactPayload(
        meta=ArtifactMeta(id="art-009-feature-code", name="feature_transform.py", kind="text"),
        content=FEATURE_CODE,
        truncated=truncated,
    )


def split_payload() -> ArtifactPayload:
    return ArtifactPayload(
        meta=ArtifactMeta(id="art-007-split", name="split_manifest.json", kind="json"),
        content=json.dumps(SPLIT_MANIFEST),
        truncated=False,
    )


def tools_for(
    snippet_out: dict | None = None,
    run_result: RunResult | Exception | None = None,
    feature_truncated: bool = False,
) -> FakeTools:
    if run_result is None:
        run_result = RunResult(
            stdout=json.dumps(SNIPPET_OUT if snippet_out is None else snippet_out),
            artifacts_written=["art-011-importance"],
        )
    return FakeTools(
        run_results=[run_result],
        artifacts={
            "art-009-feature-code": feature_code_payload(feature_truncated),
            "art-007-split": split_payload(),
        },
    )


def choice(name: str = "logistic_l2") -> ModelChoice:
    return ModelChoice(chosen=name, rationale="strongest cv and holdout score")


def test_writes_the_model_fields_and_nothing_else():
    tools = tools_for()
    model = ScriptedModel({ModelChoice: choice()})

    update = modeler(state(), tools=tools, model=model)

    assert set(update) == {
        "candidates",
        "chosen_model",
        "importance_artifact",
        "top_importances",
        "node_trace",
    }
    assert update["importance_artifact"] == "art-011-importance"


def test_the_claimed_score_comes_from_the_snippet_not_the_model():
    """The single most important test in this file."""
    tools = tools_for()
    model = ScriptedModel({ModelChoice: ModelChoice(chosen="hist_gbdt", rationale="I scored 0.99")})

    update = modeler(state(), tools=tools, model=model)

    assert update["chosen_model"].name == "hist_gbdt"
    assert update["chosen_model"].claimed_holdout_score == 0.95


def test_the_published_params_are_the_actual_seed_not_the_sentinel():
    """Finding 10, second half: `entry["params"]` used to be built from the raw candidate spec, so
    a published `ModelResult.params` would carry the literal placeholder string "__SEED__" rather
    than the seed that was actually used to fit -- a published field lying about what happened.
    `SNIPPET_OUT` here represents what the FIXED snippet prints (resolved params), so this
    assertion is what actually exercises the pass-through; without it, hand-writing 20260822 into
    the fixture proves nothing about whether the node (or the snippet) ever resolves anything."""
    tools = tools_for()
    model = ScriptedModel({ModelChoice: choice()})

    update = modeler(state(), tools=tools, model=model)

    assert update["chosen_model"].params["random_state"] == 20260822
    assert not any(
        v == "__SEED__"
        for c in update["candidates"]
        for v in c.params.values()
        if isinstance(v, str)
    )


def test_the_seed_sentinel_is_interpolated_and_params_are_resolved_in_the_snippet():
    """Finding 10, first half: MODEL_SNIPPET used to hard-code the literal string "__SEED__" in
    its comparison inside build(), so changing SEED_SENTINEL in this module would silently stop
    the seeding from working -- nothing would break, it would just quietly stop matching. Regression
    guard on the emitted snippet source: the sentinel must be threaded through `.format()`, and the
    params recorded on each candidate must come from the resolved dict, not the raw spec."""
    import importlib

    # Not `import ds_agents.nodes.modeler as ...`: `nodes/__init__.py` does
    # `from ds_agents.nodes.modeler import modeler`, which rebinds the `modeler` attribute on the
    # `ds_agents.nodes` package from the submodule to the function. A dotted `import ... as`
    # resolves through that shadowed attribute; `importlib.import_module` goes through
    # `sys.modules` instead and gets the real submodule.
    modeler_module = importlib.import_module("ds_agents.nodes.modeler")

    tools = tools_for()
    modeler(state(), tools=tools, model=ScriptedModel({ModelChoice: choice()}))
    code = tools.code_run[0]

    assert f"SENTINEL = {modeler_module.SEED_SENTINEL!r}" in code
    assert "v == SENTINEL" in code
    assert 'v == "__SEED__"' not in code
    assert '"params": resolved_params' in code
    assert '"params": {p: v for _s, _p, prm in spec for p, v in prm.items()}' not in code


def test_a_positive_class_mismatch_is_recorded_but_does_not_stop_the_run():
    """Finding 7: `spec.positive_class` not matching any observed class used to silently
    substitute the last class alphabetically, which flips ROC-AUC to 1-AUC with no trace anywhere
    -- a wrong number in whichever direction happens to be flattering. The snippet now reports the
    mismatch and the node records it as a (non-fatal) PipelineError; the run still finishes and
    still chooses a model."""
    snippet_out = {
        **SNIPPET_OUT,
        "positive_class": "False",
        "positive_class_mismatch": {"requested": "1", "used": "False"},
    }
    tools = tools_for(snippet_out)
    model = ScriptedModel({ModelChoice: choice()})

    update = modeler(state(), tools=tools, model=model)

    assert update["chosen_model"] is not None
    mismatch_errors = [e for e in update["errors"] if "positive_class" in e.message]
    assert len(mismatch_errors) == 1
    assert mismatch_errors[0].recoverable is True
    assert "'1'" in mismatch_errors[0].message
    assert "'False'" in mismatch_errors[0].message


def test_no_positive_class_mismatch_key_means_no_extra_error():
    """The mismatch key is absent whenever spec.positive_class matched (or the task isn't
    binary) -- `result.get(...)` must not misfire into an error on a run that has nothing wrong."""
    tools = tools_for()  # SNIPPET_OUT carries no positive_class_mismatch key
    model = ScriptedModel({ModelChoice: choice()})

    update = modeler(state(), tools=tools, model=model)

    assert "errors" not in update


def test_top_importances_are_source_columns():
    tools = tools_for()
    model = ScriptedModel({ModelChoice: choice()})

    update = modeler(state(), tools=tools, model=model)

    assert update["top_importances"]
    assert not any("=" in name for name, _mean in update["top_importances"])
    assert update["top_importances"][0] == ("account_status_code", 0.367)


def test_the_candidate_scores_reach_the_prompt():
    tools = tools_for()
    model = ScriptedModel({ModelChoice: choice()})

    modeler(state(), tools=tools, model=model)

    (_system, user, _schema) = model.calls[0]
    facts = json.loads(user)
    names = {c["name"] for c in facts["candidates"]}
    assert names == {"logistic_l2", "hist_gbdt"}
    by_name = {c["name"]: c for c in facts["candidates"]}
    assert by_name["logistic_l2"]["cv_mean"] == 0.9121
    assert by_name["logistic_l2"]["holdout_score"] == 0.9833
    assert by_name["hist_gbdt"]["cv_mean"] == 0.8718
    assert by_name["hist_gbdt"]["holdout_score"] == 0.95
    assert "greater_is_better" in facts


def test_choosing_a_candidate_that_does_not_exist_falls_back_to_best_by_cv():
    tools = tools_for()
    model = ScriptedModel(
        {ModelChoice: ModelChoice(chosen="xgboost_turbo", rationale="sounds fast")}
    )

    update = modeler(state(), tools=tools, model=model)

    assert update["chosen_model"].name == SNIPPET_OUT["best_by_cv"]
    assert any("unknown candidate" in e.message for e in update["errors"])


def test_a_model_client_error_still_chooses_and_records():
    tools = tools_for()
    model = ScriptedModel({ModelChoice: TimeoutError("read timed out")})

    update = modeler(state(), tools=tools, model=model)

    assert update["chosen_model"].name == SNIPPET_OUT["best_by_cv"]
    assert any("model call failed" in e.message for e in update["errors"])


def test_a_candidate_that_failed_to_fit_is_recorded_but_not_chosen():
    snippet_out = {
        **SNIPPET_OUT,
        "candidates": [
            SNIPPET_OUT["candidates"][0],
            {
                "name": "hist_gbdt",
                "params": {},
                "cv_scores": [],
                "cv_mean": None,
                "holdout_score": None,
                "importances": [],
                "fit_error": "ValueError: could not fit",
            },
        ],
        "best_by_cv": "logistic_l2",
    }
    tools = tools_for(snippet_out)
    model = ScriptedModel({ModelChoice: choice("logistic_l2")})

    update = modeler(state(), tools=tools, model=model)

    names = [c.name for c in update["candidates"]]
    assert "hist_gbdt" in names
    failed = next(c for c in update["candidates"] if c.name == "hist_gbdt")
    assert failed.cv_scores == []
    assert failed.claimed_holdout_score is None
    assert update["chosen_model"].name == "logistic_l2"
    assert any("hist_gbdt failed to fit" in e.message for e in update["errors"])


def test_all_candidates_failing_leaves_no_chosen_model_but_still_a_row():
    snippet_out = {
        **SNIPPET_OUT,
        "candidates": [
            {
                "name": "logistic_l2",
                "params": {},
                "cv_scores": [],
                "cv_mean": None,
                "holdout_score": None,
                "importances": [],
                "fit_error": "ValueError: singular matrix",
            },
            {
                "name": "hist_gbdt",
                "params": {},
                "cv_scores": [],
                "cv_mean": None,
                "holdout_score": None,
                "importances": [],
                "fit_error": "ValueError: could not fit",
            },
        ],
        "best_by_cv": None,
    }
    tools = tools_for(snippet_out)
    model = ScriptedModel({})

    update = modeler(state(), tools=tools, model=model)

    assert update["chosen_model"] is None
    assert len(update["candidates"]) == 2
    assert model.calls == []
    assert update["errors"]
    assert all(e.recoverable for e in update["errors"])


def test_the_feature_code_and_the_pinned_split_reach_the_snippet():
    tools = tools_for()
    model = ScriptedModel({ModelChoice: choice()})

    modeler(state(), tools=tools, model=model)

    code = tools.code_run[0]
    assert "def transform(df):" in code
    assert "churned" in code
    assert "160" in code  # a train row index from the pinned split


def test_a_truncated_feature_code_artifact_is_unrecoverable():
    tools = tools_for(feature_truncated=True)
    tools.run_results = []  # run_python must never be reached
    model = ScriptedModel({})

    update = modeler(state(), tools=tools, model=model)

    assert update["errors"][0].recoverable is False
    assert tools.code_run == []


def test_no_feature_code_artifact_is_unrecoverable():
    update = modeler(state(feature_code_artifact=None), tools=FakeTools(), model=ScriptedModel({}))

    assert update["errors"][0].recoverable is False
    assert "candidates" not in update


def test_no_split_artifact_is_unrecoverable():
    update = modeler(state(split_artifact=None), tools=FakeTools(), model=ScriptedModel({}))

    assert update["errors"][0].recoverable is False
    assert "candidates" not in update


def test_a_failed_snippet_writes_an_error_and_no_model_fields():
    tools = tools_for(run_result=RunResult(exit_code=1, stderr="ValueError: boom"))
    model = ScriptedModel({})

    update = modeler(state(), tools=tools, model=model)

    assert "candidates" not in update
    assert "ValueError" in update["errors"][0].message


def test_a_raised_tool_error_is_caught():
    tools = tools_for(run_result=ToolError("sandbox unavailable"))
    model = ScriptedModel({})

    update = modeler(state(), tools=tools, model=model)

    assert "candidates" not in update
    assert "sandbox unavailable" in update["errors"][0].message


def test_metrics_are_logged_with_the_run_id():
    tools = tools_for()
    model = ScriptedModel({ModelChoice: choice()})
    st = state()

    modeler(st, tools=tools, model=model)

    assert (st.config.run_id, "cv_mean.logistic_l2", 0.9121) in tools.metrics
    assert (st.config.run_id, "cv_mean.hist_gbdt", 0.8718) in tools.metrics
    assert (st.config.run_id, "claimed_holdout_score", 0.9833) in tools.metrics


def test_the_regression_branch_uses_the_regression_candidates():
    reg_snippet_out = {
        **SNIPPET_OUT,
        "metric": "neg_root_mean_squared_error",
        "task_type": "regression",
        "candidates": [
            {
                "name": "ridge",
                "params": {"alpha": 1.0},
                "cv_scores": [-0.30, -0.31],
                "cv_mean": -0.305,
                "holdout_score": -0.30,
                "importances": [],
                "fit_error": None,
            },
            {
                "name": "hist_gbdt",
                "params": {"max_iter": 200},
                "cv_scores": [-0.45, -0.44],
                "cv_mean": -0.445,
                "holdout_score": -0.45,
                "importances": [],
                "fit_error": None,
            },
        ],
        "best_by_cv": "ridge",
    }
    reg_state = state(
        spec=TaskSpec(target="price", task_type="regression", metric="rmse"),
        final_features=["a", "b"],
    )
    tools = tools_for(reg_snippet_out)
    model = ScriptedModel({ModelChoice: choice("ridge")})

    modeler(reg_state, tools=tools, model=model)

    code = tools.code_run[0]
    assert "Ridge" in code
    assert "neg_root_mean_squared_error" in code
    assert "LogisticRegression" not in code


def test_lower_is_better_metrics_pick_the_smaller_fallback():
    """The fallback trusts the snippet's own `best_by_cv`, which is computed with the metric's
    direction already applied -- for rmse, the smaller error wins."""
    reg_snippet_out = {
        **SNIPPET_OUT,
        "metric": "neg_root_mean_squared_error",
        "task_type": "regression",
        "candidates": [
            {
                "name": "ridge",
                "params": {"alpha": 1.0},
                "cv_scores": [-0.30, -0.31],
                "cv_mean": -0.305,
                "holdout_score": 0.30,
                "importances": [],
                "fit_error": None,
            },
            {
                "name": "hist_gbdt",
                "params": {"max_iter": 200},
                "cv_scores": [-0.45, -0.44],
                "cv_mean": -0.445,
                "holdout_score": 0.45,
                "importances": [],
                "fit_error": None,
            },
        ],
        "best_by_cv": "ridge",
    }
    reg_state = state(
        spec=TaskSpec(target="price", task_type="regression", metric="rmse"),
        final_features=["a", "b"],
    )
    tools = tools_for(reg_snippet_out)
    model = ScriptedModel({ModelChoice: RuntimeError("client unavailable")})

    update = modeler(reg_state, tools=tools, model=model)

    assert update["chosen_model"].name == "ridge"
    assert update["chosen_model"].claimed_holdout_score == 0.30
