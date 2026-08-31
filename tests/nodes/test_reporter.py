"""reporter writes `report_artifact`.

The tests that matter here are the two properties CLAUDE.md and the node design call out
explicitly: the reporter must never raise, even from a bare state where nothing upstream ran, and
it must never render `planted_leakage_columns` / `verified_holdout_score` / the baseline --
those are harness-written answer keys that happen to be in scope because the node receives the
whole state, and writing them into an artifact would leak the answer key into a file a Phase 5
generalist arm might read through the artifact store.
"""

from datetime import UTC, datetime

import pytest
from conftest import FakeTools, ScriptedModel

from ds_agents.nodes.reporter import reporter
from ds_agents.state import (
    ColumnProfile,
    LeakageCandidate,
    ModelResult,
    NodeEvent,
    NodeName,
    Objection,
    PipelineState,
    ProfileReport,
    ReviewPass,
    TaskSpec,
)
from ds_agents.tools.protocol import ToolError

pytestmark = pytest.mark.fast


def _event(node: NodeName, cost: float = 0.001) -> NodeEvent:
    started = datetime(2026, 8, 26, 12, 0, 0, tzinfo=UTC)
    return NodeEvent(
        node=node,
        started=started,
        ended=started,
        input_tokens=100,
        output_tokens=50,
        cost_usd=cost,
        model="haiku",
    )


def full_state() -> PipelineState:
    """A fully-populated fake state, one of every kind of content the report renders."""
    profile = ProfileReport(
        n_rows=200,
        n_columns=8,
        columns=[
            ColumnProfile(name="tenure_months", dtype="int64", missing_fraction=0.0, n_unique=71),
            ColumnProfile(
                name="monthly_charges", dtype="float64", missing_fraction=0.08, n_unique=190
            ),
            ColumnProfile(
                name="account_status_code", dtype="object", missing_fraction=0.0, n_unique=4
            ),
        ],
        leakage_candidates=[
            LeakageCandidate(
                column="account_status_code",
                reason="status is assigned after the churn decision",
                evidence="normalized mutual information with the target is 0.518",
                suspicion="high",
            )
        ],
        target_balance={"0": 0.745, "1": 0.255},
    )
    resolved = Objection(
        id="ob-resolved",
        category="leakage",
        subcategory="post-outcome field",
        target_node="feature_eng",
        columns=["account_status_code"],
        evidence="mutual information 0.518",
        severity="high",
        raised_at_iteration=0,
    )
    open_obj = Objection(
        id="ob-open",
        category="overfit",
        subcategory="train/holdout gap",
        target_node="modeler",
        columns=[],
        evidence="cv 0.91 vs holdout 0.98",
        severity="medium",
        raised_at_iteration=1,
    )
    review_pass = ReviewPass(
        iteration=0,
        claim="block",
        routed_to="feature_eng",
        dispositions={"ob-resolved": "resolved"},
    )
    chosen = ModelResult(
        name="hist_gbdt",
        params={"max_iter": 200},
        cv_scores=[0.87, 0.88],
        claimed_holdout_score=0.95,
    )
    failed = ModelResult(name="logistic_l2", cv_scores=[], claimed_holdout_score=None)
    return PipelineState(
        dataset_id="toy",
        task_description="Predict churned, report roc_auc.",
        spec=TaskSpec(target="churned", task_type="binary", metric="roc_auc"),
        profile=profile,
        split_artifact="art-007-split-manifest",
        feature_code_artifact="art-011-feature-transform",
        feature_summary="6 source columns -> 12 matrix columns, fitted on 160 train rows.",
        final_features=[
            "tenure_months",
            "monthly_charges",
            "support_tickets_90d",
            "region",
            "plan_tier",
            "account_status_code",
        ],
        dropped_features=["customer_id"],
        candidates=[chosen, failed],
        chosen_model=chosen,
        importance_artifact="art-014-feature-importance",
        top_importances=[("account_status_code", 0.367), ("support_tickets_90d", 0.034)],
        review_iterations=1,
        objections=[resolved, open_obj],
        review_passes=[review_pass],
        reviewer_claim="block",
        review_verdict="block",
        node_trace=[_event("intake"), _event("profiler"), _event("feature_eng")],
    )


def test_writes_only_the_report_artifact():
    tools = FakeTools()

    update = reporter(full_state(), tools=tools, model=ScriptedModel({}))

    assert set(update) == {"report_artifact", "node_trace"}
    assert len(update["node_trace"]) == 1


def test_the_report_names_the_chosen_model_and_the_claimed_score():
    tools = FakeTools()

    update = reporter(full_state(), tools=tools, model=ScriptedModel({}))

    content = tools.artifacts[update["report_artifact"]].content
    assert "hist_gbdt" in content
    assert "0.9500" in content


def test_the_report_does_not_contain_the_answer_key():
    """The test that keeps the artifact store clean for the Phase 5 generalist arm.

    `planted_leakage_columns`, `verified_holdout_score`, and the two baseline points are
    harness-written
    ground truth that happens to be in scope because the node receives the whole state. None of
    them may reach the rendered artifact.
    """
    state = full_state().model_copy(
        update={
            "planted_leakage_columns": ["secret_col"],
            "verified_holdout_score": 0.4242,
            "baseline_zero_score": 0.5151,
            "baseline_unit_score": 0.6262,
        }
    )
    tools = FakeTools()

    update = reporter(state, tools=tools, model=ScriptedModel({}))

    content = tools.artifacts[update["report_artifact"]].content
    assert "secret_col" not in content
    assert "0.4242" not in content
    assert "0.5151" not in content
    assert "0.6262" not in content


def test_the_report_survives_a_state_where_everything_upstream_failed():
    bare = PipelineState(dataset_id="toy", task_description="x")
    tools = FakeTools()

    update = reporter(bare, tools=tools, model=ScriptedModel({}))

    assert "report_artifact" in update
    content = tools.artifacts[update["report_artifact"]].content
    assert "ds-agents run report -- toy" in content
    assert "_intake produced no spec; see Errors._" in content
    assert "_profiler did not run; see Errors._" in content
    assert "_feature_eng did not run; see Errors._" in content
    assert "_modeler did not run; see Errors._" in content


def test_objections_render_with_their_columns_and_status():
    tools = FakeTools()

    update = reporter(full_state(), tools=tools, model=ScriptedModel({}))

    content = tools.artifacts[update["report_artifact"]].content
    assert "account_status_code" in content
    assert "resolved" in content
    assert "not_reviewed" in content  # ob-open was never dispositioned in the one review pass


def test_the_cost_table_covers_every_node_event():
    state = full_state()
    tools = FakeTools()

    update = reporter(state, tools=tools, model=ScriptedModel({}))

    content = tools.artifacts[update["report_artifact"]].content
    for event in state.node_trace:
        assert event.node in content
    assert "0.0030" in content  # 3 events at 0.001 each, in the total row


def test_a_write_artifact_failure_leaves_an_error_and_a_trace_event():
    class FailingWriteTools(FakeTools):
        def write_artifact(self, *args, **kwargs):
            raise ToolError("artifact store unavailable")

    tools = FailingWriteTools()

    update = reporter(full_state(), tools=tools, model=ScriptedModel({}))

    assert "report_artifact" not in update
    (error,) = update["errors"]
    assert error.recoverable is True
    assert "artifact store unavailable" in error.message
    assert len(update["node_trace"]) == 1


def test_the_reporter_never_calls_a_model():
    tools = FakeTools()
    model = ScriptedModel({})

    reporter(full_state(), tools=tools, model=model)

    assert model.calls == []


def test_the_report_says_the_score_is_a_claim():
    tools = FakeTools()

    update = reporter(full_state(), tools=tools, model=ScriptedModel({}))

    content = tools.artifacts[update["report_artifact"]].content
    assert "Nothing here is a grade" in content
    assert "a claim, not a verified score" in content
