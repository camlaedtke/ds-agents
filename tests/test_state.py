"""The contract every node depends on. If these fail, nothing downstream means anything."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ds_agents.state import (
    ModelResult,
    NodeEvent,
    Objection,
    PipelineState,
    ProfileReport,
    ReviewPass,
    RunConfig,
    TaskSpec,
)

pytestmark = pytest.mark.fast


def leak_objection(**overrides) -> Objection:
    kwargs = {
        "category": "leakage",
        "subcategory": "post_hoc_status_code",
        "target_node": "feature_eng",
        "columns": ["account_status_code"],
        "evidence": "agrees with target on 91% of rows",
        "severity": "high",
        "raised_at_iteration": 0,
    }
    return Objection(**{**kwargs, **overrides})


def populated_state() -> PipelineState:
    """Every optional field set, so round-tripping actually exercises the schema."""
    return PipelineState(
        dataset_id="toy",
        task_description="predict churn",
        spec=TaskSpec(target="churned", task_type="binary", metric="roc_auc"),
        profile=ProfileReport(n_rows=200, n_columns=8),
        split_artifact="art-split-1",
        feature_code_artifact="art-features-1",
        feature_summary="one-hot region and plan_tier",
        final_features=["support_tickets_90d", "tenure_months"],
        dropped_features=["account_status_code"],
        candidates=[ModelResult(name="logreg", cv_scores=[0.8, 0.82, 0.79])],
        chosen_model=ModelResult(name="logreg", cv_scores=[0.8], claimed_holdout_score=0.81),
        shap_artifact="art-shap-1",
        top_importances=[("support_tickets_90d", 0.62)],
        review_iterations=1,
        objections=[leak_objection()],
        reviewer_claim="block",
        review_verdict="block",
        report_artifact="art-report-1",
        planted_leakage_columns=["account_status_code"],
        verified_holdout_score=0.74,
        baseline_score=0.70,
        node_trace=[NodeEvent(node="intake", started=datetime(2026, 8, 22, tzinfo=UTC))],
        ended_at=datetime(2026, 8, 22, 0, 5, tzinfo=UTC),
    )


class TestRoundTrip:
    def test_dumped_state_can_be_read_back(self):
        state = populated_state()
        assert PipelineState.from_dump(json.loads(state.model_dump_json())) == state

    def test_computed_fields_survive_into_the_dump(self):
        """The harness reads these out of the JSONL; a plain @property would vanish silently."""
        dumped = json.loads(populated_state().model_dump_json())
        for key in ("total_cost_usd", "loop_exhausted", "wall_seconds", "score_ratio"):
            assert key in dumped
        assert dumped["spec"]["greater_is_better"] is True
        assert dumped["node_trace"][0]["wall_seconds"] is None

    def test_minimal_state_needs_only_intake_inputs(self):
        state = PipelineState(dataset_id="toy", task_description="predict churn")
        assert state.review_verdict == "pending"
        assert state.objections == []
        assert state.config.loop_cap == 3


class TestContractEnforcement:
    def test_unknown_field_is_rejected(self):
        """A node inventing a field is a contract violation, not a convenience."""
        with pytest.raises(ValidationError):
            PipelineState(dataset_id="toy", task_description="x", sneaky_field="nope")

    def test_objection_category_is_closed(self):
        with pytest.raises(ValidationError):
            leak_objection(category="vibes")

    def test_reviewer_cannot_route_to_a_non_routable_node(self):
        with pytest.raises(ValidationError):
            leak_objection(target_node="reporter")

    def test_leakage_objection_without_columns_is_rejected(self):
        """leakage_caught is a set comparison. An objection naming no column cannot be scored."""
        with pytest.raises(ValidationError, match="requires at least one column"):
            leak_objection(columns=[])

    def test_non_column_scoped_objection_may_omit_columns(self):
        assert (
            Objection(
                category="overfit",
                subcategory="cv_holdout_gap",
                target_node="modeler",
                evidence="cv 0.99 vs holdout 0.62",
                severity="high",
                raised_at_iteration=1,
            ).columns
            == []
        )

    def test_run_config_is_frozen(self):
        """A node must not be able to raise its own loop cap mid-run."""
        state = PipelineState(dataset_id="toy", task_description="x")
        with pytest.raises(ValidationError):
            state.config.loop_cap = 99

    def test_keyed_split_strategies_require_a_key(self):
        with pytest.raises(ValidationError, match="requires split_key"):
            TaskSpec(target="y", task_type="binary", metric="roc_auc", split_strategy="temporal")

    def test_estimator_params_accept_nested_values(self):
        """Real params include lists and dicts. A ValidationError here would cost the whole run."""
        result = ModelResult(name="xgb", params={"max_depth": [3, 5], "cb": {"eta": 0.1}})
        assert result.params["max_depth"] == [3, 5]

    def test_sample_values_are_capped(self):
        from ds_agents.state import ColumnProfile

        with pytest.raises(ValidationError):
            ColumnProfile(
                name="c",
                dtype="object",
                missing_fraction=0.0,
                n_unique=9,
                sample_values=list("abcdef"),
            )


class TestReviewLoop:
    def test_loop_exhausted_reads_the_frozen_cap(self):
        state = PipelineState(dataset_id="toy", task_description="x", config=RunConfig(loop_cap=3))
        assert not state.loop_exhausted
        state.review_iterations = 3
        assert state.loop_exhausted

    def test_open_objections_respects_review_pass_dispositions(self):
        """Silence and acceptance must not look the same."""
        raised = leak_objection()
        state = PipelineState(dataset_id="toy", task_description="x", objections=[raised])
        assert len(state.open_objections()) == 1

        state.review_passes = [
            ReviewPass(
                iteration=1,
                claim="pass",
                routed_to="reporter",
                dispositions={raised.id: "resolved"},
            )
        ]
        assert state.open_objections() == []

    def test_an_unmentioned_objection_stays_open(self):
        """A reviewer that goes quiet has not accepted anything."""
        raised = leak_objection()
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            objections=[raised],
            review_passes=[
                ReviewPass(
                    iteration=1,
                    claim="pass",
                    routed_to="reporter",
                    dispositions={raised.id: "not_reviewed"},
                )
            ],
        )
        assert len(state.open_objections()) == 1

    def test_open_objections_filters_by_target_node(self):
        state = populated_state()
        assert state.open_objections(target_node="feature_eng")[0].category == "leakage"
        assert state.open_objections(target_node="modeler") == []


class TestDerivedNumbers:
    def test_score_ratio_is_direction_aware(self):
        higher = PipelineState(
            dataset_id="d",
            task_description="x",
            spec=TaskSpec(target="y", task_type="binary", metric="roc_auc"),
            verified_holdout_score=0.8,
            baseline_score=0.7,
        )
        lower = PipelineState(
            dataset_id="d",
            task_description="x",
            spec=TaskSpec(target="y", task_type="regression", metric="rmse"),
            verified_holdout_score=0.7,
            baseline_score=0.8,
        )
        assert higher.score_ratio > 1
        assert lower.score_ratio > 1, "beating an RMSE baseline means a LOWER score"

    def test_cv_mean_is_none_without_scores(self):
        assert ModelResult(name="empty").cv_mean is None
        assert ModelResult(name="m", cv_scores=[0.5, 0.7]).cv_mean == pytest.approx(0.6)

    def test_total_cost_sums_the_node_trace(self):
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            node_trace=[
                NodeEvent(node="intake", started=datetime(2026, 8, 22, tzinfo=UTC), cost_usd=0.01),
                NodeEvent(
                    node="profiler", started=datetime(2026, 8, 22, tzinfo=UTC), cost_usd=0.02
                ),
            ],
        )
        assert state.total_cost_usd == pytest.approx(0.03)

    def test_timestamps_are_timezone_aware(self):
        """Naive datetimes make committed JSONL inconsistent across machines."""
        assert PipelineState(dataset_id="t", task_description="x").started_at.tzinfo is not None


class TestResultsRow:
    def test_catching_the_planted_leak_scores_as_a_catch(self):
        row = populated_state().results_row()
        assert row["leakage_caught"] is True
        assert row["leakage_recall"] == 1.0
        assert row["false_alarm"] == 0

    def test_caught_and_remediated_are_separate_numbers(self):
        """Flagging the leak and actually removing it are different events."""
        state = populated_state()
        state.final_features = ["support_tickets_90d", "account_status_code"]
        row = state.results_row()
        assert row["leakage_caught"] is True
        assert row["leakage_remediated"] is False

    def test_flagging_a_clean_column_is_a_false_alarm_not_a_catch(self):
        state = populated_state()
        state.objections = [leak_objection(columns=["region"])]
        row = state.results_row()
        assert row["leakage_caught"] is False
        assert row["false_alarm_columns"] == ["region"]
        assert row["leakage_precision"] == 0.0

    def test_claimed_and_verified_scores_are_both_reported(self):
        """The gap between what the agent said and what we measured is itself a finding."""
        row = populated_state().results_row()
        assert row["claimed_holdout_score"] == 0.81
        assert row["verified_holdout_score"] == 0.74
        assert row["holdout_claim_gap"] == pytest.approx(0.07)

    def test_row_is_json_serialisable(self):
        json.dumps(populated_state().results_row())
