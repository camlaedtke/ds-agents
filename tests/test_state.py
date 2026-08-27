"""The contract every node depends on. If these fail, nothing downstream means anything."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ds_agents.state import (
    ColumnProfile,
    LeakageCandidate,
    ModelResult,
    NodeEvent,
    Objection,
    PipelineError,
    PipelineState,
    ProfileReport,
    ReviewPass,
    RunConfig,
    TaskSpec,
    utc_now,
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
    objection = leak_objection()
    return PipelineState(
        dataset_id="toy",
        task_description="predict churn",
        spec=TaskSpec(target="churned", task_type="binary", metric="roc_auc"),
        profile=ProfileReport(
            n_rows=200,
            n_columns=8,
            columns=[
                ColumnProfile(
                    name="account_status_code",
                    dtype="object",
                    missing_fraction=0.0,
                    n_unique=4,
                    sample_values=["CLOSED_R2", "ACTIVE_S1"],
                ),
                ColumnProfile(
                    name="monthly_charges",
                    dtype="float64",
                    missing_fraction=0.03,
                    n_unique=187,
                    sample_values=["77.72", "35.30"],
                ),
            ],
            leakage_candidates=[
                LeakageCandidate(
                    column="account_status_code",
                    reason="assigned after the churn decision, not available at predict time",
                    evidence="agrees with target on 91% of rows",
                    suspicion="high",
                )
            ],
            target_balance={"0": 0.745, "1": 0.255},
        ),
        split_artifact="art-split-1",
        feature_code_artifact="art-features-1",
        feature_summary="one-hot region and plan_tier",
        final_features=["support_tickets_90d", "tenure_months"],
        dropped_features=["account_status_code"],
        candidates=[ModelResult(name="logreg", cv_scores=[0.8, 0.82, 0.79])],
        chosen_model=ModelResult(name="logreg", cv_scores=[0.8], claimed_holdout_score=0.81),
        importance_artifact="art-importance-1",
        top_importances=[("support_tickets_90d", 0.62)],
        review_iterations=1,
        objections=[objection],
        review_passes=[
            ReviewPass(
                iteration=1,
                claim="block",
                routed_to="feature_eng",
                dispositions={objection.id: "still_open"},
                new_objection_ids=[objection.id],
            )
        ],
        reviewer_claim="block",
        review_verdict="block",
        report_artifact="art-report-1",
        planted_leakage_columns=["account_status_code"],
        verified_holdout_score=0.74,
        baseline_score=0.70,
        errors=[
            PipelineError(
                node="profiler",
                message="monthly_charges has missing values; imputed with median",
            )
        ],
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


class TestOpenObjectionsOrdering:
    def test_sorts_by_iteration_not_list_position(self):
        """Resolved in pass 1, reopened in pass 2: the objection must read as OPEN.

        `review_passes` is built with iteration=2 listed before iteration=1 to prove the
        method sorts by `iteration`, not by whatever order the list happens to be in.
        """
        objection = leak_objection()
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            objections=[objection],
            review_passes=[
                ReviewPass(
                    iteration=2,
                    claim="block",
                    routed_to="feature_eng",
                    dispositions={objection.id: "still_open"},
                ),
                ReviewPass(
                    iteration=1,
                    claim="pass",
                    routed_to="reporter",
                    dispositions={objection.id: "resolved"},
                ),
            ],
        )
        assert state.open_objections() == [objection]


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

    def test_zero_baseline_r2_gives_none_ratio_without_raising(self):
        """The predict-the-mean baseline for r2 is exactly 0.0. A ratio against it is meaningless,
        not an exception."""
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            spec=TaskSpec(target="y", task_type="regression", metric="r2"),
            verified_holdout_score=0.4,
            baseline_score=0.0,
        )
        assert state.score_ratio is None
        assert state.results_row()["score_ratio"] is None

    def test_zero_verified_rmse_gives_none_ratio_without_raising(self):
        """rmse 0.0 is what a perfect leaked copy scores. Dividing by it must not raise."""
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            spec=TaskSpec(target="y", task_type="regression", metric="rmse"),
            verified_holdout_score=0.0,
            baseline_score=1.2,
        )
        assert state.score_ratio is None
        assert state.results_row()["score_ratio"] is None


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

    def test_the_naming_condition_is_on_the_row(self):
        """A leakage number is not interpretable without it. The two naming arms run over
        byte-identical rows, so a row missing this is indistinguishable from the other arm's."""
        assert populated_state().results_row()["naming"] == "descriptive"

    def test_an_opaque_run_says_so(self):
        state = populated_state()
        state.config = RunConfig(naming="opaque")
        assert state.results_row()["naming"] == "opaque"


class TestProfilerColumnsOnTheRow:
    """The profiler's nominations, scored separately from the reviewer's objections.

    They answer different questions and on the trap fixtures they give different answers: the
    profiler nominates the planted column and the reviewer, shown the same run, says nothing. The
    name-transparency ablation moves this number and not the reviewer's, so without these fields
    the experiment's dependent variable is absent from its own results file.
    """

    def _state(self, nominated: list[str], planted: list[str]) -> PipelineState:
        return PipelineState(
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=planted,
            profile=ProfileReport(
                n_rows=200,
                n_columns=8,
                leakage_candidates=[
                    LeakageCandidate(
                        column=column, reason="looks post hoc", evidence="nmi 0.4", suspicion="high"
                    )
                    for column in nominated
                ],
            ),
        )

    def test_nominating_the_planted_column_scores_as_a_catch(self):
        row = self._state(["leaky_col"], ["leaky_col"]).results_row()
        assert row["profiler_caught"] is True
        assert row["profiler_recall"] == 1.0
        assert row["profiler_false_alarm"] == 0
        assert row["profiler_nominated"] == ["leaky_col"]

    def test_one_of_two_traps_is_half_recall(self):
        """claims_timing plants two columns, so this is a real value and not a rounding of 1.0."""
        row = self._state(["trap_a"], ["trap_a", "trap_b"]).results_row()
        assert row["profiler_recall"] == 0.5
        assert row["profiler_caught"] is True

    def test_nominating_a_clean_column_is_a_profiler_false_alarm(self):
        """The blind spot this closes: an id column flagged upstream never reached a results row,
        because the reviewer never objected to it and nothing else was counting."""
        row = self._state(["customer_id"], ["leaky_col"]).results_row()
        assert row["profiler_caught"] is False
        assert row["profiler_false_alarm"] == 1
        assert row["profiler_recall"] == 0.0

    def test_looking_and_finding_nothing_is_zero_not_null(self):
        row = self._state([], ["leaky_col"]).results_row()
        assert row["profiler_caught"] is False
        assert row["profiler_recall"] == 0.0
        assert row["profiler_nominated"] == []

    def test_a_profiler_that_never_ran_is_null_not_zero(self):
        """A node that crashed nominated nothing in a different sense than one that declined to,
        and averaging those together over a benchmark would be a lie."""
        state = PipelineState(
            dataset_id="toy", task_description="x", planted_leakage_columns=["leaky_col"]
        )
        row = state.results_row()
        assert row["profiler_nominated"] is None
        assert row["profiler_caught"] is None
        assert row["profiler_recall"] is None
        assert row["profiler_false_alarm"] is None

    def test_no_planted_columns_means_no_recall(self):
        row = self._state(["a"], []).results_row()
        assert row["profiler_recall"] is None
        assert row["profiler_false_alarm"] == 1


class TestResultsRowBranches:
    """One assertion-focused test per `results_row()` branch that a full `populated_state()`
    run can never exercise, because its fields are always set together."""

    def test_no_chosen_model_means_no_claim_or_gap(self):
        state = PipelineState(dataset_id="toy", task_description="x", verified_holdout_score=0.7)
        row = state.results_row()
        assert row["claimed_holdout_score"] is None
        assert row["holdout_claim_gap"] is None

    def test_no_planted_leakage_means_no_recall_or_remediation(self):
        state = PipelineState(dataset_id="toy", task_description="x", final_features=["a"])
        row = state.results_row()
        assert row["leakage_recall"] is None
        assert row["leakage_remediated"] is None

    def test_no_flags_means_no_precision(self):
        state = PipelineState(dataset_id="toy", task_description="x", planted_leakage_columns=["c"])
        row = state.results_row()
        assert row["leakage_precision"] is None

    def test_empty_final_features_is_not_remediation(self):
        """A crashed feature_eng leaves final_features=[], which trivially contains no planted
        column. That must not score as remediation."""
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=["leaky_col"],
            final_features=[],
        )
        row = state.results_row()
        assert row["leakage_remediated"] is None

    def test_holdout_claim_gap_is_positive_for_overstatement_in_both_directions(self):
        """Positive must always mean 'the agent overstated itself', whichever way the metric
        runs. Both states overstate by the same 0.05 margin."""
        roc_state = PipelineState(
            dataset_id="toy",
            task_description="x",
            spec=TaskSpec(target="y", task_type="binary", metric="roc_auc"),
            chosen_model=ModelResult(name="m", claimed_holdout_score=0.85),
            verified_holdout_score=0.80,
        )
        rmse_state = PipelineState(
            dataset_id="toy",
            task_description="x",
            spec=TaskSpec(target="y", task_type="regression", metric="rmse"),
            chosen_model=ModelResult(name="m", claimed_holdout_score=0.80),
            verified_holdout_score=0.85,
        )
        assert roc_state.results_row()["holdout_claim_gap"] == pytest.approx(0.05)
        assert rmse_state.results_row()["holdout_claim_gap"] == pytest.approx(0.05)

    def test_withdrawn_objection_still_counts_raised_but_not_standing(self):
        """An objection raised then withdrawn in a later pass stays in the ever-raised numbers
        but must drop out of the standing ones -- a reviewer that takes back a bad flag is
        behaving better than one that never looks again."""
        objection = leak_objection(columns=["region"])
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=["other_col"],
            objections=[objection],
            review_passes=[
                ReviewPass(
                    iteration=1,
                    claim="pass",
                    routed_to="reporter",
                    dispositions={objection.id: "withdrawn"},
                )
            ],
        )
        row = state.results_row()
        assert row["leakage_flagged"] == ["region"]
        assert row["false_alarm"] == 1
        assert row["leakage_flagged_standing"] == []
        assert row["false_alarm_standing"] == 0


# --- the placeholder guard --------------------------------------------------------------------
# Nothing structurally stopped a StubModel run from being written to a results file. These cover
# the gate that does.


def test_a_trace_containing_a_stub_event_is_not_publishable():
    state = PipelineState(dataset_id="toy", task_description="t")
    state.node_trace = [
        NodeEvent(node="intake", started=utc_now(), model="claude-haiku-4-5"),
        NodeEvent(node="profiler", started=utc_now(), model="stub"),
    ]
    ok, reason = state.publishable()
    assert ok is False
    assert "stub" in reason
    assert state.placeholder_models() == ["stub"]


def test_a_fully_real_trace_is_publishable():
    state = PipelineState(dataset_id="toy", task_description="t")
    state.node_trace = [NodeEvent(node="intake", started=utc_now(), model="claude-haiku-4-5")]
    assert state.publishable() == (True, "")
    assert state.placeholder_models() == []


def test_an_empty_trace_is_not_publishable():
    # A row with no events would report cost 0.0 and look like a free, successful run.
    ok, reason = PipelineState(dataset_id="toy", task_description="t").publishable()
    assert ok is False
    assert "nothing ran" in reason


def test_a_node_that_called_no_model_does_not_count_as_a_placeholder():
    # The router and reporter run no model, so their events carry model=None. Treating that as a
    # placeholder would make every real run unpublishable.
    state = PipelineState(dataset_id="toy", task_description="t")
    state.node_trace = [
        NodeEvent(node="intake", started=utc_now(), model="claude-haiku-4-5"),
        NodeEvent(node="router", started=utc_now(), model=None),
    ]
    assert state.publishable()[0] is True
