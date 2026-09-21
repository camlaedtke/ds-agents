"""`capture.py`: freezing a finished run into a viewer-facing `Capture`, and replaying its trace
into fidelity-labelled `Step`s.

`timeline()` is the subtle half -- there is no checkpointer, so an earlier occurrence of a node
that ran again has no persisted intermediate value, and the whole point of the fidelity label is
that the viewer must say so rather than silently show the final value in its place. The hand-built
looping state below is the fixture that actually exercises that: two full review cycles
(feature_eng/modeler/reviewer/router, twice) so a real "not_recorded then reconstructed" pair
exists to assert against, alongside the router/reviewer alignment the docstring describes.

`envelope()` is the simpler half operationally (read some files, dump the state) but is the one
that touches the filesystem and the fixture registry, so its test lives on `tmp_path` and rebuilds
a fake `<run_root>/artifacts/` by hand.
"""

from datetime import UTC, datetime, timedelta

import pytest

from ds_agents.capture import (
    ARTIFACT_TEXT_LIMIT,
    NODE_READS,
    NODE_WRITES,
    Capture,
    envelope,
    timeline,
)
from ds_agents.state import (
    ModelResult,
    NodeEvent,
    Objection,
    PipelineError,
    PipelineState,
    ProfileReport,
    ReviewPass,
    RunConfig,
    TaskSpec,
)

pytestmark = pytest.mark.fast

BASE = datetime(2026, 9, 1, tzinfo=UTC)


def _event(node: str, index: int) -> NodeEvent:
    """One trace event, `index` seconds after `BASE`, one second long. Router events carry no
    model, matching `nodes/router.py`'s `NodeRun` (the router never calls a model)."""
    started = BASE + timedelta(seconds=index)
    return NodeEvent(
        node=node,
        started=started,
        ended=started + timedelta(seconds=1),
        model=None if node == "router" else "claude-haiku-4-5",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0001,
    )


LOOP_NODE_ORDER = [
    "intake",
    "profiler",
    "feature_eng",
    "modeler",
    "reviewer",
    "router",
    "feature_eng",
    "modeler",
    "reviewer",
    "router",
    "reporter",
]


def looping_state() -> tuple[PipelineState, Objection]:
    """Two full review cycles: pass 1 blocks back to feature_eng, pass 2 passes to reporter.

    One objection, raised on the first reviewer occurrence (`raised_at_iteration=0`), still
    binding on feature_eng at the end (its latest disposition is `resolved`, which does not
    release it under the default `forced_drop_release="withdrawn_only"`) -- so the final
    `feature_eng` step's `forced_drop_columns` is non-empty, matching what a real block-then-fix
    run looks like.
    """
    objection = Objection(
        category="leakage",
        subcategory="post_hoc_status_code",
        target_node="feature_eng",
        columns=["account_status_code"],
        evidence="agrees with target on 91% of rows",
        severity="high",
        raised_at_iteration=0,
    )
    trace = [_event(node, i) for i, node in enumerate(LOOP_NODE_ORDER)]
    state = PipelineState(
        dataset_id="toy",
        task_description="predict churn",
        spec=TaskSpec(target="churned", task_type="binary", metric="roc_auc"),
        profile=ProfileReport(n_rows=200, n_columns=8),
        split_artifact="art-001-split",
        feature_code_artifact="art-002-features",
        feature_summary="one-hot region and plan_tier",
        final_features=["tenure_months", "monthly_charges"],
        dropped_features=["account_status_code"],
        candidates=[ModelResult(name="logreg", cv_scores=[0.8, 0.82])],
        chosen_model=ModelResult(name="logreg", cv_scores=[0.8, 0.82], claimed_holdout_score=0.81),
        importance_artifact="art-003-importance",
        top_importances=[("tenure_months", 0.5)],
        review_iterations=2,
        objections=[objection],
        review_passes=[
            ReviewPass(
                iteration=1,
                claim="block",
                routed_to="feature_eng",
                dispositions={objection.id: "still_open"},
                new_objection_ids=[objection.id],
            ),
            ReviewPass(
                iteration=2,
                claim="pass",
                routed_to="reporter",
                dispositions={objection.id: "resolved"},
                new_objection_ids=[],
            ),
        ],
        reviewer_claim="pass",
        review_verdict="pass",
        report_artifact="art-004-report",
        node_trace=trace,
    )
    return state, objection


class TestNodeWritesPinsRealFields:
    def test_every_write_is_a_real_pipelinestate_field(self):
        fields = set(PipelineState.model_fields)
        for node, names in NODE_WRITES.items():
            for name in names:
                assert name in fields, f"NODE_WRITES[{node!r}] names {name!r}, not a real field"


class TestTimelineOnALoopingState:
    """The two-cycle fixture above, walked end to end."""

    def test_index_and_node_order(self):
        state, _ = looping_state()
        steps = timeline(state)
        assert [s.node for s in steps] == LOOP_NODE_ORDER
        assert [s.index for s in steps] == list(range(len(LOOP_NODE_ORDER)))

    def test_occurrence_and_last_occurrence(self):
        state, _ = looping_state()
        steps = timeline(state)
        fe1, fe2 = steps[2], steps[6]
        mo1, mo2 = steps[3], steps[7]
        assert (fe1.occurrence, fe1.is_last_occurrence) == (1, False)
        assert (fe2.occurrence, fe2.is_last_occurrence) == (2, True)
        assert (mo1.occurrence, mo1.is_last_occurrence) == (1, False)
        assert (mo2.occurrence, mo2.is_last_occurrence) == (2, True)
        # Single-occurrence nodes are trivially their own last occurrence.
        assert steps[0].is_last_occurrence is True  # intake
        assert steps[10].is_last_occurrence is True  # reporter

    def test_iterations_match_router_events_before_each_step(self):
        state, _ = looping_state()
        steps = timeline(state)
        # 0 router events precede anything up to and including the first router step (index 5);
        # 1 router event (index 5 itself) precedes everything from index 6 through the second
        # router step (index 9); 2 precede the reporter.
        expected = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2]
        assert [s.iteration for s in steps] == expected
        # Pinned explicitly, matching the docstring: reviewer #1 -> 0, reviewer #2 -> 1.
        assert steps[4].iteration == 0
        assert steps[8].iteration == 1

    def test_feature_eng_fidelity_and_writes_values_gating(self):
        state, _ = looping_state()
        steps = timeline(state)
        fe1, fe2 = steps[2], steps[6]
        assert fe1.fidelity == "not_recorded"
        assert fe1.writes_values is None
        assert fe2.fidelity == "reconstructed"
        assert fe2.writes_values is not None
        assert fe2.writes_values["final_features"] == ["tenure_months", "monthly_charges"]
        assert fe2.writes_values["dropped_features"] == ["account_status_code"]
        assert set(fe2.writes) == set(NODE_WRITES["feature_eng"])
        assert set(fe2.reads) == set(NODE_READS["feature_eng"])

    def test_feature_eng_last_occurrence_detail_names_the_forced_drop(self):
        state, objection = looping_state()
        steps = timeline(state)
        detail = steps[6].detail
        assert detail is not None
        assert detail["forced_drop_columns"] == objection.columns

    def test_a_non_column_scoped_objection_naming_a_column_is_not_a_forced_drop(self):
        """`feature_eng._forced_drops` only acts on COLUMN_SCOPED_CATEGORIES, so the replay
        must apply the same filter: an `overfit` objection that happens to name a column never
        forced anything out of the matrix, and displaying it as forced would be a false claim
        about what the node did."""
        state, objection = looping_state()
        state = state.model_copy(
            update={
                "objections": [
                    objection,
                    Objection(
                        category="overfit",
                        subcategory="cv_holdout_spread",
                        target_node="feature_eng",
                        columns=["tenure_months"],
                        evidence="cv mean 0.82 vs claimed 0.95",
                        severity="medium",
                        raised_at_iteration=0,
                    ),
                ]
            }
        )
        steps = timeline(state)
        detail = steps[6].detail
        assert detail is not None
        assert detail["forced_drop_columns"] == objection.columns
        assert "tenure_months" not in detail["forced_drop_columns"]

    def test_reviewer_detail_and_headline_every_occurrence(self):
        state, objection = looping_state()
        steps = timeline(state)
        reviewer1, reviewer2 = steps[4], steps[8]

        assert reviewer1.fidelity == "recorded"
        assert reviewer2.fidelity == "recorded"

        raised1 = reviewer1.detail["objections_raised"]
        assert [o["id"] for o in raised1] == [objection.id]
        assert reviewer2.detail["objections_raised"] == []

        assert "1 objection" in reviewer1.headline
        assert "block" in reviewer1.headline
        assert "nothing" in reviewer2.headline
        assert "pass" in reviewer2.headline

    def test_router_verdicts_and_pass_alignment(self):
        state, objection = looping_state()
        steps = timeline(state)
        router1, router2 = steps[5], steps[9]

        assert router1.fidelity == "recorded"
        assert router1.verdict == "block"
        assert router1.routed_to == "feature_eng"
        assert router1.dispositions == {objection.id: "still_open"}
        assert router1.new_objection_ids == [objection.id]

        assert router2.verdict == "pass"
        assert router2.routed_to == "reporter"
        assert router2.dispositions == {objection.id: "resolved"}
        assert router2.new_objection_ids == []


class TestTimelineOnAHaltedState:
    """intake ran, blew up fatally, and the reporter still produced a row. No spec was ever set."""

    def test_no_crash_error_attribution_and_degraded_headlines(self):
        trace = [_event("intake", 0), _event("reporter", 1)]
        state = PipelineState(
            dataset_id="toy",
            task_description="predict churn",
            node_trace=trace,
            errors=[
                PipelineError(node="intake", message="could not infer a target", recoverable=False)
            ],
        )

        steps = timeline(state)

        assert len(steps) == 2
        intake_step, reporter_step = steps
        assert len(intake_step.errors) == 1
        assert intake_step.errors[0]["node"] == "intake"
        assert intake_step.errors[0]["recoverable"] is False
        assert reporter_step.errors == []
        # A halted run has spec=None; the headline must degrade, not crash or fabricate a target.
        assert "not recorded" in intake_step.headline or "no result" in intake_step.headline
        assert "not recorded" in reporter_step.headline or "no result" in reporter_step.headline

    def test_an_error_naming_a_node_with_no_trace_event_is_skipped_not_crashed(self):
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            node_trace=[_event("intake", 0)],
            errors=[PipelineError(node="modeler", message="never got here", recoverable=False)],
        )
        steps = timeline(state)
        assert len(steps) == 1
        assert steps[0].errors == []


class TestRouterTolerance:
    def test_no_claim_and_zero_passes_is_pending(self):
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            node_trace=[_event("router", 0)],
        )
        step = timeline(state)[0]
        assert step.verdict == "pending"
        assert step.routed_to is None
        assert step.dispositions == {}
        assert step.new_objection_ids == []

    def test_a_block_at_the_cap_is_exhausted(self):
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            config=RunConfig(loop_cap=2),
            review_passes=[
                ReviewPass(iteration=2, claim="block", routed_to="reporter", dispositions={})
            ],
            node_trace=[_event("router", 0)],
        )
        step = timeline(state)[0]
        assert step.verdict == "exhausted"
        assert step.routed_to == "reporter"


class TestEnvelope:
    def test_artifacts_matched_binary_truncated_and_round_trip(self, tmp_path):
        state = PipelineState(
            dataset_id="toy",
            task_description="predict churn",
            split_artifact="art-001-split",
            node_trace=[_event("intake", 0)],
        )
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "art-001-split-split.json").write_text('{"rows": [1, 2, 3]}')
        (artifacts_dir / "notes.txt").write_text("an uncited intermediate the snippet dropped")
        (artifacts_dir / "blob.bin").write_bytes(b"\xff\xfe\x00\x01\x02")
        big_text = "x" * (ARTIFACT_TEXT_LIMIT + 500)
        (artifacts_dir / "big.txt").write_text(big_text)

        cap = envelope(state, tmp_path, note="kept because it looped twice")

        by_name = {a.filename: a for a in cap.artifacts}
        assert len(by_name) == 4
        assert by_name["art-001-split-split.json"].artifact_id == "art-001-split"
        # Uncited: keyed by its own filename.
        assert by_name["notes.txt"].artifact_id == "notes.txt"
        assert by_name["notes.txt"].binary is False
        assert by_name["blob.bin"].binary is True
        assert by_name["blob.bin"].text == ""
        assert by_name["big.txt"].truncated is True
        assert len(by_name["big.txt"].text) == ARTIFACT_TEXT_LIMIT
        assert by_name["art-001-split-split.json"].truncated is False

        assert cap.dataset_id == "toy"
        assert cap.note == "kept because it looped twice"
        assert cap.results_row["dataset_id"] == "toy"
        assert cap.node_seconds.get("intake") == pytest.approx(1.0)
        # "toy" is a real fixture, and the default `RunConfig.dataset_source` is "fixture".
        assert cap.fixture_manifest is not None
        assert cap.fixture_manifest["target"] == "churned"

        round_tripped = Capture.model_validate_json(cap.model_dump_json())
        rebuilt_state = PipelineState.from_dump(round_tripped.state)
        assert rebuilt_state == state

    def test_missing_artifacts_directory_is_tolerated(self, tmp_path):
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            config=RunConfig(dataset_source="benchmark"),
            node_trace=[_event("intake", 0)],
        )
        cap = envelope(state, tmp_path)  # no artifacts/ dir under tmp_path at all
        assert cap.artifacts == []
        assert cap.fixture_manifest is None

    def test_benchmark_run_has_no_fixture_manifest(self, tmp_path):
        state = PipelineState(
            dataset_id="claims_timing",
            task_description="x",
            config=RunConfig(dataset_source="benchmark"),
            node_trace=[_event("intake", 0)],
        )
        cap = envelope(state, tmp_path)
        assert cap.fixture_manifest is None
