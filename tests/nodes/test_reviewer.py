"""reviewer writes `objections` (append), `reviewer_claim`, `reviewer_dispositions`, `node_trace`,
and `errors` conditionally -- never `review_verdict`, `review_iterations`, or `review_passes`,
which belong to the router.

The tests that matter most: a model failure must clear both handoff fields (the staleness bug the
plan calls out -- nothing else clears `reviewer_claim`, so a pass-2 crash would let the router
count pass 1's "block" again), and a column-scoped objection left with no surviving column must be
REJECTED, not downgraded to `other`.
"""

import pytest
from conftest import FakeTools, ScriptedModel

from ds_agents.nodes.reviewer import DispositionUpdate, ProposedObjection, ReviewFinding, reviewer
from ds_agents.state import (
    Objection,
    PipelineState,
    ProfileReport,
    ReviewPass,
    RunConfig,
    TaskSpec,
)
from ds_agents.tools.protocol import ArtifactMeta, ArtifactPayload

pytestmark = pytest.mark.fast

PROFILE = ProfileReport(
    n_rows=200,
    n_columns=8,
    columns=[
        {"name": "customer_id", "dtype": "object", "missing_fraction": 0.0, "n_unique": 200},
        {"name": "tenure_months", "dtype": "int64", "missing_fraction": 0.0, "n_unique": 60},
        {"name": "account_status_code", "dtype": "object", "missing_fraction": 0.0, "n_unique": 4},
        {"name": "region", "dtype": "object", "missing_fraction": 0.0, "n_unique": 4},
        {"name": "churned", "dtype": "int64", "missing_fraction": 0.0, "n_unique": 2},
    ],
)


def state(**overrides) -> PipelineState:
    fields = {
        "dataset_id": "toy",
        "task_description": "Predict churned, report roc_auc.",
        "spec": TaskSpec(target="churned", task_type="binary", metric="roc_auc"),
        "profile": PROFILE,
        "config": RunConfig(reviewer_enabled=True, reviewer_sees_code=True),
    }
    fields.update(overrides)
    return PipelineState(**fields)


def objection(
    oid: str = "obj-1", target_node: str = "feature_eng", column: str = "account_status_code"
) -> Objection:
    return Objection(
        id=oid,
        category="leakage",
        subcategory="planted status code",
        target_node=target_node,
        columns=[column],
        evidence="normalized mutual information with the target is 0.518",
        severity="high",
        raised_at_iteration=0,
    )


def finding(**overrides) -> ReviewFinding:
    fields = {"claim": "pass", "objections": [], "dispositions": [], "summary": "looks fine"}
    fields.update(overrides)
    return ReviewFinding(**fields)


def feature_code_payload(truncated: bool = False) -> ArtifactPayload:
    return ArtifactPayload(
        meta=ArtifactMeta(id="art-feat", name="feature_transform.py", kind="text"),
        content="def transform(df):\n    return df\n",
        truncated=truncated,
    )


def tools_with_code(truncated: bool = False) -> FakeTools:
    return FakeTools(artifacts={"art-feat": feature_code_payload(truncated)})


# --- writes set / basic contract ------------------------------------------------------------


def test_writes_the_expected_fields_and_nothing_else():
    model = ScriptedModel({ReviewFinding: finding(claim="pass")})

    update = reviewer(state(), tools=FakeTools(), model=model)

    assert set(update) == {"objections", "reviewer_claim", "reviewer_dispositions", "node_trace"}


def test_never_writes_verdict_or_counters_or_passes():
    model = ScriptedModel({ReviewFinding: finding(claim="block", objections=[])})

    update = reviewer(
        state(objections=[objection()], review_iterations=1), tools=FakeTools(), model=model
    )

    assert "review_verdict" not in update
    assert "review_iterations" not in update
    assert "review_passes" not in update


def test_exactly_one_model_call():
    model = ScriptedModel({ReviewFinding: finding()})

    reviewer(state(), tools=FakeTools(), model=model)

    assert len(model.calls) == 1


# --- id / iteration stamping ------------------------------------------------------------------


def test_the_node_stamps_id_and_the_pre_increment_iteration():
    proposal = ProposedObjection(
        category="leakage",
        subcategory="status code",
        target_node="feature_eng",
        columns=["account_status_code"],
        evidence="nmi 0.518",
        severity="high",
    )
    model = ScriptedModel({ReviewFinding: finding(claim="block", objections=[proposal])})

    update = reviewer(state(review_iterations=2), tools=FakeTools(), model=model)

    (raised,) = update["objections"]
    assert raised.id  # a fresh id was stamped
    assert raised.raised_at_iteration == 2  # pre-increment: the router has not counted this pass


# --- column filtering ---------------------------------------------------------------------------


def test_column_scoped_objection_with_no_surviving_column_is_rejected_with_an_error():
    proposal = ProposedObjection(
        category="leakage",
        subcategory="bogus",
        target_node="feature_eng",
        columns=["not_a_real_column"],
        evidence="made up",
        severity="high",
    )
    model = ScriptedModel({ReviewFinding: finding(claim="block", objections=[proposal])})

    update = reviewer(state(), tools=FakeTools(), model=model)

    assert update["objections"] == []
    assert any("rejected" in e.message for e in update["errors"])


def test_unknown_columns_are_filtered_out_but_a_known_one_survives():
    proposal = ProposedObjection(
        category="leakage",
        subcategory="status code",
        target_node="feature_eng",
        columns=["account_status_code", "not_a_real_column"],
        evidence="nmi 0.518",
        severity="high",
    )
    model = ScriptedModel({ReviewFinding: finding(claim="block", objections=[proposal])})

    update = reviewer(state(), tools=FakeTools(), model=model)

    (raised,) = update["objections"]
    assert raised.columns == ["account_status_code"]
    assert any("dropped" in e.message for e in update["errors"])


def test_the_target_column_is_filtered_out_like_any_unknown_column():
    proposal = ProposedObjection(
        category="leakage",
        subcategory="target named directly",
        target_node="feature_eng",
        columns=["churned"],
        evidence="the model named the target itself",
        severity="high",
    )
    model = ScriptedModel({ReviewFinding: finding(claim="block", objections=[proposal])})

    update = reviewer(state(), tools=FakeTools(), model=model)

    assert update["objections"] == []
    assert any("rejected" in e.message for e in update["errors"])


def test_a_column_scoped_objection_naming_no_columns_at_all_does_not_sink_the_pass():
    """The live failure of 2026-08-27: Haiku raised `implausible_importance` with `columns: []`.

    While `ProposedObjection` carried `Objection`'s column validator, that one malformed objection
    failed the whole `ReviewFinding`, and a pass that found something real read as a crash. The
    rule belongs to the filtering loop, one objection at a time.
    """
    bad = ProposedObjection(
        category="implausible_importance",
        subcategory="named no column",
        target_node="modeler",
        columns=[],
        evidence="importances look wrong but I named nothing",
        severity="high",
    )
    good = ProposedObjection(
        category="leakage",
        subcategory="status code",
        target_node="feature_eng",
        columns=["account_status_code"],
        evidence="nmi 0.518",
        severity="high",
    )
    model = ScriptedModel({ReviewFinding: finding(claim="block", objections=[bad, good])})

    update = reviewer(state(), tools=FakeTools(), model=model)

    assert update["reviewer_claim"] == "block"
    (raised,) = update["objections"]
    assert raised.columns == ["account_status_code"]
    assert any("rejected" in e.message for e in update["errors"])


def test_a_non_column_scoped_objection_survives_with_empty_columns():
    proposal = ProposedObjection(
        category="metric_mismatch",
        subcategory="wrong metric for the task",
        target_node="modeler",
        columns=[],
        evidence="binary task scored with rmse",
        severity="medium",
    )
    model = ScriptedModel({ReviewFinding: finding(claim="block", objections=[proposal])})

    update = reviewer(state(), tools=FakeTools(), model=model)

    (raised,) = update["objections"]
    assert raised.columns == []
    assert "errors" not in update


# --- dispositions --------------------------------------------------------------------------------


def test_unknown_or_closed_disposition_ids_are_dropped():
    open_obj = objection(oid="open-1")
    closed_obj = objection(oid="closed-1", column="region")
    model = ScriptedModel(
        {
            ReviewFinding: finding(
                claim="pass",
                dispositions=[
                    DispositionUpdate(objection_id="open-1", disposition="resolved"),
                    DispositionUpdate(objection_id="closed-1", disposition="resolved"),
                    DispositionUpdate(objection_id="never-existed", disposition="withdrawn"),
                ],
            )
        }
    )
    st = state(
        objections=[open_obj, closed_obj],
        review_passes=[
            ReviewPass(
                iteration=1,
                claim="pass",
                routed_to="reporter",
                dispositions={"closed-1": "resolved"},
            )
        ],
    )

    update = reviewer(st, tools=FakeTools(), model=model)

    assert update["reviewer_dispositions"] == {"open-1": "resolved"}


def test_every_open_objection_gets_a_disposition_defaulting_to_not_reviewed():
    open_obj = objection(oid="open-1")
    model = ScriptedModel({ReviewFinding: finding(claim="pass", dispositions=[])})

    update = reviewer(state(objections=[open_obj]), tools=FakeTools(), model=model)

    assert update["reviewer_dispositions"] == {"open-1": "not_reviewed"}


# --- feature code -----------------------------------------------------------------------------


def test_feature_code_is_read_when_sees_code_is_true():
    model = ScriptedModel({ReviewFinding: finding()})

    reviewer(
        state(config=RunConfig(reviewer_sees_code=True), feature_code_artifact="art-feat"),
        tools=tools_with_code(),
        model=model,
    )

    (_system, user, _schema) = model.calls[0]
    assert "def transform(df):" in user


def test_feature_code_is_not_read_when_sees_code_is_false():
    model = ScriptedModel({ReviewFinding: finding()})

    reviewer(
        state(config=RunConfig(reviewer_sees_code=False), feature_code_artifact="art-feat"),
        tools=tools_with_code(),
        model=model,
    )

    (_system, user, _schema) = model.calls[0]
    assert "def transform(df):" not in user
    assert "withheld by config" in user


def test_a_truncated_feature_code_artifact_is_omitted_and_recorded():
    model = ScriptedModel({ReviewFinding: finding()})

    update = reviewer(
        state(config=RunConfig(reviewer_sees_code=True), feature_code_artifact="art-feat"),
        tools=tools_with_code(truncated=True),
        model=model,
    )

    (_system, user, _schema) = model.calls[0]
    assert "def transform(df):" not in user
    assert any("truncated" in e.message for e in update["errors"])
    assert update["reviewer_claim"] == "pass"  # the pass continues; not a failure return


def test_a_read_tool_error_does_not_sink_the_pass():
    model = ScriptedModel({ReviewFinding: finding()})
    tools = FakeTools()  # no "art-feat" registered -> ToolError

    update = reviewer(
        state(config=RunConfig(reviewer_sees_code=True), feature_code_artifact="art-feat"),
        tools=tools,
        model=model,
    )

    assert update["reviewer_claim"] == "pass"
    assert any("could not read feature code artifact" in e.message for e in update["errors"])


# --- prompt facts -----------------------------------------------------------------------------


def test_open_objection_ids_reach_the_prompt():
    model = ScriptedModel({ReviewFinding: finding()})
    st = state(objections=[objection(oid="obj-777")])

    reviewer(st, tools=FakeTools(), model=model)

    (_system, user, _schema) = model.calls[0]
    assert "obj-777" in user


# --- failure paths ----------------------------------------------------------------------------


def test_a_model_failure_clears_both_handoff_fields_and_keeps_the_trace_event():
    """The staleness guard: nothing else clears `reviewer_claim`, so a crashed pass-2 must not let
    the router re-count pass 1's stale 'block'."""
    model = ScriptedModel({ReviewFinding: TimeoutError("read timed out")})

    update = reviewer(
        state(reviewer_claim="block", reviewer_dispositions={"x": "still_open"}),
        tools=FakeTools(),
        model=model,
    )

    assert update["reviewer_claim"] is None
    assert update["reviewer_dispositions"] == {}
    (event,) = update["node_trace"]
    assert event.node == "reviewer"
    assert update["errors"]


def test_no_spec_or_profile_is_unrecoverable_and_clears_handoff_fields():
    st = state(spec=None)

    update = reviewer(st, tools=FakeTools(), model=ScriptedModel({}))

    assert update["reviewer_claim"] is None
    assert update["reviewer_dispositions"] == {}
    assert update["errors"][0].recoverable is False


def test_disabled_reviewer_is_a_zero_cost_no_op():
    model = ScriptedModel({})

    update = reviewer(
        state(config=RunConfig(reviewer_enabled=False), reviewer_claim="block"),
        tools=FakeTools(),
        model=model,
    )

    assert set(update) == {"reviewer_claim", "reviewer_dispositions", "node_trace"}
    assert update["reviewer_claim"] is None
    assert update["reviewer_dispositions"] == {}
    (event,) = update["node_trace"]
    assert event.cost_usd == 0.0
    assert event.model is None
    assert model.calls == []
