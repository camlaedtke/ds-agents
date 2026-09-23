"""reviewer writes `objections` (append), `reviewer_claim`, `reviewer_dispositions`, `node_trace`,
and `errors` conditionally -- never `review_verdict`, `review_iterations`, or `review_passes`,
which belong to the router.

The tests that matter most: a model failure must clear both handoff fields (the staleness bug the
plan calls out -- nothing else clears `reviewer_claim`, so a pass-2 crash would let the router
count pass 1's "block" again), and a column-scoped objection left with no surviving column must be
REJECTED, not downgraded to `other`.
"""

import json

import pytest
from conftest import FakeTools, QueuedModel, ScriptedModel

from ds_agents.fixtures import available, load_fixture
from ds_agents.nodes.reviewer import (
    BLOCK_RETRY_PREFIX,
    CLOSURE_RULE,
    REVIEWER_SYSTEM,
    WHICH_COLUMN_RULE,
    DispositionUpdate,
    ProposedObjection,
    ReviewFinding,
    reviewer,
)
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
    """One malformed objection (a column-scoped category with `columns: []`) must not fail the
    whole `ReviewFinding`; the validation belongs to the filtering loop, one objection at a time."""
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


def test_a_resolved_objection_is_not_re_shown_to_the_reviewer():
    """An objection already closed, re-presented every pass, would be re-adjudicated forever and
    closure would be a treadmill rather than a termination condition."""
    ob = objection()
    closed = ReviewPass(
        iteration=0, claim="block", routed_to="feature_eng", dispositions={ob.id: "resolved"}
    )
    model = ScriptedModel({ReviewFinding: finding()})

    reviewer(
        state(objections=[ob], review_passes=[closed], review_iterations=1),
        tools=FakeTools(),
        model=model,
    )

    (_system, user, _schema) = model.calls[0]
    assert json.loads(user)["open_objections"] == []


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


# --- the prompt condition ---------------------------------------------------------------------


def _system_for(**config) -> str:
    model = ScriptedModel({ReviewFinding: finding()})
    reviewer(
        state(config=RunConfig(reviewer_enabled=True, **config)),
        tools=FakeTools(),
        model=model,
    )
    (system, _user, _schema) = model.calls[0]
    return system


# (reviewer_prompt, objection_closure, expected_system): both are config changes, not a second
# code path, and must compose in a fixed order so a row's config always predicts its prompt.
PROMPT_VARIANT_CASES = [
    pytest.param("base", "off", REVIEWER_SYSTEM, id="base_off"),
    pytest.param("which_column", "off", REVIEWER_SYSTEM + WHICH_COLUMN_RULE, id="which_column_off"),
    pytest.param("base", "on", REVIEWER_SYSTEM + CLOSURE_RULE, id="base_on"),
    pytest.param(
        "which_column", "on", REVIEWER_SYSTEM + WHICH_COLUMN_RULE + CLOSURE_RULE, id="both_on"
    ),
]


@pytest.mark.parametrize(("reviewer_prompt", "objection_closure", "expected"), PROMPT_VARIANT_CASES)
def test_the_prompt_and_closure_conditions_compose_in_a_fixed_order(
    reviewer_prompt, objection_closure, expected
):
    system = _system_for(reviewer_prompt=reviewer_prompt, objection_closure=objection_closure)
    assert system == expected


def test_no_appended_rule_names_a_fixture_column():
    """The answer-injection guard, absent even for WHICH_COLUMN_RULE until now.

    Both appended rules claim to repair the reviewer's reasoning rather than hand it the answer.
    That claim is only worth something if it is checked: a rule that named a planted column -- or
    any column of any registered fixture -- would make its arm measure the hint instead of the
    reviewer. Iterating every fixture rather than the one under test means a future fixture cannot
    quietly turn an existing rule into a cheat sheet.
    """
    for name in available():
        manifest = load_fixture(name).manifest
        for column in manifest.planted_columns:
            assert column not in WHICH_COLUMN_RULE
            assert column not in CLOSURE_RULE


def test_the_prompt_variant_is_a_config_change_not_a_second_code_path():
    """/add-node's rule for ablations, asserted: still one node, one model call."""
    model = ScriptedModel({ReviewFinding: finding()})

    reviewer(
        state(config=RunConfig(reviewer_enabled=True, reviewer_prompt="which_column")),
        tools=FakeTools(),
        model=model,
    )

    assert len(model.calls) == 1


# --- a block with nothing to act on ------------------------------------------------------------


def bad_proposal(subcategory: str = "bogus") -> ProposedObjection:
    """A column-scoped objection naming a column the profile does not have: the filter rejects it
    outright, which is one of the three ways a `block` ends up with nothing to act on."""
    return ProposedObjection(
        category="leakage",
        subcategory=subcategory,
        target_node="feature_eng",
        columns=["not_a_real_column"],
        evidence="made up",
        severity="high",
    )


def good_proposal() -> ProposedObjection:
    return ProposedObjection(
        category="leakage",
        subcategory="status code",
        target_node="feature_eng",
        columns=["account_status_code"],
        evidence="nmi 0.518",
        severity="high",
    )


class TestABlockWithNothingToActOn:
    """The zero-objection `block` bug: the reviewer claims `block` while nothing survives for
    feature_eng or modeler to act on, the router correctly refuses it, and the run goes straight to
    the reporter having dropped nothing (`route_sequence: ["reporter"]`, `errored: true`).

    Live base rate was 1-2 runs in 10 and it was the single largest cause of failure -- and, once
    it drew 3-and-0 across the two arms of the forced-drop cell, a measurement hazard as well: it
    eats a cell's numerator without touching what the cell is measuring.

    The node now asks the router's own question (`would_be_open`) one step early and, when the
    answer is "nothing", re-asks the model ONCE with its own rejection reasons. The claim is still
    never repaired here -- a retry that produces nothing actionable falls through to the router's
    documented terminal-block path, which is why invariant 2 in this module's docstring survives.
    """

    # (findings, state_kwargs): a block whose objections don't survive to something actionable,
    # for three different reasons the retry trigger must be blind to.
    RETRY_CASES = [
        pytest.param(
            [
                finding(claim="block", objections=[bad_proposal()]),
                finding(claim="block", objections=[good_proposal()]),
            ],
            {},
            id="only_objection_filtered_away",
        ),
        pytest.param(
            [
                finding(claim="block", objections=[]),
                finding(claim="block", objections=[good_proposal()]),
            ],
            {},
            id="no_objections_at_all",
        ),
        pytest.param(
            [
                finding(
                    claim="block",
                    dispositions=[DispositionUpdate(objection_id="obj-1", disposition="resolved")],
                ),
                finding(claim="block", objections=[good_proposal()]),
            ],
            {"objections": [objection()]},
            id="dispositions_close_every_open_objection",
        ),
    ]

    @pytest.mark.parametrize(("findings", "state_kwargs"), RETRY_CASES)
    def test_a_block_with_nothing_actionable_retries_once(self, findings, state_kwargs):
        model = QueuedModel(ReviewFinding, findings)

        update = reviewer(state(**state_kwargs), tools=FakeTools(), model=model)

        assert len(model.calls) == 2
        assert update["reviewer_claim"] == "block"
        assert update["objections"][-1].columns == ["account_status_code"]

    # (model, state_kwargs, claim): the response already has something actionable, so the retry
    # trigger (the router's own `would_be_open` question) must not fire.
    NEVER_RETRY_CASES = [
        pytest.param(
            ScriptedModel({ReviewFinding: finding(claim="block", objections=[])}),
            {"objections": [objection()]},
            "block",
            id="surviving_prior_open_objection",
        ),
        pytest.param(
            ScriptedModel({ReviewFinding: finding(claim="pass")}),
            {},
            "pass",
            id="pass_claim",
        ),
        pytest.param(
            ScriptedModel({ReviewFinding: finding(claim="block", objections=[good_proposal()])}),
            {},
            "block",
            id="surviving_objection",
        ),
    ]

    @pytest.mark.parametrize(("model", "state_kwargs", "claim"), NEVER_RETRY_CASES)
    def test_a_block_with_something_actionable_never_retries(self, model, state_kwargs, claim):
        update = reviewer(state(**state_kwargs), tools=FakeTools(), model=model)

        assert len(model.calls) == 1
        assert update["reviewer_claim"] == claim

    def test_a_successful_retry_still_records_why_the_first_response_failed(self):
        """The rejection diagnostic is the only evidence of what the model actually got wrong, and
        a retry that works is exactly the case where it would otherwise be lost -- `adjudged` is
        replaced wholesale, so an error appended only for the survivor would erase it."""
        model = QueuedModel(
            ReviewFinding,
            [
                finding(claim="block", objections=[bad_proposal("adjuster touches")]),
                finding(claim="block", objections=[good_proposal()]),
            ],
        )

        update = reviewer(state(), tools=FakeTools(), model=model)

        assert any("adjuster touches" in e.message for e in update["errors"])
        assert any(BLOCK_RETRY_PREFIX in e.message for e in update["errors"])

    def test_the_retry_names_the_rejection_reason_and_the_known_columns(self):
        """The correction travels INSIDE the JSON facts block, keyed `retry_reason` and
        `known_columns`, for the same reason `feature_code` does: `tools/llm.py:_payload` locates
        the block with find("{")/rfind("}"), so prose appended after it is invisible to the stub
        and to anything else reading the same span."""
        model = QueuedModel(
            ReviewFinding,
            [
                finding(claim="block", objections=[bad_proposal("adjuster touches")]),
                finding(claim="pass"),
            ],
        )

        reviewer(state(), tools=FakeTools(), model=model)

        first_facts = json.loads(model.calls[0][1])
        retry_facts = json.loads(model.calls[1][1])
        assert "retry_reason" not in first_facts
        assert "known_columns" not in first_facts
        assert "adjuster touches" in retry_facts["retry_reason"]
        assert "account_status_code" in retry_facts["known_columns"]

    def test_the_known_columns_offered_are_the_profile_columns_minus_the_target(self):
        """The retry may only hand back data the reviewer is already shown. It names no fixture and
        no trap -- the same line WHICH_COLUMN_RULE draws between repairing a prompt and injecting
        the answer -- and in particular it never offers the target."""
        model = QueuedModel(
            ReviewFinding, [finding(claim="block", objections=[]), finding(claim="pass")]
        )

        reviewer(state(), tools=FakeTools(), model=model)

        offered = json.loads(model.calls[1][1])["known_columns"]
        assert set(offered) == {c.name for c in PROFILE.columns} - {"churned"}

    def test_the_first_call_is_byte_identical_to_a_run_that_never_retried(self):
        """The comparability guarantee: if the retry mechanism changed the first call, no reviewer
        number from a run that retried would be comparable with one that never did."""
        retrying = QueuedModel(
            ReviewFinding, [finding(claim="block", objections=[]), finding(claim="pass")]
        )
        clean = ScriptedModel({ReviewFinding: finding(claim="pass")})

        reviewer(state(), tools=FakeTools(), model=retrying)
        reviewer(state(), tools=FakeTools(), model=clean)

        assert retrying.calls[0][0] == clean.calls[0][0]  # system prompt
        assert retrying.calls[0][1] == clean.calls[0][1]  # user message

    def test_both_calls_are_billed_to_one_node_event(self):
        """One pass is one row in the cost table. The retry is a real cost and must show up, but a
        second `NodeEvent` would make a pass that happened once look like two."""
        model = QueuedModel(
            ReviewFinding, [finding(claim="block", objections=[]), finding(claim="pass")]
        )

        update = reviewer(state(), tools=FakeTools(), model=model)

        (event,) = update["node_trace"]
        assert event.input_tokens == 22  # 11 per call, both booked
        assert event.cost_usd == pytest.approx(0.0002)

    def test_a_retry_that_still_produces_nothing_falls_through_to_the_block(self):
        """The claim is never repaired here. A second empty answer leaves `block` standing so the
        router can refuse it on the record -- promoting it to `pass` would hide the failure in the
        one column (`review_verdict`) the eval reads."""
        model = QueuedModel(
            ReviewFinding,
            [finding(claim="block", objections=[]), finding(claim="block", objections=[])],
        )

        update = reviewer(state(), tools=FakeTools(), model=model)

        assert update["reviewer_claim"] == "block"
        assert update["objections"] == []
        assert any(BLOCK_RETRY_PREFIX in e.message for e in update["errors"])

    def test_a_failing_retry_keeps_the_first_response(self):
        """A retry is a repair attempt, not a new failure mode: if the second call raises, the pass
        still reports what the model actually said the first time."""
        existing = objection()
        model = QueuedModel(
            ReviewFinding,
            [
                finding(
                    claim="block",
                    dispositions=[
                        DispositionUpdate(objection_id=existing.id, disposition="resolved")
                    ],
                ),
                RuntimeError("rate limited"),
            ],
        )

        update = reviewer(state(objections=[existing]), tools=FakeTools(), model=model)

        assert update["reviewer_claim"] == "block"
        assert update["reviewer_dispositions"][existing.id] == "resolved"
        assert any("retry call failed" in e.message for e in update["errors"])

    def test_the_retry_replaces_the_first_response_wholesale(self):
        """Chosen over merging the two responses: one pass is one adjudication act, and splicing a
        claim from one call onto dispositions from another reports something the model never said.

        The consequence is deliberate and pinned here -- dispositions the first call made are gone,
        so an objection it resolved reverts to `not_reviewed` and stays open. That is conservative
        (the column stays dropped) and it agrees with the model's own standing `block`.
        """
        existing = objection()
        model = QueuedModel(
            ReviewFinding,
            [
                finding(
                    claim="block",
                    dispositions=[
                        DispositionUpdate(objection_id=existing.id, disposition="resolved")
                    ],
                ),
                finding(claim="block", objections=[good_proposal()]),
            ],
        )

        update = reviewer(state(objections=[existing]), tools=FakeTools(), model=model)

        assert update["reviewer_dispositions"][existing.id] == "not_reviewed"

    def test_the_retry_is_attempted_at_most_once(self):
        """No loop and no config knob. A model that answers empty twice has told us something, and
        a third call would just cost money to hear it again."""
        model = QueuedModel(
            ReviewFinding,
            [
                finding(claim="block", objections=[]),
                finding(claim="block", objections=[]),
                finding(claim="pass"),
            ],
        )

        reviewer(state(), tools=FakeTools(), model=model)

        assert len(model.calls) == 2
