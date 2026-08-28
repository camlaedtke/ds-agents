"""router writes `review_verdict` (always) and `review_iterations` + `review_passes` (only on a
completed pass). It is the sole authority that turns a claim into a `routed_to` destination:
`route_target` reads that decision back off the latest `ReviewPass` rather than re-deriving it.

The tests that matter here are about the two off-by-one-shaped rules in router.py's docstring --
the counter increments BEFORE the cap is checked, and a claim of `block` that lands exactly at
the cap becomes `exhausted`, never `pass` -- plus the newer ones from the `ReviewPass` ownership
fix: the router mints exactly one pass per completed invocation, `routed_to` is computed against
the PROJECTED post-pass open set (this pass's dispositions applied on top of what was open going
in), and `new_objection_ids` matches the reviewer's pre-increment clock, not the router's own
post-increment counter.
"""

from typing import Literal

import pytest
from conftest import FakeTools, ScriptedModel

from ds_agents.nodes.router import route_target, router
from ds_agents.state import (
    Disposition,
    Objection,
    ObjectionCategory,
    ObjectionRouting,
    PipelineState,
    ReviewPass,
    ReviewVerdict,
    RoutableNode,
    RunConfig,
)

pytestmark = pytest.mark.fast


def state(
    *,
    reviewer_claim: Literal["pass", "block"] | None = None,
    review_iterations: int = 0,
    loop_cap: int = 3,
    reviewer_enabled: bool = True,
    objections: list[Objection] | None = None,
    review_passes: list[ReviewPass] | None = None,
    review_verdict: ReviewVerdict = "pending",
    reviewer_dispositions: dict[str, Disposition] | None = None,
    objection_routing: ObjectionRouting = "as_addressed",
) -> PipelineState:
    return PipelineState(
        dataset_id="toy",
        task_description="Predict churned, report roc_auc.",
        config=RunConfig(
            loop_cap=loop_cap,
            reviewer_enabled=reviewer_enabled,
            objection_routing=objection_routing,
        ),
        reviewer_claim=reviewer_claim,
        review_iterations=review_iterations,
        objections=objections or [],
        review_passes=review_passes or [],
        review_verdict=review_verdict,
        reviewer_dispositions=reviewer_dispositions or {},
    )


def objection(
    target_node: RoutableNode = "feature_eng",
    column: str = "account_status_code",
    raised_at_iteration: int = 0,
    category: ObjectionCategory = "leakage",
) -> Objection:
    return Objection(
        category=category,
        subcategory="planted status code",
        target_node=target_node,
        columns=[column],
        evidence="normalized mutual information with the target is 0.518",
        severity="high",
        raised_at_iteration=raised_at_iteration,
    )


def tools() -> FakeTools:
    return FakeTools()


def model() -> ScriptedModel:
    return ScriptedModel({})


def applied(state_before: PipelineState, update: dict) -> PipelineState:
    """Merge a router update onto the state that produced it, the way LangGraph would (for
    `review_passes`, `PipelineState` uses `operator.add`, but a fresh test state starts empty so
    replacing is equivalent to appending)."""
    return state_before.model_copy(update=update)


# --- router() -----------------------------------------------------------------------------


def test_no_claim_does_not_increment_and_leaves_pending():
    update = router(
        state(reviewer_claim=None, reviewer_enabled=False), tools=tools(), model=model()
    )

    assert set(update) == {"review_verdict", "node_trace"}
    assert update["review_verdict"] == "pending"
    assert "review_iterations" not in update
    assert "review_passes" not in update


def test_no_claim_with_the_reviewer_enabled_is_an_error():
    update = router(
        state(reviewer_claim=None, reviewer_enabled=True, loop_cap=3), tools=tools(), model=model()
    )

    assert update["review_verdict"] == "pending"
    (error,) = update["errors"]
    assert error.node == "router"
    assert "no claim" in error.message


def test_no_claim_with_the_reviewer_disabled_is_not_an_error():
    """Distinguishes 'off' from 'crashed': the reviewer-disabled arm must not read as a failure."""
    update = router(
        state(reviewer_claim=None, reviewer_enabled=False), tools=tools(), model=model()
    )

    assert update["review_verdict"] == "pending"
    assert "errors" not in update


def test_a_pass_increments_mints_one_pass_routed_to_the_reporter():
    update = router(
        state(reviewer_claim="pass", review_iterations=0, loop_cap=3), tools=tools(), model=model()
    )

    assert update["review_iterations"] == 1
    assert update["review_verdict"] == "pass"
    assert "errors" not in update
    (pass_,) = update["review_passes"]
    assert pass_.iteration == 1
    assert pass_.claim == "pass"
    assert pass_.routed_to == "reporter"


def test_a_block_below_the_cap_is_a_block():
    update = router(
        state(
            reviewer_claim="block",
            review_iterations=1,
            loop_cap=3,
            objections=[objection()],
        ),
        tools=tools(),
        model=model(),
    )

    assert update["review_iterations"] == 2
    assert update["review_verdict"] == "block"
    assert "errors" not in update
    (pass_,) = update["review_passes"]
    assert pass_.routed_to == "feature_eng"


def test_a_block_at_the_cap_becomes_exhausted_and_still_mints_a_pass_to_the_reporter():
    update = router(
        state(reviewer_claim="block", review_iterations=2, loop_cap=3), tools=tools(), model=model()
    )

    assert update["review_iterations"] == 3
    assert update["review_verdict"] == "exhausted"
    # No open-objection check on this branch: a reviewer that hits the cap is exhausted
    # regardless of whether it left anything actionable behind.
    assert "errors" not in update
    (pass_,) = update["review_passes"]
    assert pass_.claim == "block"
    assert pass_.routed_to == "reporter"


def test_the_router_never_calls_a_model():
    m = ScriptedModel({})

    router(state(reviewer_claim="block", objections=[objection()]), tools=tools(), model=m)

    assert m.calls == []


def test_the_router_never_writes_reviewer_dispositions():
    update = router(
        state(reviewer_claim="pass", reviewer_dispositions={"x": "resolved"}),
        tools=tools(),
        model=model(),
    )

    assert "reviewer_dispositions" not in update


def test_dispositions_land_on_the_pass_verbatim():
    update = router(
        state(
            reviewer_claim="block",
            objections=[objection(), objection(column="region")],
            reviewer_dispositions={"a": "resolved", "b": "still_open"},
        ),
        tools=tools(),
        model=model(),
    )

    (pass_,) = update["review_passes"]
    assert pass_.dispositions == {"a": "resolved", "b": "still_open"}


def test_new_objection_ids_use_the_reviewers_pre_increment_clock():
    """The reviewer stamps `raised_at_iteration` from `state.review_iterations` BEFORE this call
    increments it. An objection from this pass therefore matches the pre-increment value, not the
    `review_iterations` the router writes to the update."""
    this_pass = objection(raised_at_iteration=1)
    older = objection(raised_at_iteration=0, column="region")

    update = router(
        state(reviewer_claim="block", review_iterations=1, objections=[older, this_pass]),
        tools=tools(),
        model=model(),
    )

    (pass_,) = update["review_passes"]
    assert pass_.iteration == 2  # post-increment
    assert pass_.new_objection_ids == [this_pass.id]


def test_the_reviewer_cannot_raise_its_own_cap():
    """`loop_cap=2` permits at most two completed reviewer invocations. Round 1 still has budget
    and comes back as `block`; round 2 lands exactly on the cap and is forced to `exhausted`. The
    reviewer wants a third round -- it keeps claiming `block` -- but `route_target` refuses to
    send the run back to it once the verdict is `exhausted`, so a third `router` invocation never
    happens and `review_iterations` never climbs past the cap it was supposed to enforce."""
    round1 = router(
        state(reviewer_claim="block", review_iterations=0, loop_cap=2, objections=[objection()]),
        tools=tools(),
        model=model(),
    )
    assert round1["review_verdict"] == "block"
    assert round1["review_iterations"] == 1

    round2 = router(
        state(
            reviewer_claim="block",
            review_iterations=round1["review_iterations"],
            loop_cap=2,
            objections=[objection()],
        ),
        tools=tools(),
        model=model(),
    )
    assert round2["review_verdict"] == "exhausted"
    assert round2["review_iterations"] == 2

    # The reviewer still "wants" a third block; the graph structurally will not ask it again.
    after_round2 = applied(state(objections=[objection()]), round2)
    assert route_target(after_round2) == "reporter"
    assert round2["review_iterations"] <= 2


def test_a_block_with_no_open_objection_is_an_error_and_the_pass_routes_to_the_reporter():
    update = router(
        state(reviewer_claim="block", review_iterations=0, loop_cap=3, objections=[]),
        tools=tools(),
        model=model(),
    )

    assert update["review_verdict"] == "block"
    (error,) = update["errors"]
    assert "no open objection" in error.message
    (pass_,) = update["review_passes"]
    assert pass_.routed_to == "reporter"


def test_a_block_that_resolves_its_last_open_objection_this_pass_routes_to_the_reporter():
    """The projection case: `open_objections()` alone (pre-pass) would still say this objection is
    open, since nothing has been committed to `review_passes` yet. The router has to apply
    `reviewer_dispositions` on top of that to see there is nothing left, and the same
    no-open-objection error fires -- it is keyed off the projection now, not the raw set."""
    ob = objection()
    update = router(
        state(
            reviewer_claim="block",
            objections=[ob],
            reviewer_dispositions={ob.id: "resolved"},
        ),
        tools=tools(),
        model=model(),
    )

    (pass_,) = update["review_passes"]
    assert pass_.routed_to == "reporter"
    assert any("no open objection" in e.message for e in update["errors"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reviewer_claim": None},
        {"reviewer_claim": "pass", "review_iterations": 0, "loop_cap": 3},
        {
            "reviewer_claim": "block",
            "review_iterations": 0,
            "loop_cap": 3,
            "objections": [objection()],
        },
        {"reviewer_claim": "block", "review_iterations": 2, "loop_cap": 3},
        {"reviewer_claim": "block", "review_iterations": 0, "loop_cap": 3, "objections": []},
    ],
    ids=["no_claim", "pass", "block_below_cap", "block_at_cap", "block_no_objection"],
)
def test_the_trace_event_is_written_on_every_branch(kwargs):
    update = router(state(**kwargs), tools=tools(), model=model())

    (event,) = update["node_trace"]
    assert event.node == "router"
    assert event.cost_usd == 0.0
    assert event.model is None


# --- the dual-authority guard: routed_to == where route_target actually sends the run ------------


def test_a_rerouted_objection_resolved_this_pass_still_routes_to_the_reporter():
    """`_route_for_block` now asks `open_objections(destination)`, which applies the routing
    condition. This pins that the rewrite kept the disposition projection AHEAD of the reroute:
    an objection the reviewer closed in this very pass must not drag the run upstream just
    because `by_category` would have sent it there had it still been open."""
    ob = objection(target_node="modeler", category="implausible_importance")
    update = router(
        state(
            reviewer_claim="block",
            objections=[ob],
            reviewer_dispositions={ob.id: "resolved"},
            objection_routing="by_category",
        ),
        tools=tools(),
        model=model(),
    )

    (pass_,) = update["review_passes"]
    assert pass_.routed_to == "reporter"
    assert any("no open objection" in e.message for e in update["errors"])


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"reviewer_claim": "pass"}, "reporter"),
        (
            {
                "reviewer_claim": "block",
                "objections": [
                    objection(target_node="modeler"),
                    objection(target_node="feature_eng"),
                ],
            },
            "feature_eng",
        ),
        ({"reviewer_claim": "block", "objections": [objection(target_node="modeler")]}, "modeler"),
        ({"reviewer_claim": "block", "review_iterations": 2, "loop_cap": 3}, "reporter"),
        ({"reviewer_claim": "block", "objections": []}, "reporter"),
        # The live misrouting, both ways round. Same objection, same reviewer; the condition is
        # the only thing that differs, and it is what decides whether any node can act.
        (
            {
                "reviewer_claim": "block",
                "objections": [objection(target_node="modeler", category="implausible_importance")],
            },
            "modeler",
        ),
        (
            {
                "reviewer_claim": "block",
                "objections": [objection(target_node="modeler", category="implausible_importance")],
                "objection_routing": "by_category",
            },
            "feature_eng",
        ),
    ],
    ids=[
        "pass",
        "feature_eng_first",
        "modeler_only",
        "exhausted",
        "block_no_objection",
        "misaddressed_as_addressed",
        "misaddressed_by_category",
    ],
)
def test_routed_to_matches_where_route_target_actually_sends_the_run(kwargs, expected):
    """Applies the router's own update to the state that produced it, then calls the edge function
    on the result -- the same two-step the graph actually takes. A router that computed one
    destination while `route_target` (independently) computed another would pass neither half of
    this test by accident."""
    before = state(**kwargs)
    update = router(before, tools=tools(), model=model())
    after = applied(before, update)

    (pass_,) = update["review_passes"]
    assert pass_.routed_to == expected
    assert route_target(after) == expected


# --- route_target() -------------------------------------------------------------------------


def test_route_target_reads_the_written_verdict_not_the_claim():
    """`reviewer_claim` says 'block' but the router already wrote 'exhausted' -- route_target must
    follow the field it owns, not re-derive the verdict from the claim."""
    s = state(reviewer_claim="block", review_verdict="exhausted", objections=[objection()])

    assert route_target(s) == "reporter"


def test_route_target_with_a_block_verdict_but_no_minted_pass_falls_back_to_the_reporter():
    """Not reachable from a real run -- the router always mints a pass alongside 'block' -- but a
    directly constructed state must not crash `max()` on an empty `review_passes`."""
    s = state(review_verdict="block", objections=[objection()])

    assert route_target(s) == "reporter"


def test_route_target_reads_the_latest_pass_by_iteration_not_list_order():
    older = ReviewPass(iteration=1, claim="block", routed_to="modeler", dispositions={})
    newer = ReviewPass(iteration=2, claim="block", routed_to="feature_eng", dispositions={})
    s = state(review_verdict="block", review_passes=[newer, older])

    assert route_target(s) == "feature_eng"


def test_route_target_on_pending_or_pass_always_goes_to_the_reporter():
    assert route_target(state(review_verdict="pending")) == "reporter"
    assert route_target(state(review_verdict="pass")) == "reporter"
