"""router writes `review_verdict` (always) and `review_iterations` (only on a completed pass).

The tests that matter here are about the two off-by-one-shaped rules in router.py's docstring:
the counter increments BEFORE the cap is checked, and a claim of `block` that lands exactly at
the cap becomes `exhausted`, never `pass` -- because a reviewer that gives up at the cap must not
read the same as one that passed on merit in a results table.
"""

from typing import Literal

import pytest
from conftest import FakeTools, ScriptedModel

from ds_agents.nodes.router import route_target, router
from ds_agents.state import (
    Objection,
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
) -> PipelineState:
    return PipelineState(
        dataset_id="toy",
        task_description="Predict churned, report roc_auc.",
        config=RunConfig(loop_cap=loop_cap, reviewer_enabled=reviewer_enabled),
        reviewer_claim=reviewer_claim,
        review_iterations=review_iterations,
        objections=objections or [],
        review_passes=review_passes or [],
        review_verdict=review_verdict,
    )


def objection(
    target_node: RoutableNode = "feature_eng", column: str = "account_status_code"
) -> Objection:
    return Objection(
        category="leakage",
        subcategory="planted status code",
        target_node=target_node,
        columns=[column],
        evidence="normalized mutual information with the target is 0.518",
        severity="high",
        raised_at_iteration=0,
    )


def tools() -> FakeTools:
    return FakeTools()


def model() -> ScriptedModel:
    return ScriptedModel({})


# --- router() -----------------------------------------------------------------------------


def test_no_claim_does_not_increment_and_leaves_pending():
    update = router(
        state(reviewer_claim=None, reviewer_enabled=False), tools=tools(), model=model()
    )

    assert set(update) == {"review_verdict", "node_trace"}
    assert update["review_verdict"] == "pending"
    assert "review_iterations" not in update


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


def test_a_pass_increments_and_routes_to_the_reporter():
    update = router(
        state(reviewer_claim="pass", review_iterations=0, loop_cap=3), tools=tools(), model=model()
    )

    assert update["review_iterations"] == 1
    assert update["review_verdict"] == "pass"
    assert "errors" not in update
    assert route_target(state(review_verdict="pass")) == "reporter"


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


def test_a_block_at_the_cap_becomes_exhausted():
    update = router(
        state(reviewer_claim="block", review_iterations=2, loop_cap=3), tools=tools(), model=model()
    )

    assert update["review_iterations"] == 3
    assert update["review_verdict"] == "exhausted"
    # No open-objection check on this branch: a reviewer that hits the cap is exhausted
    # regardless of whether it left anything actionable behind.
    assert "errors" not in update


def test_the_router_never_calls_a_model():
    m = ScriptedModel({})

    router(state(reviewer_claim="block", objections=[objection()]), tools=tools(), model=m)

    assert m.calls == []


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
    after_round2 = state(
        review_verdict=round2["review_verdict"],
        review_iterations=round2["review_iterations"],
        objections=[objection()],
    )
    assert route_target(after_round2) == "reporter"
    assert round2["review_iterations"] <= 2


def test_a_block_with_no_open_objection_is_an_error_and_goes_to_the_reporter():
    update = router(
        state(reviewer_claim="block", review_iterations=0, loop_cap=3, objections=[]),
        tools=tools(),
        model=model(),
    )

    assert update["review_verdict"] == "block"
    (error,) = update["errors"]
    assert "no open objection" in error.message

    after = state(review_verdict=update["review_verdict"], objections=[])
    assert route_target(after) == "reporter"


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


# --- route_target() -------------------------------------------------------------------------


def test_route_target_reads_the_written_verdict_not_the_claim():
    """`reviewer_claim` says 'block' but the router already wrote 'exhausted' -- route_target must
    follow the field it owns, not re-derive the verdict from the claim."""
    s = state(reviewer_claim="block", review_verdict="exhausted", objections=[objection()])

    assert route_target(s) == "reporter"


def test_an_open_feature_eng_objection_routes_upstream_of_modeler():
    s = state(
        review_verdict="block",
        objections=[objection(target_node="modeler"), objection(target_node="feature_eng")],
    )

    assert route_target(s) == "feature_eng"


def test_an_open_modeler_only_objection_routes_to_modeler():
    s = state(review_verdict="block", objections=[objection(target_node="modeler")])

    assert route_target(s) == "modeler"


def test_a_resolved_objection_does_not_route_back():
    """Guards that `route_target` calls `open_objections()`, not the raw `objections` list."""
    ob = objection()
    s = state(
        review_verdict="block",
        objections=[ob],
        review_passes=[
            ReviewPass(
                iteration=0, claim="pass", routed_to="reporter", dispositions={ob.id: "resolved"}
            )
        ],
    )

    assert route_target(s) == "reporter"


def test_route_target_on_pending_or_pass_always_goes_to_the_reporter():
    assert route_target(state(review_verdict="pending")) == "reporter"
    assert route_target(state(review_verdict="pass")) == "reporter"
