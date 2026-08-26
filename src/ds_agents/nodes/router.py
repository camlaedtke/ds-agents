"""router: turn the reviewer's claim into the verdict and the routing decision.

Reads `reviewer_claim`, `review_iterations`, `config.loop_cap`, `config.reviewer_enabled`,
`open_objections()`. Writes `review_verdict` on every path, `review_iterations` only when a
reviewer pass actually completed, `node_trace`, and `errors` conditionally.
Tools: none. Model: none -- `model` is accepted only for signature uniformity with every other
node and is never called.

The router exists because the reviewer is a model under test and cannot be trusted to enforce its
own cap or certify its own outcome. Two things here are load-bearing:

1. **The counter increments BEFORE the cap is checked.** `loop_cap=3` therefore permits at most
   three completed reviewer invocations and two returns upstream (to `feature_eng` or `modeler`):
   pass 1 and pass 2 can still come back as `block`, pass 3 is compared against the cap only after
   being counted and is forced to `exhausted` if it still wants to block. A reviewer that raised
   its own counter, or a router that checked the cap before incrementing, could buy the model one
   extra loop it was not entitled to.
2. **`block` at the cap becomes `exhausted`, never `pass`.** Otherwise a weak reviewer that simply
   gives up and starts emitting "pass" once it hits the cap is indistinguishable, in a results
   table, from a reviewer that actually cleared every objection on merit. `exhausted` is a
   published eval outcome (see docs/DECISIONS.md), not an implementation detail.

A third, smaller thing: when there is no claim at all (`reviewer_claim is None`), the verdict is
`pending`, not `pass`. This keeps the reviewer-disabled arm distinguishable from a run where the
reviewer was supposed to answer and silently did not -- "never adjudicated" and "adjudicated and
cleared" must never collapse into the same verdict, or a crashed reviewer scores as a clean pass.
"""

from typing import Any, Literal

from ds_agents.nodes._run import NodeRun
from ds_agents.state import PipelineError, PipelineState
from ds_agents.tools.llm import StructuredModel
from ds_agents.tools.protocol import Tools


def router(state: PipelineState, *, tools: Tools, model: StructuredModel) -> dict[str, Any]:
    run = NodeRun("router")
    claim = state.reviewer_claim
    update: dict[str, Any] = {}
    errors: list[PipelineError] = []

    if claim is None:
        # No completed pass, so nothing to count. `pending` is the honest verdict for the
        # reviewer-off arm: never adjudicated is not the same as adjudicated and cleared.
        update["review_verdict"] = "pending"
        if state.config.reviewer_enabled and state.config.loop_cap > 0:
            errors.append(
                PipelineError(
                    node="router",
                    message="reviewer is enabled but wrote no claim; recording 'pending' rather "
                    "than letting a missing reviewer read as a pass",
                )
            )
    else:
        iterations = state.review_iterations + 1
        update["review_iterations"] = iterations
        if claim == "pass":
            update["review_verdict"] = "pass"
        elif iterations >= state.config.loop_cap:
            update["review_verdict"] = "exhausted"
        else:
            update["review_verdict"] = "block"
            if not state.open_objections():
                errors.append(
                    PipelineError(
                        node="router",
                        message="reviewer claimed 'block' with no open objection; there is "
                        "nothing for feature_eng or modeler to act on, routing to reporter",
                    )
                )

    if errors:
        update["errors"] = errors
    # Minted here, not at the top, so the event covers the work rather than reading a flat 0.00s.
    # Every other node mints on return; the router doing otherwise would make it the one row in
    # the cost table that means something different.
    update["node_trace"] = [run.event()]
    return update


def route_target(state: PipelineState) -> Literal["feature_eng", "modeler", "reporter"]:
    """Where the graph goes after the router.

    Reads `review_verdict`, which the router has already written, rather than re-deriving it from
    `reviewer_claim` -- an edge function that recomputed the verdict could disagree with the node
    that owns it, and the router is the sole authority for `review_verdict`.
    """
    if state.review_verdict != "block":
        return "reporter"
    open_now = state.open_objections()
    if any(o.target_node == "feature_eng" for o in open_now):
        # Upstream first. Re-fitting on a matrix that still holds the objected column, then
        # dropping it, invalidates the fit that was just made.
        return "feature_eng"
    if any(o.target_node == "modeler" for o in open_now):
        return "modeler"
    return "reporter"
