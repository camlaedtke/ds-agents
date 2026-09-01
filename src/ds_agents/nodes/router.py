"""router: turn the reviewer's claim into the verdict, and mint the ReviewPass that decides where
the run goes next.

Reads `reviewer_claim`, `reviewer_dispositions`, `review_iterations`, `config.loop_cap`,
`config.reviewer_enabled`, `open_objections()`. Writes `review_verdict` on every path,
`review_iterations` and `review_passes` only when a reviewer pass actually completed, `node_trace`,
and `errors` conditionally. Never touches `reviewer_dispositions` -- that field is the reviewer's
handoff to write and the router's to read, once, while minting the pass below.
Tools: none. Model: none -- `model` is accepted only for signature uniformity with every other
node and is never called.

The router exists because the reviewer is a model under test and cannot be trusted to enforce its
own cap, certify its own outcome, or decide where its own claim sends the run. Four things here are
load-bearing:

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
3. **The router mints the whole `ReviewPass`, including `routed_to`.** `route_target` (the
   conditional edge) does not re-derive a destination from `open_objections()` -- it just reads the
   `routed_to` the router already decided, off the latest pass. The router and the edge function
   run against different states (the router sees the state as of its own invocation; the edge sees
   the state after that update has been merged), so a second, independent derivation of "where does
   a block go" could disagree with the first. Single authority avoids that.
4. **`routed_to` is computed against the PROJECTED post-pass open set**, not the raw
   `open_objections()`: this pass's `reviewer_dispositions` are applied on top of the objections
   that were open going in. Using the raw pre-pass set would send a run back to `feature_eng` for
   an objection this very pass just resolved.

A fifth, smaller thing: when there is no claim at all (`reviewer_claim is None`), the verdict is
`pending`, not `pass`, and no `ReviewPass` is minted. This keeps the reviewer-disabled arm
distinguishable from a run where the reviewer was supposed to answer and silently did not --
"never adjudicated" and "adjudicated and cleared" must never collapse into the same verdict, or a
crashed reviewer scores as a clean pass.
"""

from collections.abc import Callable
from typing import Any, Literal

from ds_agents.nodes._run import NodeRun
from ds_agents.state import (
    Disposition,
    NodeName,
    PipelineError,
    PipelineState,
    ReviewPass,
)
from ds_agents.tools.llm import StructuredModel
from ds_agents.tools.protocol import Tools

RoutedTo = Literal["feature_eng", "modeler", "reporter"]


def _route_for_block(state: PipelineState, dispositions: dict[str, Disposition]) -> RoutedTo:
    """Where a 'block' claim sends the run, against the objections still open once THIS pass's
    dispositions are applied -- not the objections open going in. Upstream first: re-fitting on a
    matrix that still holds an objected column, then dropping it, invalidates the fit that was
    just made.

    Asks `would_be_open(target_node=destination)` rather than comparing `Objection.target_node`
    here, so this function and `feature_eng._forced_drops` inherit the `objection_routing`
    condition from the one place that applies it (`PipelineState.effective_target`). A
    `target_node ==` check in this file would be a second, independently maintained answer to "who
    acts on this" -- the same dual-authority split that invariant 3 above exists to close -- and
    under `by_category` the two answers would disagree: the router would send the run to `modeler`
    while `feature_eng` was the only node able to act.

    `would_be_open` is shared with the reviewer, which asks the same question one node earlier to
    decide whether its own `block` is a dead end. Two derivations of "is this actionable" is
    exactly how a `block` with nothing to act on used to reach this function at all.
    """
    upstream_first: tuple[RoutedTo, ...] = ("feature_eng", "modeler")
    for destination in upstream_first:
        if state.would_be_open(dispositions=dispositions, target_node=destination):
            return destination
    return "reporter"


def router(state: PipelineState, *, tools: Tools, model: StructuredModel) -> dict[str, Any]:
    run = NodeRun("router")
    claim = state.reviewer_claim
    update: dict[str, Any] = {}
    errors: list[PipelineError] = []

    if claim is None:
        # No completed pass, so nothing to count and no ReviewPass to mint. `pending` is the
        # honest verdict for the reviewer-off arm: never adjudicated is not the same as
        # adjudicated and cleared.
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
        dispositions = dict(state.reviewer_dispositions)

        if claim == "pass":
            verdict: str = "pass"
            routed_to: RoutedTo = "reporter"
        elif iterations >= state.config.loop_cap:
            verdict = "exhausted"
            routed_to = "reporter"
        else:
            verdict = "block"
            routed_to = _route_for_block(state, dispositions)
            if routed_to == "reporter":
                errors.append(
                    PipelineError(
                        node="router",
                        message="reviewer claimed 'block' with no open objection; there is "
                        "nothing for feature_eng or modeler to act on, routing to reporter",
                    )
                )

        update["review_verdict"] = verdict
        update["review_passes"] = [
            ReviewPass(
                iteration=iterations,
                claim=claim,
                routed_to=routed_to,
                dispositions=dispositions,
                # Pre-increment clock: the reviewer stamped `raised_at_iteration` from
                # `state.review_iterations` before THIS call incremented it, so a new objection
                # from this pass matches the counter's value on entry, not `iterations` above.
                new_objection_ids=[
                    o.id
                    for o in state.objections
                    if o.raised_at_iteration == state.review_iterations
                ],
            )
        ]

    if errors:
        update["errors"] = errors
    # Minted here, not at the top, so the event covers the work rather than reading a flat 0.00s.
    # Every other node mints on return; the router doing otherwise would make it the one row in
    # the cost table that means something different.
    update["node_trace"] = [run.event()]
    return update


def route_target(state: PipelineState) -> RoutedTo:
    """Where the graph goes after the router.

    Reads `review_verdict` first, which the router has already written, rather than re-deriving it
    from `reviewer_claim` -- the router is the sole authority for `review_verdict`. For a `block`
    verdict, reads `routed_to` off the latest minted `ReviewPass` (max by iteration) rather than
    recomputing a destination from `open_objections()` here: the router already decided this,
    against a state this edge function does not see the same way, so a second derivation could
    disagree with the first. `review_passes` empty with a `block` verdict is not reachable from a
    normal run -- the router always mints a pass alongside a `block` verdict -- but falls back to
    `reporter` rather than crashing on a directly constructed state.
    """
    if state.review_verdict != "block":
        return "reporter"
    if not state.review_passes:
        return "reporter"
    latest = max(state.review_passes, key=lambda rp: rp.iteration)
    return latest.routed_to


def halt_or(destination: NodeName) -> Callable[[PipelineState], NodeName]:
    """What a straight-line edge in `graph.py` does: go to `destination`, or to the reporter.

    A node that returns `recoverable=False` has said nothing downstream can produce a trustworthy
    result. Until this existed, nothing in the graph or the router read `recoverable` at all, so
    the run carried on through every remaining node, spent a full run's tokens and wrote a row that
    looked like a measurement. Four of the thirteen benchmark datasets failed exactly that way.

    To `reporter` and not `END`, because `nodes/reporter.py` is written for this path -- its
    docstring says the harness needs a row for every dataset including the ones that blew up, or
    the hardest datasets vanish and every published table biases upward. The reporter calls no
    model, so halting is cheap.

    The router's OWN conditional edges are deliberately not wrapped. The router is unreachable once
    a fatal error is on the state, since every edge that could reach it halts first, and a `block`
    verdict minted with no `ReviewPass` is the state `route_target` calls unreachable.

    It lives beside `route_target` rather than in `graph.py` for the reason `graph.py`'s docstring
    gives: that file is wiring, and this is a decision about where a run goes.
    """

    def edge(state: PipelineState) -> NodeName:
        return "reporter" if state.halted() else destination

    return edge
