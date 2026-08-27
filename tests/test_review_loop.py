"""The review loop, live end to end on the toy dataset through `LocalTools`.

Not marked `fast`: like `test_toy_pipeline.py`, it launches subprocesses that import pandas and
scikit-learn, and it runs the graph's cycle edges for real rather than asserting on a single node's
narrow dict. `tests/nodes/test_reviewer.py` and `tests/nodes/test_router.py` cover the reviewer's
and the router's own behaviour in isolation; this file covers the thing neither of those can: that
the two actually compose into a working loop through LangGraph, with `feature_eng` and `modeler`
re-running for real.

`QueuedModel` scripts only the `ReviewFinding` schema pass by pass; every other schema (intake,
profiler, feature_eng, modeler) still gets the ordinary `StubModel` answer, which is what keeps
these tests from having to re-derive the whole toy run's behaviour.
"""

from pathlib import Path

from conftest import QueuedModel

from ds_agents.cli import _toy_state
from ds_agents.graph import run_pipeline
from ds_agents.nodes.reviewer import DispositionUpdate, ProposedObjection, ReviewFinding
from ds_agents.state import PipelineState
from ds_agents.tools.llm import StubModel
from ds_agents.tools.local import LocalTools

TOY = Path(__file__).parent / "fixtures" / "toy" / "toy.csv"


def _block_account_status_code(payload: dict) -> ReviewFinding:
    """Raise one leakage objection against the planted column, targeting feature_eng."""
    return ReviewFinding(
        claim="block",
        objections=[
            ProposedObjection(
                category="leakage",
                subcategory="planted status code",
                target_node="feature_eng",
                columns=["account_status_code"],
                evidence="account_status_code tracks the target almost exactly",
                severity="high",
            )
        ],
        dispositions=[],
        summary="account_status_code looks leaky; feature_eng should drop it",
    )


def _pass_and_resolve(payload: dict) -> ReviewFinding:
    """Resolve every objection this pass was shown as open, then pass. Reads the ids off the
    prompt's own facts rather than hard-coding one, because the node mints a fresh id per
    objection and this callable has no other way to know it."""
    open_ids = [o["id"] for o in payload.get("open_objections", [])]
    return ReviewFinding(
        claim="pass",
        objections=[],
        dispositions=[
            DispositionUpdate(objection_id=oid, disposition="resolved") for oid in open_ids
        ],
        summary="account_status_code is gone; nothing left open",
    )


def _with_loop_cap(state: PipelineState, loop_cap: int) -> PipelineState:
    return state.model_copy(
        update={"config": state.config.model_copy(update={"loop_cap": loop_cap})}
    )


def test_block_once_then_pass(tmp_path):
    """The whole point of the loop: a reviewer that blocks on real evidence, feature_eng acting on
    the objection it's handed, and the reviewer clearing its own objection once the fix lands."""
    tools = LocalTools(tmp_path / "loop-block-pass", dataset_path=TOY, dataset_id="toy")
    model = QueuedModel(ReviewFinding, [_block_account_status_code, _pass_and_resolve])

    state = run_pipeline(_toy_state(), tools=tools, model=model)

    trace_nodes = [e.node for e in state.node_trace]
    assert trace_nodes.count("feature_eng") == 2, "the cycle must actually re-run feature_eng"
    assert trace_nodes.count("reviewer") == 2
    assert state.review_iterations == 2
    assert state.review_verdict == "pass"
    assert "account_status_code" not in (state.final_features or [])

    passes = sorted(state.review_passes, key=lambda rp: rp.iteration)
    assert [rp.routed_to for rp in passes] == ["feature_eng", "reporter"]
    assert len(state.objections) == 1
    assert state.open_objections() == []


def test_always_block_ends_exhausted_at_the_cap(tmp_path):
    """A reviewer that never resolves its own objection must not loop forever: `loop_cap=2` forces
    the second pass to `exhausted` even though the claim is still `block`."""
    tools = LocalTools(tmp_path / "loop-exhausted", dataset_path=TOY, dataset_id="toy")
    model = QueuedModel(ReviewFinding, [_block_account_status_code, _block_account_status_code])
    state_in = _with_loop_cap(_toy_state(), loop_cap=2)

    state = run_pipeline(state_in, tools=tools, model=model)

    assert state.review_verdict == "exhausted"
    assert state.review_iterations == 2


def test_the_stub_reviewer_leaves_the_run_clean(tmp_path):
    """`StubModel` now answers `ReviewFinding` too (claim `pass`, nothing raised). With the
    reviewer flipped on in `_toy_state`, the offline run must still finish with no errors -- the
    reviewer being wired in must not make every stub run look broken."""
    tools = LocalTools(tmp_path / "loop-stub", dataset_path=TOY, dataset_id="toy")

    state = run_pipeline(_toy_state(), tools=tools, model=StubModel())

    assert state.errors == []
    assert state.review_verdict == "pass"
