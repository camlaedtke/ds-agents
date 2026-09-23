"""What happens after a node says `recoverable=False`.

Until this landed, the answer was "nothing". `feature_eng` and `modeler` both refuse to fit on a
truncated split manifest, which is correct, and both mark the refusal unrecoverable, which is also
correct -- but nothing in `graph.py` or the router read `recoverable`, so the run carried on through
the reviewer and the reporter, spent a full run's tokens, and `publishable()` accepted the result.
Four of the thirteen benchmark datasets failed exactly that way and looked like measurements.

The fix is two things that must not be confused:

* `halt_or` stops the SPENDING. A fatal error routes straight to the reporter and no further node
  runs a model.
* `halted_at` stops the row from LYING. The row is still written -- deliberately, because
  `nodes/reporter.py` exists so the hardest datasets do not vanish from the results file -- and it
  names the node that refused.

The end-to-end test below reproduces the original defect exactly, by truncating the split artifact
on the way out of the store rather than by finding a 36k-row dataset to run.
"""

import json
from pathlib import Path

import pytest

from ds_agents.graph import run_pipeline
from ds_agents.nodes.router import halt_or
from ds_agents.state import NodeEvent, PipelineError, PipelineState, utc_now
from ds_agents.tools.llm import StubModel
from ds_agents.tools.local import LocalTools
from tests.conftest import _toy_state

TOY = Path(__file__).parent / "fixtures" / "toy" / "toy.csv"

fast = pytest.mark.fast


def _state(*errors: PipelineError) -> PipelineState:
    state = PipelineState(dataset_id="toy", task_description="t")
    state.errors = list(errors)
    return state


@fast
class TestTheStateHelpers:
    def test_a_clean_state_is_not_halted(self):
        state = _state()

        assert state.fatal_errors() == []
        assert state.halted() is False
        assert state.halted_at() is None

    def test_a_recoverable_error_is_not_a_halt(self):
        """The distinction `errored` cannot make. A column whose association could not be computed
        is an error; a dataset that cannot be run is a different thing entirely."""
        state = _state(PipelineError(node="profiler", message="no association", recoverable=True))

        assert state.halted() is False
        assert state.results_row()["errored"] is True
        assert state.results_row()["halted_at"] is None

    def test_the_first_fatal_error_names_the_halt(self):
        """`errors` uses an `operator.add` reducer, so the first fatal error is on the state for
        the rest of the run. A later node cannot un-halt it, which is correct by the definition of
        `recoverable=False` and is worth pinning rather than leaving to be inferred."""
        state = _state(
            PipelineError(node="profiler", message="warned", recoverable=True),
            PipelineError(node="feature_eng", message="truncated split", recoverable=False),
            PipelineError(node="modeler", message="no features", recoverable=False),
        )

        assert [e.node for e in state.fatal_errors()] == ["feature_eng", "modeler"]
        assert state.halted_at() == "feature_eng"

    def test_a_halted_run_is_still_publishable(self):
        """Deliberate, and the argument is `nodes/reporter.py`'s docstring: the harness needs a row
        for every dataset including the ones that blew up, or the hardest datasets vanish and every
        published table biases upward. Refusing the row here would leave the reason in stdout
        scrollback, which is the same shape as the defect `halted_at` was added to fix."""
        state = _state(PipelineError(node="feature_eng", message="x", recoverable=False))
        state.node_trace = [
            NodeEvent(node="feature_eng", started=utc_now(), model="claude-haiku-4-5-20251001")
        ]

        assert state.publishable() == (True, "")


@fast
class TestTheEdgeFunction:
    def test_it_goes_where_the_wiring_says_when_nothing_is_wrong(self):
        assert halt_or("modeler")(_state()) == "modeler"

    def test_it_goes_to_the_reporter_on_a_fatal_error(self):
        state = _state(PipelineError(node="feature_eng", message="x", recoverable=False))

        assert halt_or("modeler")(state) == "reporter"

    def test_a_recoverable_error_does_not_divert_it(self):
        state = _state(PipelineError(node="feature_eng", message="x", recoverable=True))

        assert halt_or("modeler")(state) == "modeler"


@fast
def test_no_committed_results_row_would_have_been_halted():
    """`halted_at` ships as null on every row already in the tree, so adding it rewrites nothing.

    The same shape of check `score_ratio`'s retirement got. If this ever fails, some committed row
    was written by a run that had already refused, and every number on it is suspect.
    """
    results = Path(__file__).resolve().parents[1] / "evals" / "results"
    rows = 0
    for path in sorted(results.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            fatal = [e for e in row.get("errors", []) if e.get("recoverable") is False]
            assert not fatal, f"{path.name}: a committed row carries {fatal}"
            assert row.get("halted_at") is None
    assert rows > 100, "expected the committed corpus, did it move?"


class _TruncatesTheSplit:
    """`LocalTools`, except every read of the split manifest comes back truncated.

    This is the 1 MiB collision, reproduced without a 36k-row dataset: `read_artifact` caps at
    `DEFAULT_READ_BYTES` and the old representation measured 2,690,410 B on `higgs`. Everything
    else is delegated, so the run is the real run.
    """

    def __init__(self, inner: LocalTools) -> None:
        self.inner = inner

    def read_artifact(self, artifact_id, max_bytes=None):
        payload = self.inner.read_artifact(artifact_id, max_bytes=max_bytes)
        if payload.meta.name == "split_manifest.json":
            return payload.model_copy(update={"content": payload.content[:64], "truncated": True})
        return payload

    def run_python(self, code, timeout_s=60):
        return self.inner.run_python(code, timeout_s=timeout_s)

    def write_artifact(self, name, content, kind="text", extra=None):
        return self.inner.write_artifact(name, content, kind=kind, extra=extra)

    def log_metric(self, run_id, name, value):
        return self.inner.log_metric(run_id, name, value)


def test_a_truncated_split_stops_the_run_at_feature_eng(tmp_path):
    """The whole defect, end to end. Not `fast`: it runs the real graph in real subprocesses.

    Before `halt_or`, this run reached `reporter` through `modeler`, `reviewer` and `router`, paid
    for all three, and produced a row indistinguishable at a glance from a measurement.
    """
    inner = LocalTools(tmp_path / "run", dataset_path=TOY, dataset_id="toy")
    state = run_pipeline(_toy_state(), tools=_TruncatesTheSplit(inner), model=StubModel())

    assert [event.node for event in state.node_trace] == [
        "intake",
        "profiler",
        "feature_eng",
        "reporter",
    ]
    assert state.halted_at() == "feature_eng"
    assert "truncated" in state.fatal_errors()[0].message
    assert state.chosen_model is None
    # The reporter still ran and still produced something a person can read. That is the reason the
    # halt goes here rather than to END.
    assert state.report_artifact is not None
    # Never adjudicated is not the same as adjudicated and cleared: the router never ran, so the
    # verdict stays at its default rather than reading `pass` on a run that produced no model.
    assert state.review_verdict == "pending"
