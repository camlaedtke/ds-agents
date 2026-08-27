"""Do the planted traps actually reach the reviewer?

This is the test the two new fixtures exist for. `StubModel` nominates no leakage candidates and
proposes no drops by design, so a run under it is the case where every upstream node declines to
act -- exactly the situation the reviewer is supposed to be the last line against. If a trap is
absent from `final_features` or from `top_importances` after such a run, the reviewer is being
shown a clean matrix and cannot be graded on that fixture at all. That is how the toy fixture
turned out to be unable to demonstrate PLAN.md's "reviewer catches the leakage" box, and it failed
silently: nothing errored, the leakage numbers just came out looking like a reviewer miss.

`final_features` and `top_importances` are named specifically because they are two of the seven
things `reviewer._user_message` puts in front of the model. A column outside both is invisible.

Not marked `fast`: each case runs the whole graph.
"""

from pathlib import Path

import pytest

from ds_agents.cli import _fixture_state
from ds_agents.fixtures import load_fixture
from ds_agents.graph import run_pipeline
from ds_agents.state import PipelineState
from ds_agents.tools.llm import StubModel
from ds_agents.tools.local import LocalTools

TRAP_FIXTURES = ["claims_timing", "reissued_ids"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, PipelineState]:
    """One stub run per trap fixture, shared across the assertions below."""
    root = tmp_path_factory.mktemp("trap_survival")
    out: dict[str, PipelineState] = {}
    for name in TRAP_FIXTURES:
        fixture = load_fixture(name)
        tools = LocalTools(
            Path(root / name), dataset_path=fixture.csv_path, dataset_id=fixture.dataset_id
        )
        try:
            out[name] = run_pipeline(_fixture_state(fixture), tools=tools, model=StubModel())
        finally:
            tools.close()
    return out


@pytest.mark.parametrize("name", TRAP_FIXTURES)
def test_the_run_completes_cleanly(runs, name):
    state = runs[name]
    assert state.errors == [], [f"{e.node}: {e.message}" for e in state.errors]
    assert [e.node for e in state.node_trace] == [
        "intake",
        "profiler",
        "feature_eng",
        "modeler",
        "reviewer",
        "router",
        "reporter",
    ]


@pytest.mark.parametrize("name", TRAP_FIXTURES)
def test_the_target_was_inferred_from_the_prose_description(runs, name):
    """intake gets a sentence, not a spec. If it names the wrong column the rest is meaningless."""
    state = runs[name]
    assert state.spec is not None
    assert state.spec.target == load_fixture(name).manifest.target


@pytest.mark.parametrize("name", TRAP_FIXTURES)
def test_every_planted_column_is_in_the_matrix_the_reviewer_is_shown(runs, name):
    state = runs[name]
    planted = load_fixture(name).manifest.planted_columns
    kept = set(state.final_features or [])
    assert set(planted) <= kept, (
        f"{name}: {sorted(set(planted) - kept)} never reached the matrix, so the reviewer "
        f"cannot be graded on this fixture"
    )


@pytest.mark.parametrize("name", TRAP_FIXTURES)
def test_every_planted_column_reaches_the_reviewer_as_an_importance(runs, name):
    """Being in the matrix is not enough. `top_importances` is the reviewer's only quantitative
    handle on whether the model actually leaned on a column, and it is truncated to 15 entries."""
    state = runs[name]
    planted = load_fixture(name).manifest.planted_columns
    ranked = [column for column, _ in state.top_importances]
    assert set(planted) <= set(ranked), f"{name}: missing from top_importances: {ranked}"


@pytest.mark.parametrize("name", TRAP_FIXTURES)
def test_the_model_actually_leans_on_each_trap(runs, name):
    """A trap the model ignores is not a leak, it is a decoy.

    Permutation importance is measured on the agents' holdout, so a non-trivial value means the
    score would genuinely move if the column were removed -- which is what makes the run's claimed
    score an overstatement and gives the reviewer something real to object to.
    """
    state = runs[name]
    importance = dict(state.top_importances)
    for column in load_fixture(name).manifest.planted_columns:
        assert importance[column] > 0.02, f"{name}.{column} scores {importance[column]}"


@pytest.mark.parametrize("name", TRAP_FIXTURES)
def test_the_declared_identifier_is_still_force_dropped(runs, name):
    """The control. Each new fixture ships a genuine unique identifier alongside its trap; if the
    id rule stopped firing, a trap surviving would say nothing about the trap's design."""
    state = runs[name]
    for column in load_fixture(name).manifest.id_columns:
        assert column in state.dropped_features
        assert column not in (state.final_features or [])


@pytest.mark.parametrize("name", TRAP_FIXTURES)
def test_the_stub_run_is_refused_publication(runs, name):
    publishable, reason = runs[name].publishable()
    assert publishable is False
    assert "stub" in reason


def test_reissued_ids_keeps_the_leaky_id_and_drops_the_honest_one(runs):
    """The whole fixture in one assertion.

    Two id-shaped columns, opposite fates. The unique `application_ref` is force-dropped, and
    `member_number` -- the one that actually encodes the outcome -- survives and is leaned on.
    This is the case the profiler's prompt argues the model into getting backwards, by telling it
    that a near-unique column "predicts nothing".
    """
    state = runs["reissued_ids"]
    assert "application_ref" in state.dropped_features
    assert "member_number" in (state.final_features or [])
    assert dict(state.top_importances)["member_number"] > 0.02


def test_claims_timing_traps_outrank_the_legitimate_signal(runs):
    """Both after-the-fact columns beat every honest feature the fixture provides.

    That ordering is the fixture working as intended: the run's claimed score rests on two columns
    that will not exist at predict time, which is a finding the reviewer has evidence for and the
    profiler's mutual-information numbers alone do not make obvious.
    """
    state = runs["claims_timing"]
    ranked = [column for column, _ in state.top_importances]
    planted = set(load_fixture("claims_timing").manifest.planted_columns)
    assert set(ranked[:2]) == planted, ranked
