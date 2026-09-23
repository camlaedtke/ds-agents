"""Under a StubModel run (no upstream node acts), every planted trap must still reach the reviewer
in `final_features` and `top_importances`, in both naming arms. Not marked `fast`: runs the whole
graph."""

from pathlib import Path
from typing import NamedTuple

import pytest

from ds_agents.cli import _run_state
from ds_agents.fixtures import load_fixture
from ds_agents.graph import run_pipeline
from ds_agents.naming import NAMINGS, materialize
from ds_agents.naming import apply as apply_rename
from ds_agents.runnable import Runnable
from ds_agents.state import PipelineState
from ds_agents.tools.llm import StubModel
from ds_agents.tools.local import LocalTools

TRAP_FIXTURES = ["claims_timing", "reissued_ids"]
CASES = [(name, naming) for name in TRAP_FIXTURES for naming in NAMINGS]


class Run(NamedTuple):
    """A finished run plus the map that produced the names in it.

    The map is carried alongside because every assertion below starts from a manifest column name,
    and under `opaque` that name is not what is in the state. Resolving it through `columns()`
    rather than hardcoding is what lets one assertion grade both arms.
    """

    state: PipelineState
    rename: dict[str, str]

    def columns(self, names: list[str]) -> list[str]:
        return apply_rename(names, self.rename)


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[tuple[str, str], Run]:
    """One stub run per trap fixture per naming arm, shared across the assertions below."""
    root = tmp_path_factory.mktemp("trap_survival")
    out: dict[tuple[str, str], Run] = {}
    for name, naming in CASES:
        fixture = load_fixture(name)
        case = Path(root / f"{name}-{naming}")
        runnable_dataset = Runnable.from_fixture(fixture)
        dataset_path, rename = materialize(runnable_dataset, naming, case / "input")
        tools = LocalTools(case / "run", dataset_path=dataset_path, dataset_id=fixture.dataset_id)
        try:
            state = run_pipeline(
                _run_state(runnable_dataset, naming=naming),
                tools=tools,
                model=StubModel(),
            )
        finally:
            tools.close()
        out[(name, naming)] = Run(state, rename)
    return out


@pytest.mark.parametrize(("name", "naming"), CASES)
def test_the_run_completes_cleanly(runs, name, naming):
    state = runs[(name, naming)].state
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


@pytest.mark.parametrize(("name", "naming"), CASES)
def test_the_target_was_inferred_from_the_prose_description(runs, name, naming):
    """intake gets a sentence, not a spec. If it names the wrong column the rest is meaningless.

    The target is never renamed, so this assertion is identical in both arms -- which is the point:
    the opaque arm must not accidentally make intake's job harder as well as the profiler's, or the
    two conditions would differ in two ways instead of one.
    """
    state = runs[(name, naming)].state
    assert state.spec is not None
    assert state.spec.target == load_fixture(name).manifest.target


@pytest.mark.parametrize(("name", "naming"), CASES)
def test_the_run_records_which_arm_it_was(runs, name, naming):
    """Without this the two arms produce indistinguishable results rows."""
    assert runs[(name, naming)].state.config.naming == naming


@pytest.mark.parametrize(("name", "naming"), CASES)
def test_every_planted_column_is_in_the_matrix_the_reviewer_is_shown(runs, name, naming):
    run = runs[(name, naming)]
    planted = run.columns(load_fixture(name).manifest.planted_columns)
    kept = set(run.state.final_features or [])
    assert set(planted) <= kept, (
        f"{name}/{naming}: {sorted(set(planted) - kept)} never reached the matrix, so the reviewer "
        f"cannot be graded on this fixture"
    )


@pytest.mark.parametrize(("name", "naming"), CASES)
def test_every_planted_column_reaches_the_reviewer_as_an_importance(runs, name, naming):
    """Being in the matrix is not enough. `top_importances` is the reviewer's only quantitative
    handle on whether the model actually leaned on a column, and it is truncated to 15 entries."""
    run = runs[(name, naming)]
    planted = run.columns(load_fixture(name).manifest.planted_columns)
    ranked = [column for column, _ in run.state.top_importances]
    assert set(planted) <= set(ranked), f"{name}/{naming}: missing from top_importances: {ranked}"


@pytest.mark.parametrize(("name", "naming"), CASES)
def test_the_model_actually_leans_on_each_trap(runs, name, naming):
    """A trap the model ignores is not a leak, it is a decoy.

    Permutation importance is measured on the agents' holdout, so a non-trivial value means the
    score would genuinely move if the column were removed -- which is what makes the run's claimed
    score an overstatement and gives the reviewer something real to object to.
    """
    run = runs[(name, naming)]
    importance = dict(run.state.top_importances)
    for column in run.columns(load_fixture(name).manifest.planted_columns):
        assert importance[column] > 0.02, f"{name}/{naming}.{column} scores {importance[column]}"


@pytest.mark.parametrize(("name", "naming"), CASES)
def test_the_declared_identifier_is_still_force_dropped(runs, name, naming):
    """The control. Each new fixture ships a genuine unique identifier alongside its trap; if the
    id rule stopped firing, a trap surviving would say nothing about the trap's design.

    Doubly load-bearing in the opaque arm: `feature_eng` drops identifiers on distinctness and not
    on the name, and this is the assertion that says so. If the rule had any name heuristic in it,
    `var_01` would survive where `claim_ref` did not and the arms would differ in what they drop
    as well as in what they are called.
    """
    run = runs[(name, naming)]
    for column in run.columns(load_fixture(name).manifest.id_columns):
        assert column in run.state.dropped_features
        assert column not in (run.state.final_features or [])


@pytest.mark.parametrize(("name", "naming"), CASES)
def test_the_stub_run_is_refused_publication(runs, name, naming):
    publishable, reason = runs[(name, naming)].state.publishable()
    assert publishable is False
    assert "stub" in reason


@pytest.mark.parametrize("naming", NAMINGS)
def test_reissued_ids_keeps_the_leaky_id_and_drops_the_honest_one(runs, naming):
    """The whole fixture in one assertion.

    Two id-shaped columns, opposite fates. The unique `application_ref` is force-dropped, and
    `member_number` -- the one that actually encodes the outcome -- survives and is leaned on.
    This is the case the profiler's prompt argues the model into getting backwards, by telling it
    that a near-unique column "predicts nothing".
    """
    run = runs[("reissued_ids", naming)]
    honest, leaky = run.columns(["application_ref", "member_number"])
    assert honest in run.state.dropped_features
    assert leaky in (run.state.final_features or [])
    assert dict(run.state.top_importances)[leaky] > 0.02


@pytest.mark.parametrize("naming", NAMINGS)
def test_claims_timing_traps_outrank_the_legitimate_signal(runs, naming):
    """Both after-the-fact columns beat every honest feature the fixture provides.

    That ordering is the fixture working as intended: the run's claimed score rests on two columns
    that will not exist at predict time, which is a finding the reviewer has evidence for and the
    profiler's mutual-information numbers alone do not make obvious.

    It holds in both arms because permutation importance is computed from the values, and the
    values are identical -- which is the single fact the entire ablation rests on.
    """
    run = runs[("claims_timing", naming)]
    ranked = [column for column, _ in run.state.top_importances]
    planted = set(run.columns(load_fixture("claims_timing").manifest.planted_columns))
    assert set(ranked[:2]) == planted, ranked


def test_the_two_arms_produce_the_same_importances(runs):
    """The ablation's premise, asserted rather than assumed.

    Identical rows under different headers must produce identical numbers. If the ranking or the
    magnitudes moved between arms, something other than the names changed -- a re-formatted float,
    a shifted column, a different split -- and any difference in profiler behaviour between the
    arms would have a second candidate explanation.
    """
    for name in TRAP_FIXTURES:
        descriptive, opaque = runs[(name, "descriptive")], runs[(name, "opaque")]
        translated = [
            (opaque.rename.get(column, column), value)
            for column, value in descriptive.state.top_importances
        ]
        assert translated == pytest.approx(opaque.state.top_importances, abs=0.0), (
            f"{name}: the arms disagree on importances, so they differ in more than the names"
        )
