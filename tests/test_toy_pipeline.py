"""The toy pipeline, end to end, through the real tool shim.

Not marked `fast`: it launches subprocesses that import pandas and scikit-learn. The node tests
cover behaviour; this one covers wiring, and the thing it really checks is the split manifest,
because every later contamination objection is scored against a partition that has to be a
partition.

It runs with `StubModel`, so it asserts nothing about leakage detection. A stub finds nothing by
design and a test that expected otherwise would be asserting on a placeholder.
"""

import json
from pathlib import Path

import pytest

from ds_agents.cli import _toy_state
from ds_agents.graph import run_pipeline
from ds_agents.state import PipelineState
from ds_agents.tools.llm import StubModel
from ds_agents.tools.local import LocalTools

TOY = Path(__file__).parent / "fixtures" / "toy" / "toy.csv"


def run(root: Path) -> tuple[PipelineState, LocalTools]:
    tools = LocalTools(root, dataset_path=TOY, dataset_id="toy")
    return run_pipeline(_toy_state(), tools=tools, model=StubModel()), tools


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory) -> tuple[PipelineState, LocalTools]:
    """One run shared by the assertions below. Every snippet launch pays a fresh pandas and
    scikit-learn import, so a run per test turns a 2-second check into a 12-second one."""
    return run(tmp_path_factory.mktemp("toy_run"))


def test_the_toy_run_finishes_clean(toy_run):
    state, _tools = toy_run

    assert [event.node for event in state.node_trace] == ["intake", "profiler"]
    assert state.errors == []
    assert state.wall_seconds is not None


def test_intake_names_the_target_without_being_told_the_spec(toy_run):
    state, _tools = toy_run

    assert state.spec is not None
    assert state.spec.target == "churned"
    assert state.spec.task_type == "binary"


def test_the_profile_covers_every_column(toy_run):
    state, _tools = toy_run

    assert state.profile is not None
    assert state.profile.n_rows == 200
    assert len(state.profile.columns) == state.profile.n_columns == 8
    assert state.profile.target_balance == {"0": 0.745, "1": 0.255}


def test_the_split_manifest_is_an_actual_partition(toy_run):
    """Overlapping train and holdout ids would make every contamination objection unfalsifiable
    and every score quietly optimistic."""
    state, tools = toy_run

    assert state.split_artifact is not None
    manifest = json.loads(tools.read_artifact(state.split_artifact).content)

    train, holdout = set(manifest["train"]), set(manifest["holdout"])
    assert train & holdout == set()
    assert train | holdout == set(range(200))
    for fold in manifest["folds"]:
        assert set(fold["train"]) & set(fold["valid"]) == set()
        assert set(fold["valid"]) & holdout == set(), "a fold must not validate on the holdout"


def test_the_split_is_reproducible_from_the_seed(tmp_path: Path):
    """Two runs of the same config have to produce the same partition, or no result is
    reproducible and no ablation is a comparison."""
    first, first_tools = run(tmp_path / "a")
    second, second_tools = run(tmp_path / "b")

    left = json.loads(first_tools.read_artifact(first.split_artifact).content)
    right = json.loads(second_tools.read_artifact(second.split_artifact).content)
    assert left["holdout"] == right["holdout"]


def test_the_stub_is_recorded_as_the_model(toy_run):
    """A results row built from a stub must be identifiable as one."""
    state, _tools = toy_run

    assert {event.model for event in state.node_trace} == {"stub"}
