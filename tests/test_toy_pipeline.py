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

import pandas as pd
import pytest

from ds_agents.cli import _toy_state
from ds_agents.graph import run_pipeline
from ds_agents.state import PipelineState
from ds_agents.tools.llm import StubModel
from ds_agents.tools.local import LocalTools
from ds_agents.tools.mcp_client import MCPTools, stdio_params

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

    assert [event.node for event in state.node_trace] == [
        "intake",
        "profiler",
        "feature_eng",
        "modeler",
        "router",
        "reporter",
    ]
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

    # router and reporter call no model, so they carry None rather than a name.
    assert {event.model for event in state.node_trace} == {"stub", None}
    assert state.placeholder_models() == ["stub"]
    assert state.publishable()[0] is False


def test_final_features_are_source_names_not_one_hot_expansions(toy_run):
    """`results_row()` intersects `final_features` with `planted_leakage_columns`, which hold
    source names. A dummy name here empties that intersection and scores every run as remediated
    with the leak still present -- a failure that only ever flatters the system."""
    state, _tools = toy_run

    assert state.final_features is not None
    assert not any("=" in feature for feature in state.final_features)
    assert "region" in state.final_features


def test_the_id_column_is_dropped_without_anyone_flagging_it(toy_run):
    """The stub proposes no drops at all, so this is the node's own forced-drop rule."""
    state, _tools = toy_run

    assert "customer_id" in state.dropped_features
    assert "customer_id" not in (state.final_features or [])


def test_the_stub_run_keeps_the_leak_and_the_score_is_inflated(toy_run):
    """The leaky baseline, and the number the whole eval exists to catch.

    `StubModel` nominates nothing and proposes no drops, so `account_status_code` survives into
    the matrix and the claimed score is ~0.98 against ~0.83 for an honest feature set. This test
    asserts the *inflated* number on purpose: if it ever drops, either the leak stopped leaking or
    something started removing it, and both change what every later ablation is measured against.
    """
    state, _tools = toy_run

    assert state.chosen_model is not None
    assert "account_status_code" in (state.final_features or [])
    assert state.chosen_model.claimed_holdout_score > 0.95
    assert state.top_importances[0][0] == "account_status_code"


def test_every_node_that_ran_left_a_trace_event_and_a_report(toy_run):
    """The reporter must produce an artifact even though nothing scores it, because the harness
    needs a row for every dataset including the ones that fail."""
    state, tools = toy_run

    assert state.report_artifact is not None
    report = tools.read_artifact(state.report_artifact).content
    assert "ds-agents run report" in report
    assert state.importance_artifact is not None


def test_the_report_does_not_leak_the_planted_answer(toy_run):
    """`planted_leakage_columns` is on the state the reporter receives. Rendering it would put the
    answer key in the artifact store that the Phase 5 generalist arm reads."""
    state, tools = toy_run

    assert state.planted_leakage_columns == ["account_status_code"]
    report = tools.read_artifact(state.report_artifact).content
    # The column name appears legitimately (it survived into the features). What must not appear
    # is any statement that it was the planted one.
    assert "planted" not in report.lower()


def test_the_same_run_over_mcp_lands_in_the_same_place(toy_run, tmp_path: Path):
    """The transport is supposed to be the only difference between the two bindings.

    They are the same objects reached two ways -- the server wraps a `LocalTools` -- so a
    divergence here is not a transport bug to be papered over, it means the protocol layer started
    deciding something. Asserting the claimed score to the digit is deliberate: everything in the
    run is seeded, so an inexact match is a reproducibility finding either way.
    """
    local_state, _tools = toy_run

    with MCPTools(stdio_params(tmp_path / "mcp", TOY, "toy")) as tools:
        state = run_pipeline(_toy_state(), tools=tools, model=StubModel())

    assert state.errors == []
    assert [e.node for e in state.node_trace] == [e.node for e in local_state.node_trace]
    assert state.spec == local_state.spec
    assert state.final_features == local_state.final_features
    assert state.dropped_features == local_state.dropped_features
    assert state.chosen_model.name == local_state.chosen_model.name
    assert (
        state.chosen_model.claimed_holdout_score == local_state.chosen_model.claimed_holdout_score
    )
    assert state.top_importances == local_state.top_importances


def test_a_lower_is_better_metric_reports_scores_in_natural_units(tmp_path: Path):
    """The regression branch, through the real snippet, because the toy fixture is binary.

    sklearn scorers are always greater-is-better: `neg_root_mean_squared_error` returns -0.30 for
    an rmse of 0.30. If that convention escapes the snippet, three things break at once and no
    binary test notices: `best_by_cv` picks the WORST candidate, the prompt's stated direction
    contradicts the numbers it shows, and `claimed_holdout_score` lands in `results_row()` with the
    opposite sign from the harness's `verified_holdout_score`, fabricating a `holdout_claim_gap` of
    roughly twice the true error on every regression row.

    This runs the real `MODEL_SNIPPET` rather than asserting on canned JSON, because canned JSON is
    exactly what let the bug through the first time.
    """
    import json as _json
    import subprocess
    import sys

    from ds_agents.nodes.modeler import (
        CANDIDATE_SPECS,
        MODEL_SNIPPET,
        SCORERS,
        SEED_SENTINEL,
    )

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    # A tiny regression frame where the better model is unambiguous: `signal` predicts `y` almost
    # exactly, `noise` does not.
    frame = pd.DataFrame(
        {
            "signal": [float(i) for i in range(60)],
            "noise": [float((i * 7) % 5) for i in range(60)],
            "y": [float(i) * 2.0 + 1.0 for i in range(60)],
        }
    )
    csv = tmp_path / "reg.csv"
    frame.to_csv(csv, index=False)

    rows = list(range(60))
    split = {
        "train": rows[:45],
        "holdout": rows[45:],
        "folds": [{"train": rows[:30], "valid": rows[30:45]}],
    }
    transform = (
        "import pandas as pd\n"
        "SOURCE_COLUMNS = ['signal', 'noise']\n"
        "FEATURE_ORDER = ['signal', 'noise']\n"
        "def transform(df):\n"
        "    return df[FEATURE_ORDER].astype('float64')\n"
    )

    code = MODEL_SNIPPET.format(
        target="y",
        task_type="regression",
        positive_class=None,
        scoring=SCORERS["rmse"],
        seed=20260822,
        n_repeats=3,
        top_n=15,
        specs=CANDIDATE_SPECS["regression"],
        split_json=_json.dumps(split),
        feature_code=transform,
        greater_is_better=False,
        sentinel=SEED_SENTINEL,
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"DS_DATASET": str(csv), "DS_ARTIFACTS": str(artifacts), "PATH": ""},
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    out = _json.loads(result.stdout)

    scored = [c for c in out["candidates"] if c["cv_mean"] is not None]
    assert scored, "no candidate fit on the regression fixture"
    for candidate in scored:
        # An rmse is a distance. A negative one means sklearn's sign convention escaped.
        assert candidate["cv_mean"] >= 0, f"{candidate['name']} cv_mean is negative: sign leaked"
        assert candidate["holdout_score"] >= 0, f"{candidate['name']} holdout_score is negative"

    # And the direction is applied exactly once: best_by_cv must be the SMALLEST error.
    best = min(scored, key=lambda c: c["cv_mean"])["name"]
    assert out["best_by_cv"] == best, (
        f"best_by_cv chose {out['best_by_cv']!r} but the lowest rmse was {best!r}; "
        f"the direction is being applied twice"
    )
