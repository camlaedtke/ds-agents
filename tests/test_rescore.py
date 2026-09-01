"""The grader, and the one question that decides whether it is an instrument or a decoration.

The question is not "does it produce a number". It is "would it produce a DIFFERENT number when the
agents overstate themselves". `TestTheInstrumentDetectsAnOverclaim` answers it by construction: a
dataset whose one informative column is informative in exactly the rows the agents get, and pure
noise in the rows they are graded on. A grader that scores that run at the claimed number is not
measuring the withheld rows at all, and every `holdout_claim_gap` this project ever publishes would
be a rounding error dressed as a finding.

The rest is statuses. Each one gets its own test rather than sharing a catch-all, because the whole
reason `rescore_status` is an enum is that a reader has to be able to tell "no dataset was withheld"
from "the graph produced no model" from "the grader's sandbox died".
"""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ds_agents import rescore
from ds_agents.cli import _run_state
from ds_agents.graph import run_pipeline
from ds_agents.holdout import _withhold_rows, prepare
from ds_agents.nodes.modeler import CANDIDATE_SPECS, SEED_SENTINEL
from ds_agents.rescore import REFIT_TOLERANCE, RescoreInputs, RescoreOutcome
from ds_agents.runnable import Runnable, resolve
from ds_agents.state import (
    DEFAULT_RANDOM_SEED,
    ModelResult,
    NodeEvent,
    PipelineState,
    TaskSpec,
    utc_now,
)
from ds_agents.tools.llm import StubModel
from ds_agents.tools.local import LocalTools


def _runnable(csv_path: Path, target: str = "y") -> Runnable:
    return Runnable(
        dataset_id="synthetic",
        csv_path=csv_path,
        target=target,
        task_description=f"Predict {target} and report roc_auc.",
        planted_columns=[],
        withheld_fraction=0.2,
        source="benchmark",
    )


def _write_split_leak(path: Path, n: int = 400, seed: int = 7) -> None:
    """A dataset whose only signal is present for the agents and absent where they are graded.

    The carve is deterministic given the label vector, so the withheld positions are computable
    BEFORE the file is written. `leak` equals the target everywhere the agents will see it and is
    a coin flip everywhere they will not. A faithful grader must report ~1.0 claimed and ~0.5
    verified; a grader that quietly rescored the agents' own rows would report ~1.0 twice.
    """
    rng = np.random.default_rng(seed)
    y = np.array([i % 2 for i in range(n)])
    withheld = set(_withhold_rows([str(v) for v in y], 0.2, DEFAULT_RANDOM_SEED))
    leak = np.array(
        [rng.integers(0, 2) if i in withheld else y[i] for i in range(n)], dtype="int64"
    )
    pd.DataFrame({"leak": leak, "noise": rng.normal(size=n), "y": y}).to_csv(path, index=False)


def _write_high_cardinality_signal(
    path: Path, n: int = 400, n_levels: int = 60, seed: int = 11
) -> None:
    """A dataset the agents structurally cannot use and the baseline can.

    `group` is a 60-level STRING column and the target is `level < 30` -- one ordinal threshold.
    `feature_eng` decides encodable columns by dtype and cardinality on the train rows, and skips a
    string column above MAX_ONE_HOT_LEVELS (20), so `group` never reaches `SOURCE_COLUMNS` and the
    pipeline is fit on `noise` alone. The grader's own encoder keeps it, so the RandomForest finds
    the threshold the agents were never shown.

    This is the ONLY mechanism in this file that separates the baseline from the pipeline.
    `_write_split_leak` cannot: under `StubModel` nothing is nominated and `feature_eng` drops
    nothing, so both would see the same columns and a grader that simply reported the pipeline's
    number twice would pass. The gap this fixture forces is what says it does not.
    """
    rng = np.random.default_rng(seed)
    level = rng.integers(0, n_levels, size=n)
    pd.DataFrame(
        {
            "group": [f"g{v:02d}" for v in level],
            "noise": rng.normal(size=n),
            "y": (level < n_levels // 2).astype("int64"),
        }
    ).to_csv(path, index=False)


def _pipeline_run(tmp_path: Path, csv_path: Path) -> PipelineState:
    dataset = _runnable(csv_path)
    prepared = prepare(
        dataset,
        "descriptive",
        into=tmp_path / "input",
        withheld_into=tmp_path / "withheld",
        seed=DEFAULT_RANDOM_SEED,
    )
    root = tmp_path / "run"
    tools = LocalTools(root, dataset_path=prepared.agent_csv, dataset_id=dataset.dataset_id)
    try:
        state = run_pipeline(_run_state(dataset, prepared=prepared), tools=tools, model=StubModel())
        inputs = rescore.read_inputs(state, tools)
    finally:
        tools.close()
    assert isinstance(inputs, RescoreInputs), f"could not read the run's artifacts: {inputs}"
    outcome = rescore.rescore(state, prepared, inputs, root=tmp_path / "rescore")
    scale = rescore.baseline(state, prepared, inputs, outcome, root=tmp_path / "baseline")
    return rescore.apply(state, outcome, scale)


class TestTheInstrumentDetectsAnOverclaim:
    """The test this whole module exists to pass."""

    def test_a_claim_that_does_not_survive_the_withheld_rows_is_caught(self, tmp_path):
        csv_path = tmp_path / "split_leak.csv"
        _write_split_leak(csv_path)
        state = _pipeline_run(tmp_path, csv_path)

        assert state.rescore_status == "ok", state.rescore_detail
        assert state.chosen_model is not None
        claimed = state.chosen_model.claimed_holdout_score
        verified = state.verified_holdout_score
        assert claimed is not None and verified is not None
        assert claimed > 0.9, f"the leak did not reach the model; claimed {claimed}"
        assert verified < 0.7, f"the withheld rows were not independent; verified {verified}"
        row = state.results_row()
        assert row["holdout_claim_gap"] > 0.25, (
            "a gap this large is the whole finding; if it collapses, the grader is rescoring the "
            "agents' own rows"
        )

    def test_the_refit_reproduces_the_claim_on_the_agents_own_holdout(self, tmp_path):
        """The self-check that earns the withheld number, on the same run.

        `refit_claim_gap` near zero says the pipeline the grader built IS the one the modeler
        scored -- same transform, same seed, same positive class, same scorer sign, same split.
        Without it the number above is a number computed on some rows.
        """
        csv_path = tmp_path / "split_leak.csv"
        _write_split_leak(csv_path)
        state = _pipeline_run(tmp_path, csv_path)
        assert state.refit_claim_gap is not None
        assert abs(state.refit_claim_gap) <= REFIT_TOLERANCE

    def test_no_file_the_run_could_reach_contains_a_withheld_row(self, tmp_path):
        """Containment, asserted rather than assumed.

        Stated honestly: the sandbox has no filesystem namespace, so a snippet could in principle
        open the withheld CSV by relative path. What this test proves is the property the layout
        actually gives -- nothing the run WROTE, and nothing it was mounted on, carries a withheld
        row. That is what `verified_holdout_score` rests on. Real isolation waits on the Docker
        backend parked behind `SandboxPool`.
        """
        csv_path = tmp_path / "split_leak.csv"
        _write_split_leak(csv_path)
        _pipeline_run(tmp_path, csv_path)

        withheld = [
            line
            for line in (tmp_path / "withheld" / "split_leak.csv").read_text().splitlines()[1:]
            if line.strip()
        ]
        assert withheld, "nothing was withheld, so this test proves nothing"
        # The grader's own directory is excluded on purpose: it is the one place the withheld rows
        # are SUPPOSED to be readable, and it is created after the graph has finished.
        for path in (tmp_path / "run").rglob("*"):
            if not path.is_file() or "rescore" in path.parts:
                continue
            try:
                text = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for row in withheld:
                assert row not in text, f"{path} carries a withheld row"

    def test_the_withheld_rows_are_not_in_the_split_the_agents_were_given(self, tmp_path):
        csv_path = tmp_path / "split_leak.csv"
        _write_split_leak(csv_path)
        _pipeline_run(tmp_path, csv_path)
        split_path = tmp_path / "run" / "artifacts"
        manifests = list(split_path.rglob("split_manifest.json"))
        assert manifests, "the profiler wrote no split manifest"
        split = json.loads(manifests[0].read_text())
        assert split["n_rows"] == 320, "the agents' frame is not the post-carve frame"
        assert max(split["train"] + split["holdout"]) < 320


@pytest.mark.fast
class TestTheRefitRecipeHasOneSource:
    def test_the_resolved_spec_matches_what_the_modeler_recorded(self):
        """`ModelResult.params` is a flat merge across steps, so the grader rebuilds from
        `CANDIDATE_SPECS` instead. This pins the two together: if they drift, the grader is
        fitting a different estimator from the one that produced the claim."""
        for task_type in ("binary", "multiclass", "regression"):
            for name, spec in CANDIDATE_SPECS[task_type].items():
                merged: dict = {}
                for _step, _path, params in spec:
                    merged.update(
                        {
                            k: (DEFAULT_RANDOM_SEED if v == SEED_SENTINEL else v)
                            for k, v in params.items()
                        }
                    )
                assert merged, f"{task_type}/{name} resolved to no params at all"
                assert SEED_SENTINEL not in merged.values()


def _state(**update) -> PipelineState:
    """A minimal state that clears every re-scorer precondition, so a test can fail exactly one."""
    base = dict(
        dataset_id="d",
        task_description="x",
        spec=TaskSpec(
            target="y", task_type="binary", metric="roc_auc", split_strategy="stratified"
        ),
        final_features=["a"],
        chosen_model=ModelResult(name="logistic_l2", claimed_holdout_score=0.9),
    )
    base.update(update)
    return PipelineState(**base)


def _prepared(tmp_path, *, withheld: bool = True):
    csv_path = tmp_path / "d.csv"
    rows = "a,y\n" + "".join(f"{i},{i % 2}\n" for i in range(60))
    csv_path.write_text(rows)
    dataset = _runnable(csv_path).model_copy(update={"withheld_fraction": 0.2 if withheld else 0.0})
    return prepare(
        dataset,
        "descriptive",
        into=tmp_path / "i",
        withheld_into=tmp_path / "w",
        seed=DEFAULT_RANDOM_SEED,
    )


@pytest.mark.fast
class TestEveryStatusHasItsOwnReason:
    """One test per non-`ok` status. A catch-all here would defeat the point of the enum."""

    _state = staticmethod(_state)
    _prepared = staticmethod(_prepared)

    def _run(self, tmp_path, state, *, withheld=True) -> RescoreOutcome:
        return rescore.rescore(
            state,
            self._prepared(tmp_path, withheld=withheld),
            RescoreInputs(feature_code="", split_json="{}"),
            root=tmp_path / "rescore",
        )

    def test_a_fixture_is_not_graded_and_says_so(self, tmp_path):
        outcome = self._run(tmp_path, self._state(), withheld=False)
        assert outcome.status == "no_withheld_holdout"
        assert outcome.verified_holdout_score is None

    def test_no_spec(self, tmp_path):
        outcome = self._run(tmp_path, self._state(spec=None))
        assert outcome.status == "no_spec"
        assert outcome.verified_holdout_score is None

    def test_no_model(self, tmp_path):
        """The recorded 2026-08-31 failure: a run that produced no model and passed review."""
        outcome = self._run(tmp_path, self._state(chosen_model=None))
        assert outcome.status == "no_model"
        assert outcome.verified_holdout_score is None

    def test_empty_matrix(self, tmp_path):
        outcome = self._run(tmp_path, self._state(final_features=[]))
        assert outcome.status == "empty_matrix"
        assert outcome.verified_holdout_score is None

    def test_unknown_model_spec(self, tmp_path):
        outcome = self._run(
            tmp_path, self._state(chosen_model=ModelResult(name="xgboost_from_2019"))
        )
        assert outcome.status == "unknown_model_spec"
        assert "CANDIDATE_SPECS" in outcome.detail

    def test_a_broken_snippet_is_reported_not_raised(self, tmp_path):
        """Feature code that will not exec. The run already happened; losing its row would drop
        exactly the worst outcomes and bias every table upward."""
        outcome = rescore.rescore(
            self._state(),
            self._prepared(tmp_path),
            RescoreInputs(feature_code="this is not python", split_json="{}"),
            root=tmp_path / "rescore",
        )
        assert outcome.status == "snippet_failed"
        assert outcome.verified_holdout_score is None
        assert outcome.detail

    def test_no_feature_code_when_feature_eng_produced_none(self):
        outcome = rescore.read_inputs(self._state(), tools=None)  # type: ignore[arg-type]
        assert isinstance(outcome, RescoreOutcome)
        assert outcome.status == "no_feature_code"


@pytest.mark.fast
class TestTheSelfCheckIsLoadBearing:
    def _state(self, claimed: float) -> PipelineState:
        return PipelineState(
            dataset_id="d",
            task_description="x",
            chosen_model=ModelResult(name="logistic_l2", claimed_holdout_score=claimed),
        )

    def test_a_disagreeing_refit_keeps_the_score_and_flags_it(self):
        """Kept, not deleted. Deleting would hide the finding; the status is what says do not
        pool it."""
        outcome = rescore._outcome(
            self._state(0.90),
            {"verified_holdout_score": 0.61, "refit_agent_holdout_score": 0.72},
        )
        assert outcome.status == "refit_mismatch"
        assert outcome.verified_holdout_score == 0.61
        assert outcome.refit_claim_gap == pytest.approx(-0.18)
        assert "0.9" in outcome.detail

    def test_an_agreeing_refit_is_ok(self):
        outcome = rescore._outcome(
            self._state(0.90),
            {"verified_holdout_score": 0.61, "refit_agent_holdout_score": 0.90},
        )
        assert outcome.status == "ok"
        assert outcome.verified_holdout_score == 0.61

    def test_a_single_class_holdout_is_its_own_status(self):
        outcome = rescore._outcome(
            self._state(0.90),
            {
                "verified_holdout_score": None,
                "refit_agent_holdout_score": 0.90,
                "single_class_holdout": True,
                "n_withheld_rows": 40,
            },
        )
        assert outcome.status == "single_class_holdout"
        assert outcome.verified_holdout_score is None
        assert outcome.n_withheld_rows == 40


@pytest.mark.fast
class TestAGraderFailureIsNotARunFailure:
    def test_a_rescore_failure_appends_no_error_and_still_publishes(self):
        """`errored` means the RUN went wrong. ARCHITECTURE requires a row even on hard failure,
        because losing the rows for the worst outcomes biases every table upward."""
        state = PipelineState(dataset_id="d", task_description="x")
        state = state.model_copy(update={"node_trace": [], "errors": []})
        graded = rescore.apply(state, RescoreOutcome(status="no_model", detail="none promoted"))
        assert graded.errors == []
        assert graded.results_row()["errored"] is False
        assert graded.results_row()["rescore_status"] == "no_model"


@pytest.mark.fast
class TestTheCarveIsNotInTheCache:
    def test_a_benchmark_run_never_writes_into_the_shared_dataset_cache(self, tmp_path):
        """`materialize` under `descriptive` hands back the source's own path, which for a
        benchmark dataset is the shared `.cache/datasets/` copy. Writing the carved frame there
        would corrupt it for every later run of every dataset."""
        dataset = resolve("credit_g")
        before = dataset.csv_path.read_bytes()
        prepared = prepare(
            dataset,
            "descriptive",
            into=tmp_path / "i",
            withheld_into=tmp_path / "w",
            seed=DEFAULT_RANDOM_SEED,
        )
        assert prepared.agent_csv != dataset.csv_path
        assert dataset.csv_path.read_bytes() == before


class TestTheBaselineAndThePipelineCanDisagree:
    """The question `TestTheInstrumentDetectsAnOverclaim` asks, asked of the yardstick.

    Not "does it produce a number" but "would it produce a DIFFERENT number from the pipeline's".
    A baseline that merely tracked the run would put every row at 1.0 and say nothing, and
    `baseline_normalised_score` would be an expensive way of writing a constant.
    """

    def test_the_random_forest_finds_what_the_agents_were_never_shown(self, tmp_path):
        csv_path = tmp_path / "high_card.csv"
        _write_high_cardinality_signal(csv_path)
        state = _pipeline_run(tmp_path, csv_path)

        assert state.rescore_status == "ok", state.rescore_detail
        assert state.baseline_status == "ok", state.baseline_detail
        verified = state.verified_holdout_score
        unit = state.baseline_unit_score
        assert verified is not None and unit is not None
        assert verified < 0.65, (
            f"the pipeline should have had only noise to fit on; verified {verified}"
        )
        assert unit > 0.9, f"the baseline should have found the ordinal threshold; unit {unit}"
        assert unit - verified > 0.3, (
            "the baseline is tracking the pipeline rather than measuring the data"
        )

    def test_a_run_that_lost_the_signal_reads_near_the_floor_of_the_scale(self, tmp_path):
        """0.0 is the constant-prior predictor and 1.0 is the RandomForest. A run that saw none of
        the signal belongs at the bottom of that scale, and the number says so without anyone
        having to compare two columns by eye."""
        csv_path = tmp_path / "high_card.csv"
        _write_high_cardinality_signal(csv_path)
        state = _pipeline_run(tmp_path, csv_path)

        normalised = state.results_row()["baseline_normalised_score"]
        assert normalised is not None
        assert normalised < 0.2, f"expected a run near the floor, got {normalised}"

    def test_the_recipe_is_stamped_on_the_row(self, tmp_path):
        """Two rows graded against different yardsticks must not be pooled, and a reader holding a
        results file cannot see a commit."""
        csv_path = tmp_path / "high_card.csv"
        _write_high_cardinality_signal(csv_path)
        row = _pipeline_run(tmp_path, csv_path).results_row()
        assert row["baseline_recipe"] == rescore.BASELINE_RECIPE


class TestTheZeroPointIsACorrectnessAssertion:
    def test_a_constant_class_prior_predictor_scores_exactly_one_half(self, tmp_path):
        """roc_auc of a constant score is 0.5 by construction -- every pair is a tie. So this is
        not a measurement with a tolerance, it is an assertion that the grader resolved the
        positive class, applied the scorer sign, and scored the rows it meant to. Anything but 0.5
        means one of those is wrong, and would be invisible in `verified_holdout_score` alone.

        Conditional on the metric, and checked rather than assumed: `metric` is chosen by intake,
        which is a model, and `Runnable` deliberately does not carry one.
        """
        csv_path = tmp_path / "high_card.csv"
        _write_high_cardinality_signal(csv_path)
        state = _pipeline_run(tmp_path, csv_path)

        assert state.spec is not None
        if state.spec.metric != "roc_auc":
            pytest.skip(f"intake chose {state.spec.metric!r}; the 0.5 identity is roc_auc's")
        assert state.baseline_zero_score == pytest.approx(0.5, abs=1e-12)


class TestTheBaselineCanAlsoFail:
    """The mirror image, and the reason the two raw points are published rather than the ratio
    alone: the baseline is not a system that always wins."""

    def test_a_leak_fools_the_baseline_exactly_as_it_fools_the_pipeline(self, tmp_path):
        """`_write_split_leak`'s `leak` column equals the target on the train rows and is a coin
        flip on the withheld rows. The baseline keeps every raw column, so it fits on `leak` too
        and collapses on the same rows the pipeline collapses on."""
        csv_path = tmp_path / "split_leak.csv"
        _write_split_leak(csv_path)
        state = _pipeline_run(tmp_path, csv_path)

        assert state.baseline_status == "ok", state.baseline_detail
        assert state.baseline_unit_score is not None
        assert state.baseline_unit_score < 0.7, (
            f"the baseline is not 'the RandomForest always wins'; unit {state.baseline_unit_score}"
        )

    def test_a_scale_with_no_length_withholds_the_ratio_and_keeps_the_points(self, tmp_path):
        """The separation guard firing on a REAL run rather than on a hand-written state. Both
        points were measured correctly, so the status stays `ok` and both numbers stay on the row;
        only the quotient is withheld, because there is no scale to place anything on."""
        csv_path = tmp_path / "split_leak.csv"
        _write_split_leak(csv_path)
        row = _pipeline_run(tmp_path, csv_path).results_row()

        assert row["baseline_status"] == "ok"
        assert row["baseline_zero_score"] is not None
        assert row["baseline_unit_score"] is not None
        separation = row["baseline_unit_score"] - row["baseline_zero_score"]
        if separation > 1e-9:
            pytest.skip(f"the two points separated by {separation}; nothing to guard here")
        assert row["baseline_normalised_score"] is None


class TestEveryBaselineStatusHasItsOwnReason:
    """One test per value. A catch-all here would defeat the point of the enum, which is that
    "no dataset was withheld", "there was no score to place on a scale" and "the RandomForest
    died" are three facts with different consequences for a table.
    """

    @staticmethod
    def _outcome(**update) -> RescoreOutcome:
        base = {"status": "ok", "verified_holdout_score": 0.8}
        return RescoreOutcome(**{**base, **update})

    def test_a_fixture_withholds_nothing_and_says_so(self, tmp_path):
        prepared = _prepared(tmp_path, withheld=False)
        result = rescore.baseline_precondition(prepared, self._outcome())
        assert result.status == "no_withheld_holdout"

    def test_no_score_means_no_scale_and_carries_which_rescore_status_it_was(self, tmp_path):
        """The coupling that genuinely exists, named on the row so a reader never has to join two
        columns to find out why this one is null."""
        prepared = _prepared(tmp_path, withheld=True)
        result = rescore.baseline_precondition(
            prepared, self._outcome(status="no_model", verified_holdout_score=None)
        )
        assert result.status == "rescore_unavailable"
        assert "no_model" in result.detail

    def test_an_unparseable_snippet_is_reported_not_raised(self, tmp_path):
        """A split manifest with no `train` key. The snippet raises inside the sandbox, and the
        grader records that rather than taking the run down with it."""
        prepared = _prepared(tmp_path, withheld=True)
        state = _state()
        inputs = RescoreInputs(feature_code="", split_json="{}")
        result = rescore.baseline(
            state, prepared, inputs, self._outcome(), root=tmp_path / "baseline"
        )
        assert result.status == "snippet_failed"
        assert result.detail

    def test_an_empty_source_matrix_has_its_own_name(self):
        assert (
            rescore._baseline_outcome({"status": "empty_source_matrix", "detail": "no columns"})
        ).status == "empty_source_matrix"

    def test_a_single_class_train_split_has_its_own_name(self):
        assert (
            rescore._baseline_outcome({"status": "single_class_train"})
        ).status == "single_class_train"

    def test_a_failed_zero_point_keeps_the_unit_point(self):
        result = rescore._baseline_outcome(
            {"status": "zero_point_failed", "zero_score": None, "unit_score": 0.81}
        )
        assert result.status == "zero_point_failed"
        assert result.unit_score == 0.81

    def test_a_failed_unit_point_keeps_the_zero_point(self):
        """The single most important assertion here, because it is the reason the baseline runs in
        its own process at all. The RandomForest is the thing most likely to die on a wide frame,
        and when it does the zero point is still a measurement -- discarding it would hide that
        the scale has a floor and no ceiling.
        """
        result = rescore._baseline_outcome(
            {
                "status": "unit_point_failed",
                "zero_score": 0.5,
                "unit_score": None,
                "detail": "MemoryError",
            }
        )
        assert result.status == "unit_point_failed"
        assert result.zero_score == 0.5
        assert result.unit_score is None
        assert result.recipe == rescore.BASELINE_RECIPE, (
            "the unit point was attempted, so the row must say which recipe failed"
        )

    def test_a_sandbox_that_will_not_start_is_reported_not_raised(self, tmp_path, monkeypatch):
        prepared = _prepared(tmp_path, withheld=True)

        def _boom(*args, **kwargs):
            raise RuntimeError("no worker")

        monkeypatch.setattr("ds_agents.tools.local.LocalTools.run_python", _boom)
        result = rescore.baseline(
            _state(),
            prepared,
            RescoreInputs(feature_code="", split_json='{"train": [0], "holdout": [1]}'),
            self._outcome(),
            root=tmp_path / "baseline",
        )
        assert result.status == "sandbox_error"
        assert "RuntimeError" in result.detail

    def test_a_status_the_snippet_never_names_falls_back_to_snippet_failed(self):
        """A payload with no `status` is a contract violation, not an `ok`."""
        assert rescore._baseline_outcome({}).status == "snippet_failed"

    def test_only_an_attempted_unit_point_stamps_a_recipe(self):
        """A row that never reached the RandomForest must not claim to have been graded against
        it. An empty `baseline_recipe` is what says the yardstick was never built."""
        assert rescore._baseline_outcome({"status": "single_class_train"}).recipe == ""
        assert rescore._baseline_outcome({"status": "ok", "unit_score": 0.7}).recipe == (
            rescore.BASELINE_RECIPE
        )


class TestABaselineFailureIsNotARunFailure:
    def test_a_missing_scale_never_appends_a_pipeline_error(self):
        """Same rule as `rescore_status`: `errored` means the RUN went wrong. Overloading it with
        "the yardstick went wrong" is the defect NEXT.md already records against it."""
        graded = rescore.apply(
            _state(),
            RescoreOutcome(status="ok", verified_holdout_score=0.8),
            rescore.BaselineOutcome(status="unit_point_failed", zero_score=0.5),
        )
        assert graded.errors == []
        row = graded.results_row()
        assert row["errored"] is False
        assert row["baseline_status"] == "unit_point_failed"
        assert row["baseline_zero_score"] == 0.5
        assert row["baseline_normalised_score"] is None


@pytest.mark.fast
class TestTheGradersOwnWallCostIsOnTheRow:
    """`wall_seconds` stops when the graph returns, and the grader runs after it.

    Which means the term this project most needs to price -- a unit-point fit measured at 0.2s on
    credit_g and 72s on a 98k-row frame -- was invisible in every row committed before 2026-09-01.
    It was worse than invisible: the four `baseline-smoke` rows read FASTER (19.7-21.0s) than the
    four `credit-g-smoke` rows that did two fewer fits (22.8-27.3s), because the fits were never in
    the number and run-to-run LLM latency spread is about 4s.
    """

    def test_both_durations_are_none_when_neither_half_ran(self):
        """`None` and `0.0` are different claims. The precondition path does no work at all, and a
        0.0 there would read as a fit that took no time."""
        graded = rescore.apply(_state(), RescoreOutcome(status="no_model", detail="none"))

        assert graded.rescore_seconds is None
        assert graded.baseline_seconds is None
        row = graded.results_row()
        assert row["rescore_seconds"] is None
        assert row["baseline_seconds"] is None

    def test_the_two_halves_are_timed_separately(self):
        """Two processes, two timeouts, two failure modes -- so two columns. A single
        `grader_seconds` could not say which half a 900s stall was in."""
        graded = rescore.apply(
            _state(),
            RescoreOutcome(status="no_model", detail="none"),
            None,
            rescore_seconds=1.25,
            baseline_seconds=72.09,
        )

        row = graded.results_row()
        assert row["rescore_seconds"] == pytest.approx(1.25)
        assert row["baseline_seconds"] == pytest.approx(72.09)


class TestTheTimingIsActuallyWiredUp:
    """The unit tests above prove `apply` writes what it is HANDED. This one proves the call site
    hands it something -- which is the half that a live run would otherwise be the first to check,
    at eight runs' worth of money.
    """

    def test_a_graded_run_through_the_cli_records_both_durations(self, tmp_path):
        from ds_agents.cli import _run_once

        csv_path = tmp_path / "d.csv"
        _write_split_leak(csv_path)
        dataset = _runnable(csv_path)
        prepared = prepare(
            dataset,
            "descriptive",
            into=tmp_path / "input",
            withheld_into=tmp_path / "withheld",
            seed=DEFAULT_RANDOM_SEED,
        )
        state = _run_once(
            dataset,
            root=tmp_path / "run",
            prepared=prepared,
            commit="testcommit",
            transport="local",
            no_live=True,
        )

        row = state.results_row()
        assert row["rescore_seconds"] is not None, "the grader ran but its duration is unrecorded"
        assert row["baseline_seconds"] is not None, (
            "the baseline ran but its duration is unrecorded"
        )
        assert row["rescore_seconds"] > 0
        assert row["baseline_seconds"] > 0
        # The whole point: these are NOT inside wall_seconds, so the grader is extra wall time that
        # every committed row hid.
        assert row["node_seconds"], "node_seconds must not be empty on a run that executed nodes"


@pytest.mark.fast
class TestWhereTheWallTimeWent:
    """`NodeEvent` has carried per-node seconds since Phase 1 and only `ds-agents run` ever printed
    them. The harness calls `_run_once` directly, so every committed row discarded the one column
    that answers "which node dominates at scale"."""

    def test_a_node_visited_twice_is_summed_not_overwritten(self):
        """The review loop revisits `feature_eng` and `modeler`. A dict keyed by node name that
        overwrote would silently keep only the last visit, understating exactly the runs that
        looped most."""
        started = utc_now()
        state = _state(
            node_trace=[
                NodeEvent(node="modeler", started=started, ended=started + timedelta(seconds=2)),
                NodeEvent(node="modeler", started=started, ended=started + timedelta(seconds=3)),
            ]
        )

        assert state.node_seconds == {"modeler": 5.0}
        assert state.results_row()["node_seconds"] == {"modeler": 5.0}

    def test_an_unfinished_node_contributes_nothing_rather_than_raising(self):
        """A run that died mid-node still has a row worth writing."""
        started = utc_now()
        state = _state(
            node_trace=[
                NodeEvent(node="intake", started=started, ended=started + timedelta(seconds=1)),
                NodeEvent(node="profiler", started=started, ended=None),
            ]
        )

        assert state.node_seconds == {"intake": 1.0}
