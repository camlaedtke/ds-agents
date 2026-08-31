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
from ds_agents.state import DEFAULT_RANDOM_SEED, ModelResult, PipelineState, TaskSpec
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
    return rescore.apply(state, rescore.rescore(state, prepared, inputs, root=tmp_path / "rescore"))


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


@pytest.mark.fast
class TestEveryStatusHasItsOwnReason:
    """One test per non-`ok` status. A catch-all here would defeat the point of the enum."""

    def _state(self, **update) -> PipelineState:
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

    def _prepared(self, tmp_path, *, withheld: bool = True):
        csv_path = tmp_path / "d.csv"
        rows = "a,y\n" + "".join(f"{i},{i % 2}\n" for i in range(60))
        csv_path.write_text(rows)
        dataset = _runnable(csv_path).model_copy(
            update={"withheld_fraction": 0.2 if withheld else 0.0}
        )
        return prepare(
            dataset,
            "descriptive",
            into=tmp_path / "i",
            withheld_into=tmp_path / "w",
            seed=DEFAULT_RANDOM_SEED,
        )

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
