"""The generator's own logic, on fixed frames and with no network.

`tests/test_benchmark_provenance.py` proves the committed manifest matches its sources, but it is
opt-in and it only ever exercises the happy path -- every dataset it sees was, by definition,
admitted. What is tested here is the part that decides WHAT GOES IN: every rejection branch of
`select`, the two early returns in `measure`, and the check that a documented leak names a real
column. Those branches are what make the manifest's composition a rule rather than a preference,
and none of them ran under `pytest` before this file existed.
"""

from dataclasses import replace

import pandas as pd
import pytest
from pydantic import ValidationError

from ds_agents.benchmark import SELECTION_RULE, DatasetEntry, KnownLeak
from ds_agents.benchmark_build import (
    Candidate,
    Measured,
    _dataset_id,
    _n_usable_features,
    _write_csv,
    published_reference,
    render,
    select,
)

pytestmark = pytest.mark.fast

CANDIDATE = Candidate(openml_name="credit-g", openml_task_id=31, amlb_list="small")


def entry(**overrides) -> DatasetEntry:
    """A minimal admissible entry. Every test below moves exactly one field out of bounds."""
    defaults = dict(
        dataset_id="d",
        openml_data_id=1,
        openml_task_id=1,
        openml_name="d",
        openml_version=1,
        openml_format="ARFF",
        openml_url="https://example.invalid/d.arff",
        md5_checksum="0" * 32,
        amlb_list="small",
        target="y",
        task_type="binary",
        metric="roc_auc",
        positive_class="1",
        n_rows=1000,
        n_features=10,
        n_usable_features=10,
        positive_rate=0.3,
        csv_sha256="0" * 64,
        leakage_labelled=False,
    )
    return DatasetEntry(**{**defaults, **overrides})


def measured(e: DatasetEntry | None, failed: str | None = None, name: str = "d") -> Measured:
    return Measured(replace(CANDIDATE, openml_name=name), e, {"n_rows": 1}, failed)


class TestSelectRejects:
    """Each bound, one at a time. A bound nobody has watched fire is a bound nobody has tested."""

    def test_a_dataset_inside_every_bound_is_kept(self) -> None:
        kept, excluded = select([measured(entry())])
        assert [k.dataset_id for k in kept] == ["d"]
        assert excluded == []

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({"n_rows": SELECTION_RULE.max_rows + 1, "n_features": 1}, "max_rows"),
            ({"n_features": SELECTION_RULE.max_features + 1, "n_rows": 10}, "max_features"),
            ({"n_rows": 100_000, "n_features": 100}, "max_cells"),
            (
                {"n_usable_features": SELECTION_RULE.min_usable_features - 1},
                "min_usable_features",
            ),
        ],
    )
    def test_each_bound_rejects_and_records_which_one(self, overrides, expected) -> None:
        kept, excluded = select([measured(entry(**overrides))])
        assert kept == []
        assert [e.failed for e in excluded] == [expected]

    def test_the_bounds_are_checked_in_a_stable_order(self) -> None:
        """A dataset over two bounds reports the first, not an arbitrary one, so `excluded` is
        reproducible across runs and `verify --online` can diff it."""
        kept, excluded = select(
            [measured(entry(n_rows=SELECTION_RULE.max_rows + 1, n_features=1000))]
        )
        assert kept == []
        assert excluded[0].failed == "max_rows"

    def test_an_unmeasurable_candidate_is_recorded_rather_than_dropped(self) -> None:
        """A fetch that failed must appear in `excluded`, or the manifest silently understates
        what was considered and the selection stops being auditable."""
        kept, excluded = select([measured(None, failed="fetch_failed")])
        assert kept == []
        assert excluded[0].failed == "fetch_failed"

    def test_an_unmeasurable_candidate_with_no_reason_still_gets_one(self) -> None:
        kept, excluded = select([measured(None, failed=None)])
        assert excluded[0].failed == "unmeasurable"


class TestSelectTrimsToTheTargetCount:
    def test_a_surplus_is_trimmed_smallest_first_and_recorded(self) -> None:
        """Over the cap, the rule keeps the cheapest by cells and records the rest as trimmed --
        it does not silently truncate, and it does not pick by name."""
        _, upper = SELECTION_RULE.target_count
        items = [
            measured(
                entry(dataset_id=f"d{i:02d}", n_rows=100 * (i + 1), n_features=10), name=f"d{i}"
            )
            for i in range(upper + 3)
        ]
        kept, excluded = select(items)
        assert len(kept) == upper
        trimmed = [e for e in excluded if e.failed == "target_count"]
        assert len(trimmed) == 3
        # The three largest went, not the three last.
        assert {e.openml_name for e in trimmed} == {f"d{i}" for i in (upper, upper + 1, upper + 2)}

    def test_kept_entries_come_back_sorted_by_dataset_id(self) -> None:
        items = [measured(entry(dataset_id=i)) for i in ("zeta", "alpha", "mu")]
        kept, _ = select(items)
        assert [k.dataset_id for k in kept] == ["alpha", "mu", "zeta"]


class TestUsableFeatures:
    """The criterion that decides whether a matrix reaches the modeler at all."""

    def test_a_high_cardinality_categorical_is_not_usable(self) -> None:
        frame = pd.DataFrame({"y": [0, 1] * 30, "cat": [f"v{i}" for i in range(60)]})
        assert _n_usable_features(frame, "y") == 0

    def test_a_low_cardinality_categorical_is_usable(self) -> None:
        frame = pd.DataFrame({"y": [0, 1] * 30, "cat": ["a", "b"] * 30})
        assert _n_usable_features(frame, "y") == 1

    def test_a_near_unique_identifier_is_dropped(self) -> None:
        frame = pd.DataFrame({"y": [0, 1] * 30, "id": range(60)})
        assert _n_usable_features(frame, "y") == 0

    def test_a_float_column_is_never_treated_as_an_identifier(self) -> None:
        """`feature_eng` exempts floats from the identifier rule; a continuous measurement is
        near-unique by nature and dropping it would gut every numeric dataset."""
        frame = pd.DataFrame({"y": [0, 1] * 30, "x": [i / 7 for i in range(60)]})
        assert _n_usable_features(frame, "y") == 1

    def test_the_target_is_never_counted(self) -> None:
        frame = pd.DataFrame({"y": [0, 1] * 30, "a": [1, 2] * 30, "b": [3, 4] * 30})
        assert _n_usable_features(frame, "y") == 2


class TestTheCsvIsWhatGetsMeasured:
    def test_a_boolean_target_round_trips_to_capitalised_strings(self, tmp_path) -> None:
        """The bug this rule exists for. `kc1`'s target arrives from OpenML as the category levels
        'true'/'false' and comes back out of a CSV as the bools True/False, so a `positive_class`
        measured before the write names a class no run could ever match.
        """
        frame = pd.DataFrame({"y": [True, False, False, False]})
        path = tmp_path / "d.csv"
        _write_csv(frame, path)
        reread = pd.read_csv(path)
        assert set(reread["y"].astype(str)) == {"True", "False"}

    def test_writing_is_deterministic(self, tmp_path) -> None:
        frame = pd.DataFrame({"a": [1.5, 2.5], "b": ["x", None]})
        first = _write_csv(frame, tmp_path / "a.csv")
        second = _write_csv(frame, tmp_path / "b.csv")
        assert first == second


class TestDatasetIds:
    @pytest.mark.parametrize(
        ("openml_name", "expected"),
        [
            ("credit-g", "credit_g"),
            ("blood-transfusion-service-center", "blood_transfusion_service_center"),
            ("KDDCup09_appetency", "kddcup09_appetency"),
            ("numerai28.6", "numerai28_6"),
            ("kr-vs-kp", "kr_vs_kp"),
        ],
    )
    def test_openml_names_become_stable_snake_case(self, openml_name, expected) -> None:
        assert _dataset_id(openml_name) == expected


class TestPublishedReferenceFailsLoudly:
    def test_a_fetch_failure_returns_none_and_says_so(self, monkeypatch, capsys) -> None:
        """A silent `None` is indistinguishable from "OpenML has no evaluations for this task",
        so a transient timeout would quietly drop a real citable score."""

        def boom(url: str) -> dict:
            raise TimeoutError("upstream said no")

        monkeypatch.setattr("ds_agents.benchmark_build._get_json", boom)
        assert published_reference(31) is None
        assert "no published reference" in capsys.readouterr().err

    def test_a_task_with_no_evaluations_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ds_agents.benchmark_build._get_json",
            lambda url: {"evaluations": {"evaluation": []}},
        )
        assert published_reference(31) is None

    def test_the_best_uploaded_score_wins_and_carries_its_run_id(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ds_agents.benchmark_build._get_json",
            lambda url: {
                "evaluations": {
                    "evaluation": [
                        {"run_id": "1", "flow_name": "weka.OneR(1)", "value": "0.52"},
                        {"run_id": "2", "flow_name": "weka.LMT(4)", "value": "0.79"},
                        {"run_id": "3", "flow_name": "broken", "value": ""},
                    ]
                }
            },
        )
        reference = published_reference(31)
        assert reference is not None
        assert reference.openml_run_id == 2
        assert reference.value == pytest.approx(0.79)
        assert "NOT this repo" in reference.protocol


class TestRenderedYamlIsSafe:
    def test_a_documented_leak_cannot_claim_to_be_planted_ground_truth(self) -> None:
        with pytest.raises(ValidationError):
            KnownLeak(
                column="c",
                kind="k",
                why="w",
                source="s",
                source_url="https://example.invalid",
                in_planted_leakage_columns=True,
            )

    def test_render_quotes_scalars_that_yaml_would_otherwise_coerce(self) -> None:
        from ds_agents.benchmark import load_manifest

        text = render(load_manifest())
        assert "'retrieved': '2026-" in text
        assert "GENERATED FILE" in text
