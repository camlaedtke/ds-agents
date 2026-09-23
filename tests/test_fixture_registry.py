"""Every fixture is ground truth. If one drifts, every leakage number computed on it is wrong.

`tests/test_toy_fixture.py` keeps the assertions that are specific to the toy dataset's own leak.
What lives here is what must hold for any fixture, checked against every registered one, so that
adding a fixture cannot quietly add one that is unusable.

The mechanical-filter test is the one that would otherwise fail silently. A planted column that
`feature_eng` force-drops as an identifier, or that its snippet skips as high-cardinality, never
reaches the reviewer at all -- the run looks clean and the fixture measures nothing. That failure
produces no error anywhere; it just yields a leakage_recall of 0.0 that reads like a reviewer miss.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from ds_agents.fixtures import FIXTURES_ROOT, FixtureManifest, available, load_fixture
from ds_agents.nodes.feature_eng import ID_DISTINCTNESS_THRESHOLD, MAX_ONE_HOT_LEVELS

pytestmark = pytest.mark.fast

FIXTURE_NAMES = available()


def _frame(name: str) -> pd.DataFrame:
    return pd.read_csv(load_fixture(name).csv_path)


def test_at_least_the_three_expected_fixtures_are_registered():
    """A rename or a missing CSV would otherwise silently shrink every parametrized test below."""
    assert set(FIXTURE_NAMES) >= {"toy", "claims_timing", "reissued_ids"}


@pytest.mark.parametrize("name", FIXTURE_NAMES)
class TestEveryFixture:
    def test_manifest_validates(self, name):
        manifest = load_fixture(name).manifest
        assert isinstance(manifest, FixtureManifest)
        assert manifest.planted_leakage, "a fixture with nothing planted grades nothing"

    def test_declared_columns_all_exist(self, name):
        manifest = load_fixture(name).manifest
        columns = set(_frame(name).columns)
        declared = (
            [manifest.target]
            + manifest.planted_columns
            + manifest.legit_strong_features
            + manifest.noise_features
            + manifest.id_columns
        )
        assert set(declared) <= columns, f"manifest names columns not in the CSV: {name}"

    def test_shape_and_positive_rate_match_the_manifest(self, name):
        manifest = load_fixture(name).manifest
        frame = _frame(name)
        assert len(frame) == manifest.n_rows
        assert set(frame[manifest.target].unique()) == {0, 1}
        assert 0.15 < frame[manifest.target].mean() < 0.45
        assert frame[manifest.target].mean() == pytest.approx(manifest.positive_rate, abs=1e-4)

    def test_planted_columns_are_not_the_target_or_a_declared_id(self, name):
        """Overlap here would make the ground truth self-contradictory rather than merely wrong."""
        manifest = load_fixture(name).manifest
        planted = set(manifest.planted_columns)
        assert manifest.target not in planted
        assert planted.isdisjoint(manifest.id_columns)
        assert planted.isdisjoint(manifest.legit_strong_features)
        assert planted.isdisjoint(manifest.noise_features)

    def test_planted_columns_survive_the_mechanical_filters(self, name):
        """The trap has to reach the reviewer before the reviewer can be graded on it.

        Two filters run before any model opinion is involved: `feature_eng._forced_drops` removes
        a non-float column that is >=98% distinct, and its snippet keeps only numeric columns and
        categoricals with <=20 levels -- everything else lands in `skipped_high_cardinality` and
        is absent from `final_features`, which is the reviewer's whole view of the matrix.
        """
        manifest = load_fixture(name).manifest
        frame = _frame(name)
        for column in manifest.planted_columns:
            series = frame[column]
            n_unique = series.nunique(dropna=True)
            is_float = pd.api.types.is_float_dtype(series)
            assert is_float or n_unique < ID_DISTINCTNESS_THRESHOLD * len(frame), (
                f"{name}.{column} would be force-dropped as an identifier "
                f"({n_unique} distinct in {len(frame)} rows)"
            )
            numeric = pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
            assert numeric or n_unique <= MAX_ONE_HOT_LEVELS, (
                f"{name}.{column} is a {n_unique}-level non-numeric column, so the feature "
                f"snippet skips it and the reviewer never sees the name"
            )

    def test_noise_features_carry_no_real_signal(self, name):
        """If they drift into real signal, the false_alarm metric becomes unreadable."""
        manifest = load_fixture(name).manifest
        frame = _frame(name)
        for column in manifest.noise_features:
            rates = frame.groupby(column)[manifest.target].mean()
            assert rates.max() - rates.min() < 0.25

    def test_missingness_never_tracks_the_target(self, name):
        """Missingness is its own level to the profiler, so a label-correlated NaN pattern is an
        undeclared leak -- a trap the manifest does not know about and the metrics cannot score."""
        manifest = load_fixture(name).manifest
        frame = _frame(name)
        for column in frame.columns:
            if column == manifest.target or not frame[column].isna().any():
                continue
            rates = frame.groupby(frame[column].isna())[manifest.target].mean()
            assert rates.max() - rates.min() < 0.25, f"{name}.{column} leaks through missingness"

    def test_generator_is_deterministic(self, name):
        """Regenerating must reproduce the committed CSV and manifest, or the seed is decorative.

        Loaded by path under a unique module name: three fixtures each ship a `generate.py`, and
        a plain `sys.path` import would hand back whichever one landed in `sys.modules` first.
        """
        directory = load_fixture(name).csv_path.parent
        spec = importlib.util.spec_from_file_location(
            f"_fixture_generate_{name}", directory / "generate.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        regenerated, manifest = module.build()
        pd.testing.assert_frame_equal(regenerated, _frame(name), check_dtype=False)
        assert manifest == json.loads(Path(directory / "manifest.json").read_text())


@pytest.mark.parametrize("name", ["claims_timing", "reissued_ids"])
def test_the_new_traps_read_like_ordinary_features_not_like_copies(name):
    """The point of these two fixtures, as a number.

    The toy leak reads 0.518 normalized mutual information against 0.094 for the legitimate strong
    feature, which is why the profiler flags it every time and the reviewer is never shown a leaky
    matrix. These traps are tuned into the range a good feature occupies, so that flagging them
    takes reasoning about what the column means rather than reading the largest number.
    """
    manifest = load_fixture(name).manifest
    for leak in manifest.planted_leakage:
        assert leak.mutual_info_with_target is not None, "the evidence number must be recorded"
        assert 0.05 < leak.mutual_info_with_target < 0.25, (
            f"{name}.{leak.column} reads {leak.mutual_info_with_target}, outside the band where "
            f"the trap is a judgement call rather than an obvious flag"
        )


class TestResolutionFailures:
    """The paths a typo takes. Both used to be one hardcoded string in `cmd_run`."""

    def test_an_unknown_name_raises_with_the_known_fixtures_listed(self):
        with pytest.raises(SystemExit) as excinfo:
            load_fixture("no_such_fixture")
        message = str(excinfo.value)
        assert "no_such_fixture" in message
        for name in FIXTURE_NAMES:
            assert name in message, "the error has to say what the caller could have typed instead"

    def test_cmd_run_turns_an_unknown_name_into_exit_2_not_a_traceback(self, capsys):
        """A name that does not exist is a typo, not a pipeline failure. It must not read
        like one, which is what an uncaught SystemExit traceback would look like."""
        import argparse

        from ds_agents.cli import cmd_run

        args = argparse.Namespace(
            dataset="no_such_fixture",
            artifacts_dir=None,
            model="haiku",
            reviewer_model=None,
            tools="local",
            no_live=True,
        )
        assert cmd_run(args) == 2
        assert "no_such_fixture" in capsys.readouterr().err

    def test_toy_state_is_the_fixture_state_of_the_toy_fixture(self):
        """`_toy_state` is a wrapper the older tests still import. Pin the two together so the
        duplication stays deliberate rather than drifting into two definitions of a toy run."""
        from ds_agents.cli import _run_state
        from ds_agents.runnable import Runnable
        from tests.conftest import _toy_state

        wrapped = _toy_state("haiku")
        direct = _run_state(Runnable.from_fixture(load_fixture("toy")), model_name="haiku")
        assert wrapped.dataset_id == direct.dataset_id
        assert wrapped.task_description == direct.task_description
        assert wrapped.planted_leakage_columns == direct.planted_leakage_columns
        assert wrapped.config.default_model == direct.config.default_model


def test_a_directory_with_only_one_of_the_two_required_files_is_not_silently_invisible():
    """`available()` requires both `manifest.json` and `<name>.csv`. A directory holding one of
    them is a half-built fixture, and dropping it from the listing with no signal is how a typo'd
    filename becomes "the fixture does not exist" instead of "the fixture is misnamed"."""
    for entry in sorted(p for p in FIXTURES_ROOT.iterdir() if p.is_dir()):
        has_manifest = (entry / "manifest.json").exists()
        has_csv = (entry / f"{entry.name}.csv").exists()
        assert has_manifest == has_csv, (
            f"{entry.name} has one of manifest.json / {entry.name}.csv but not the other; "
            f"available() hides it entirely"
        )
