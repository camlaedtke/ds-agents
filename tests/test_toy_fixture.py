"""The toy dataset is ground truth for the reviewer. If it drifts, every leakage number is wrong."""

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

FIXTURE = Path(__file__).parent / "fixtures" / "toy"


@pytest.fixture(scope="module")
def toy() -> pd.DataFrame:
    return pd.read_csv(FIXTURE / "toy.csv")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((FIXTURE / "manifest.json").read_text())


def test_shape_matches_manifest(toy, manifest):
    assert len(toy) == manifest["n_rows"] == 200


def test_target_is_binary_and_imbalanced_but_not_degenerate(toy, manifest):
    target = manifest["target"]
    assert set(toy[target].unique()) == {0, 1}
    assert 0.15 < toy[target].mean() < 0.45


def test_positive_rate_matches_manifest(toy, manifest):
    """The committed positive_rate is a number, not a comment. Nothing else checks it drifted."""
    assert toy[manifest["target"]].mean() == pytest.approx(manifest["positive_rate"], abs=1e-4)


def test_planted_leak_is_present_and_agrees_at_the_recorded_rate(toy, manifest):
    """The whole point of the fixture.

    A drifting leak silently changes how hard the reviewer's job is.
    """
    planted = manifest["planted_leakage"][0]
    column = planted["column"]
    assert column in toy.columns

    implied_positive = (toy[column] == "CLOSED_R2").astype(int)
    agreement = (implied_positive == toy[manifest["target"]]).mean()
    assert agreement == pytest.approx(planted["agreement_with_target"], abs=1e-4)


def test_leak_is_subtler_than_a_perfect_copy(toy, manifest):
    """A 100% copy would be a trivial catch and would not discriminate between reviewers."""
    agreement = manifest["planted_leakage"][0]["agreement_with_target"]
    assert 0.85 < agreement < 0.98


def test_legit_feature_is_actually_predictive(toy, manifest):
    """If the honest signal is dead, a model can only score well by taking the bait."""
    legit = manifest["legit_strong_features"][0]
    target = manifest["target"]
    churned_mean = toy.loc[toy[target] == 1, legit].mean()
    retained_mean = toy.loc[toy[target] == 0, legit].mean()
    assert churned_mean > retained_mean * 1.3


def test_noise_features_carry_no_real_signal(toy, manifest):
    """The profiler should ignore these.

    If they drift into real signal, the false_alarm metric becomes unreadable.
    """
    target = manifest["target"]
    for column in manifest["noise_features"]:
        rates = toy.groupby(column)[target].mean()
        assert rates.max() - rates.min() < 0.25


def test_missingness_is_confined_to_the_benign_column(toy):
    missing = toy.isna().sum()
    assert missing["monthly_charges"] > 0
    assert missing.drop("monthly_charges").sum() == 0


def test_generator_is_deterministic(toy):
    """Regenerating must reproduce the committed CSV, or the fixed seed is not doing its job."""
    import sys

    sys.path.insert(0, str(FIXTURE))
    from generate import build

    regenerated, _ = build()
    pd.testing.assert_frame_equal(regenerated, toy, check_dtype=False)
