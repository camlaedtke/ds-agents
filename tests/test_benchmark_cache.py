"""Does the CSV on disk match what the manifest says about it?

Skipped when the cache is absent, which is the normal state of a fresh checkout -- `.cache/` is
gitignored, because the manifest pins each dataset by OpenML id and md5 and the CSVs are
reproducible from it. Populate with `uv run ds-agents datasets refresh`.

`csv_sha256` is deliberately NOT asserted here. It depends on pandas' float formatting and NaN
representation, so a pandas upgrade would turn it red without anything being wrong. It lives in
the manifest as a record of what was measured, and the shape assertions below are what is actually
kept green. A byte-identity claim that cannot be held is worse than no claim.
"""

import pandas as pd
import pytest

from ds_agents.benchmark import SELECTION_RULE, cached_csv_path, load_manifest
from ds_agents.nodes.feature_eng import ID_DISTINCTNESS_THRESHOLD, MAX_ONE_HOT_LEVELS

pytestmark = pytest.mark.fast

MANIFEST = load_manifest()
CACHED = [entry for entry in MANIFEST.datasets if cached_csv_path(entry).exists()]

pytestmark = [
    pytest.mark.fast,
    pytest.mark.skipif(not CACHED, reason="no fetched datasets; run `ds-agents datasets refresh`"),
]


@pytest.mark.parametrize("entry", CACHED, ids=lambda e: e.dataset_id)
def test_cached_csv_matches_the_recorded_shape(entry) -> None:
    frame = pd.read_csv(cached_csv_path(entry), low_memory=False)
    assert len(frame) == entry.n_rows
    assert frame.shape[1] - 1 == entry.n_features
    assert entry.target in frame.columns


@pytest.mark.parametrize("entry", CACHED, ids=lambda e: e.dataset_id)
def test_cached_csv_matches_the_recorded_class_balance(entry) -> None:
    """Two classes, and `positive_rate` counts the class the manifest names."""
    frame = pd.read_csv(cached_csv_path(entry), low_memory=False)
    labels = frame[entry.target]
    assert labels.nunique(dropna=True) == 2
    rate = float((labels.astype(str) == entry.positive_class).mean())
    assert rate == pytest.approx(entry.positive_rate, abs=1e-4)


@pytest.mark.parametrize("entry", CACHED, ids=lambda e: e.dataset_id)
def test_enough_columns_survive_the_mechanical_filters(entry) -> None:
    """The criterion with the sharpest teeth, re-derived from the node's own constants.

    This is `test_planted_columns_survive_the_mechanical_filters` transposed from "the trap
    reaches the reviewer" to "a matrix reaches the modeler". A dataset that is mostly
    high-cardinality categoricals arrives empty, the reviewer has nothing to object to, and the
    run records a `pass` with no model -- the open question docs/NEXT.md already carries. It is
    what excluded `Amazon_employee_access`, whose nine columns yield zero usable features.
    """
    frame = pd.read_csv(cached_csv_path(entry), low_memory=False)
    n_rows = len(frame)
    usable = 0
    for name in frame.columns:
        if name == entry.target:
            continue
        column = frame[name]
        n_unique = int(column.nunique(dropna=True))
        if (
            not pd.api.types.is_float_dtype(column)
            and n_rows > 0
            and n_unique >= ID_DISTINCTNESS_THRESHOLD * n_rows
        ):
            continue
        if not pd.api.types.is_numeric_dtype(column) and n_unique > MAX_ONE_HOT_LEVELS:
            continue
        usable += 1
    assert usable == entry.n_usable_features
    assert usable >= SELECTION_RULE.min_usable_features


@pytest.mark.parametrize(
    "entry", [e for e in CACHED if e.known_leakage], ids=lambda e: e.dataset_id
)
def test_a_documented_leak_names_a_column_that_exists(entry) -> None:
    """The check that caught a real error.

    `bank_marketing`'s leak was first recorded against `duration`, the name UCI uses -- but OpenML
    data 1461 ships anonymised headers (V1..V16) and no such column exists. A leak entry pointing
    at a phantom column reads as documented ground truth and can never fire, which is worse than
    no entry at all.
    """
    frame = pd.read_csv(cached_csv_path(entry), nrows=1, low_memory=False)
    for leak in entry.known_leakage:
        assert leak.column in frame.columns, f"{entry.dataset_id}: {leak.column}"
