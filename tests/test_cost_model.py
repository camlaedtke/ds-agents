"""The cost model, re-derived from the committed results files rather than quoted.

`SUBSETS` carries a price for all thirteen manifest datasets. Nine are measured means; four are
modelled, because those datasets have never been run. Both kinds were prose before this module
existed: the coefficients, the residual scales and the four modelled prices lived in comments in
`harness.py` and in `docs/DECISIONS.md`, and a reader had no way to check them. This repo's whole
claim is that a number is traceable to a results file, so the fit is computed here, from the files,
every time the suite runs.

What that buys, beyond checkability: the residual scale is the thing that decides whether a NEW
measurement is a miss or is noise, and getting that wrong is how `docs/NEXT.md` came to describe a
+$0.0019 residual as "13% high" when the fit's own in-sample error was larger. See
`docs/LEARNING.md`, `a-fit-is-not-a-model`.

**Two fits appear here and they are not interchangeable.**

- `PRE_REGISTERED_*` is the `bench-mid` fit as it was actually pre-registered: OLS on the four
  `est_cost_usd` values that shipped in `SUBSETS["bench-mid"]`, which are per-cell means rounded UP
  to the nearest $0.001. It is reproduced exactly because it is what the 2026-09-02 arm was measured
  against, and rewriting it later against unrounded means would retroactively move the goalposts.
- The refit in `test_a_row_term_helps_and_a_categorical_term_does_not` uses the RAW per-cell means
  off the JSONL for all nine datasets. That is the honest fit and it is what `SUBSETS["full"]`
  prices unrun datasets from.

The two disagree in the third decimal and agree on everything that was concluded.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ds_agents.benchmark import cached_csv_path, load_manifest
from ds_agents.harness import SUBSETS
from ds_agents.nodes.feature_eng import MAX_ONE_HOT_LEVELS

pytestmark = pytest.mark.fast

RESULTS = Path(__file__).resolve().parents[1] / "evals" / "results"

# Where each measured price comes from. One dataset per file per cell; a dataset measured twice
# would need a rule about which run counts, and none exists because none has been.
SOURCES = {
    "2026-09-01_bench-mid.jsonl": ("phoneme", "jasmine", "amazon_employee_access", "nomao"),
    "2026-08-31_credit-g-smoke.jsonl": ("credit_g",),
    "2026-09-02_bench-tall.jsonl": ("adult", "bank_marketing", "numerai28_6", "higgs"),
}

# The `bench-mid` fit, on the four ROUNDED estimates it shipped. Pinned rather than recomputed
# because it is a historical artefact: the 2026-09-02 arm's band was derived from it.
PRE_REGISTERED_FIT = (0.010621, 0.00023375)
PRE_REGISTERED_RESIDUAL_SD = 0.0026

# Datasets whose measured cost came in ABOVE the pre-registered column-only prediction, and which
# were not among the four it was fitted on. All five of them, which is the finding.
OUT_OF_SAMPLE = ("credit_g", "adult", "bank_marketing", "numerai28_6", "higgs")


def measured_means() -> dict[str, float]:
    """Per-dataset mean `cost_usd`, read off the committed results files."""
    costs: dict[str, list[float]] = defaultdict(list)
    for filename, datasets in SOURCES.items():
        for line in (RESULTS / filename).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["dataset_id"] in datasets:
                costs[row["dataset_id"]].append(row["cost_usd"])
    return {name: float(np.mean(values)) for name, values in costs.items()}


def _fit(y: np.ndarray, *columns: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones(len(y)), *columns])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    dof = len(y) - design.shape[1]
    return coefficients, math.sqrt(float((residuals**2).sum()) / dof)


# The exact inputs the `bench-mid` fit was computed from on 2026-09-02: (n_features, est_cost_usd)
# as `SUBSETS["bench-mid"]` carried them THAT DAY. Held as literals rather than read off `SUBSETS`
# on purpose. `amazon_employee_access` shipped $0.013 then and ships $0.014 now (see the comment on
# the subset), and recomputing this fit from live values would silently move the band that the
# 2026-09-02 arm's residuals were judged against -- which is retroactively changing an experiment's
# analysis plan after seeing the data. The fit is history; it does not get to update.
PRE_REGISTERED_INPUTS = ((5, 0.011), (9, 0.013), (118, 0.041), (144, 0.042))


def test_the_pre_registered_fit_reproduces_from_the_estimates_it_was_computed_on():
    """The band the 2026-09-02 arm was judged against.

    Its residual sd is the whole point: at 4 points and 2 parameters the fit already misses its own
    training data by $0.0026, so a new residual smaller than that was never evidence of anything.
    """
    x = np.array([cols for cols, _ in PRE_REGISTERED_INPUTS], float)
    y = np.array([cost for _, cost in PRE_REGISTERED_INPUTS])

    (intercept, slope), residual_sd = _fit(y, x)

    assert (intercept, slope) == pytest.approx(PRE_REGISTERED_FIT, abs=5e-7)
    assert residual_sd == pytest.approx(PRE_REGISTERED_RESIDUAL_SD, abs=5e-5)


def test_the_column_only_model_under_predicted_every_out_of_sample_dataset():
    """The 2026-09-02 headline, as a property rather than a sentence.

    Five for five is a sign test at p=0.031. It is the only claim from that arm that does not rest
    on a residual being larger than the noise, which is why it is the one worth pinning.
    """
    intercept, slope = PRE_REGISTERED_FIT
    entries = {entry.dataset_id: entry for entry in load_manifest().datasets}
    means = measured_means()

    residuals = {
        name: means[name] - (intercept + slope * entries[name].n_features) for name in OUT_OF_SAMPLE
    }

    assert all(value > 0 for value in residuals.values()), (
        f"the column-only model no longer under-predicts every out-of-sample dataset: {residuals}. "
        "The sign test in docs/DECISIONS.md 2026-09-02 and evals/results/LOG.md is now wrong."
    )
    assert max(residuals, key=residuals.get) == "higgs"


def test_a_row_term_helps_and_a_categorical_term_does_not():
    """The secondary endpoint of the 2026-09-02 arm, recomputed rather than quoted.

    `H_categorical` was the pre-registered hypothesis with prior support and a real mechanism -- the
    reviewer reads a transform artifact whose `LEVELS` grow with one-hot count. It is refuted here:
    adding the term makes the fit WORSE. See docs/LEARNING.md,
    `confounded-by-what-you-did-not-vary`.
    """
    entries = {entry.dataset_id: entry for entry in load_manifest().datasets}
    means = measured_means()
    names = sorted(means)
    y = np.array([means[name] for name in names])
    columns = np.array([entries[name].n_features for name in names], float)
    rows = np.array([entries[name].n_rows for name in names], float) / 1e5
    categorical = np.array([CATEGORICAL_COLUMNS[name] for name in names], float)

    _, columns_only = _fit(y, columns)
    coefficients, with_rows = _fit(y, columns, rows)
    _, with_categorical = _fit(y, columns, categorical)
    _, with_both = _fit(y, columns, rows, categorical)

    assert len(names) == 9, "nine datasets have a measured cost; a tenth changes every number here"
    assert with_rows < columns_only, "a row term must improve the fit"
    assert with_categorical > columns_only, "a categorical term must make it worse -- H_categorical"
    assert with_both == pytest.approx(0.00206, abs=5e-5)
    assert (columns_only, with_rows, with_categorical) == pytest.approx(
        (0.00296, 0.00224, 0.00320), abs=5e-5
    )
    assert coefficients == pytest.approx([0.010163, 0.00023, 0.005609], abs=5e-6)


def test_every_measured_price_in_subsets_is_its_mean_rounded_up():
    """`est_cost_usd` is planning-only and never reaches a results row, so it may be conservative --
    but never optimistic, or a cap truncates a cell it was supposed to fund."""
    means = measured_means()
    for subset in ("bench-mid", "bench-smoke", "bench-tall", "full"):
        for cell in SUBSETS[subset]:
            if cell.dataset not in means:
                continue
            expected = math.ceil(means[cell.dataset] * 1000) / 1000
            assert cell.est_cost_usd == pytest.approx(expected, abs=1e-9), (
                f"{subset} cell {cell.name!r} carries {cell.est_cost_usd}, not the measured mean "
                f"{means[cell.dataset]:.5f} rounded up to {expected}"
            )


def test_the_four_never_run_datasets_are_priced_at_or_above_the_model():
    """`australian`, `kc1`, `sylvine` and `kr_vs_kp` have a price and no runs.

    Asserted as a floor rather than an equality: `kr_vs_kp` is deliberately rounded above the model,
    because it is 36 fully categorical columns and `H_categorical` was refuted at 7 to 13, which is
    not the same as refuted at 36. What must never happen is an estimate BELOW the model.
    """
    entries = {entry.dataset_id: entry for entry in load_manifest().datasets}
    means = measured_means()
    names = sorted(means)
    y = np.array([means[name] for name in names])
    coefficients, residual_sd = _fit(
        y,
        np.array([entries[name].n_features for name in names], float),
        np.array([entries[name].n_rows for name in names], float) / 1e5,
    )

    never_run = {cell.dataset for cell in SUBSETS["full"]} - set(means)
    assert never_run == {"australian", "kc1", "sylvine", "kr_vs_kp"}

    for cell in SUBSETS["full"]:
        if cell.dataset not in never_run:
            continue
        entry = entries[cell.dataset]
        predicted = (
            coefficients[0]
            + coefficients[1] * entry.n_features
            + coefficients[2] * entry.n_rows / 1e5
        )
        floor = math.ceil((predicted + residual_sd) * 1000) / 1000
        assert cell.est_cost_usd >= floor, (
            f"`full` cell {cell.name!r} is priced at {cell.est_cost_usd}, below the model's "
            f"{predicted:.5f} plus one residual sd rounded up ({floor}). The one thing the "
            "2026-09-02 arm established is that this model under-predicts out of sample."
        )


def test_the_full_subset_costs_what_the_docs_say_it_costs():
    """$1.10 is the number the next session has to put to a person. If it drifts, the docs lie."""
    per_pass = sum(cell.est_cost_usd for cell in SUBSETS["full"])
    assert per_pass == pytest.approx(0.276, abs=0.0005)
    assert per_pass * 4 == pytest.approx(1.104, abs=0.002), "--replicates 2 --n 2 is 4 runs a cell"


# What actually happened in `evals/results/2026-09-02_full.jsonl` (52 rows: all 13 manifest
# datasets, n=4 a cell, the arm that asked whether a loop-rate term belongs on the size model).
# These three constants are a RECORD of that arm, not a model of the next one -- the whole finding
# was that 6 loop events, clustered in 3 of 13 datasets, is not enough to fit a rate. See the
# contingency note on `SUBSETS["bench-tall"]` in `harness.py`, which is what this test backs.
FULL_2026_09_02_LOOP_RATE = 6 / 52  # 11.5%, Clopper-Pearson 95% CI [4.3%, 23.4%] at this n
FULL_2026_09_02_LOOP_MULTIPLIER_AT_2_LOOPS = 1.799  # n=2: one `australian` row, one `sylvine` row
FULL_2026_09_02_LOOP_MULTIPLIER_AT_3_LOOPS = (
    2.458  # n=4: two `australian` rows, two `kr_vs_kp` rows
)


def test_the_full_arm_loop_multipliers_reproduce_from_the_results_file():
    """The two numbers `SUBSETS["bench-tall"]`'s contingency comment cites, recomputed rather than
    quoted.

    The multiplier is per-row, against that ROW's OWN dataset's 1-loop mean, then averaged --
    not dollars pooled across datasets first -- because `australian` and `kr_vs_kp` do not cost
    the same to begin with and pooling would let the cheaper dataset's ratio get outweighed by the
    pricier one's dollars rather than counted once like every other row.
    """
    costs_by_dataset_and_loops: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    loop_counts: list[int] = []
    for line in (RESULTS / "2026-09-02_full.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        costs_by_dataset_and_loops[row["dataset_id"]][row["review_loops"]].append(row["cost_usd"])
        loop_counts.append(row["review_loops"])

    assert len(loop_counts) == 52, (
        "13 datasets x n=4; a different row count changes every number here"
    )

    one_loop_mean = {
        dataset: sum(costs[1]) / len(costs[1])
        for dataset, costs in costs_by_dataset_and_loops.items()
        if 1 in costs
    }

    def multiplier_at(loop_count: int) -> float:
        ratios = [
            cost / one_loop_mean[dataset]
            for dataset, costs in costs_by_dataset_and_loops.items()
            for cost in costs.get(loop_count, [])
        ]
        return sum(ratios) / len(ratios)

    assert multiplier_at(2) == pytest.approx(FULL_2026_09_02_LOOP_MULTIPLIER_AT_2_LOOPS, abs=5e-4)
    assert multiplier_at(3) == pytest.approx(FULL_2026_09_02_LOOP_MULTIPLIER_AT_3_LOOPS, abs=5e-4)

    looped = sum(1 for loops in loop_counts if loops > 1)
    assert looped == 6
    assert looped / len(loop_counts) == pytest.approx(FULL_2026_09_02_LOOP_RATE, abs=1e-9)


# Categorical column counts, by the rule `feature_eng` actually applies: a column is one-hot
# encoded when it is NOT numeric or bool dtype and has at most MAX_ONE_HOT_LEVELS distinct values.
# Held as literals so the fit above runs without the gitignored CSV cache, and checked against the
# CSVs by the test below whenever that cache is present.
CATEGORICAL_COLUMNS = {
    "adult": 7,
    "amazon_employee_access": 0,
    "australian": 0,
    "bank_marketing": 9,
    "credit_g": 13,
    "higgs": 0,
    "jasmine": 0,
    "kc1": 0,
    "kr_vs_kp": 36,
    "nomao": 0,
    "numerai28_6": 0,
    "phoneme": 0,
    "sylvine": 0,
}


def test_the_categorical_counts_match_the_datasets_they_claim_to_describe():
    """The refutation of `H_categorical` rests on these counts, so they are not taken on trust.

    They are also the reason the hypothesis existed: every one of the four datasets the cost model
    was fitted on scores ZERO here, and nobody chose that -- `bench-mid` crossed rows with columns,
    and dtype was not one of the two axes.
    """
    checked = 0
    for entry in load_manifest().datasets:
        path = cached_csv_path(entry)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        count = sum(
            1
            for name in frame.columns
            if name != entry.target
            and not (
                pd.api.types.is_numeric_dtype(frame[name])
                or pd.api.types.is_bool_dtype(frame[name])
            )
            and frame[name].nunique(dropna=True) <= MAX_ONE_HOT_LEVELS
        )
        assert count == CATEGORICAL_COLUMNS[entry.dataset_id], (
            f"{entry.dataset_id} has {count} one-hot-encodable columns, not "
            f"{CATEGORICAL_COLUMNS[entry.dataset_id]}"
        )
        checked += 1

    if not checked:
        pytest.skip("no benchmark CSVs cached; run `uv run ds-agents datasets refresh`")

    fitted = {cell.dataset for cell in SUBSETS["bench-mid"]}
    assert all(CATEGORICAL_COLUMNS[name] == 0 for name in fitted), (
        "the confound this module records is that every dataset the cost model was fitted on has "
        "no categorical columns; if that stops being true, docs/LEARNING.md needs revisiting"
    )
