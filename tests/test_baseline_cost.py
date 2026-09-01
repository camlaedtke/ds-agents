"""What the unit point costs in wall time, at the shapes it is actually fitted on.

Opt in:

    DS_AGENTS_TIMING_TESTS=1 uv run pytest tests/test_baseline_cost.py -s

This replaces a table that existed only as prose. On 2026-08-31 the unit point was timed at three
shapes -- 0.20s at 1000x20, 9.85s at 48842x14, 72.09s at 98050x28 -- and the commit that recorded
it (`ef15609`) touched no code, so the numbers could not be re-derived, checked, or extended. They
are quoted in `docs/NEXT.md`, `docs/DECISIONS.md` and `evals/results/LOG.md` and `--subset full` is
being priced from them.

They were also measured at the WRONG SHAPE. The baseline fits on `SPLIT["train"]`
(`rescore._BASELINE_BODY`: `train_frame = df.iloc[train_rows]`), which is the profiler's 80% train
split of the agent frame, which is itself the 80% left after `WITHHELD_FRACTION` is carved off
before the graph starts. So the fit sees **0.64 x n_rows**, not `n_rows`, and the old table
overstates every dataset by roughly 1/0.64. Wrong in the safe direction, still wrong.

Gated by an environment variable rather than a marker, following
`tests/test_benchmark_provenance.py` and the reason recorded in `docs/NEXT.md`'s parking lot:
`network` tests are gated on `DS_AGENTS_NETWORK_TESTS=1` and NOT on `-m "not network"`, so that the
default `uv run pytest` stays both clean and complete. Same here.

**These are an upper bound, not a prediction.** The frames below are Gaussian noise with one weak
signal column, which is close to worst case for a decision tree: with little structure to split on,
trees grow deeper and fitting costs more than it does on real data. Measured 2026-09-01 on the
correct shapes, this table disagrees with the 2026-08-31 prose one in BOTH directions -- higgs came
out at 57.18s against a quoted 72.09s (the old figure used the full frame, 1.56x too many rows),
while adult came out at 14.94s against a quoted 9.85s despite fitting on 36% fewer rows. The
adult direction is the one that shows what synthetic noise costs. So read a row here as "the fit
will not take longer than this", never as "the fit will take this".

The headline it does support is robust to all of that: the slowest shape in the manifest uses about
6% of the timeout, so `n_estimators` does not need to drop for the large frames and
`baseline_recipe` does not need to fork.

The assertion is deliberately machine-independent. An absolute-seconds bound would flake on
different hardware and would have to be retuned rather than trusted; what is asserted instead is a
real property -- that every fit clears `BASELINE_TIMEOUT_S` with stated headroom.
"""

import os
import time

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ds_agents.benchmark import load_manifest
from ds_agents.nodes.profiler import HOLDOUT_FRACTION
from ds_agents.rescore import _RF, BASELINE_RECIPE, BASELINE_TIMEOUT_S
from ds_agents.runnable import WITHHELD_FRACTION
from ds_agents.state import DEFAULT_RANDOM_SEED

pytestmark = pytest.mark.skipif(
    os.environ.get("DS_AGENTS_TIMING_TESTS") != "1",
    reason="fits a 200-tree forest at every manifest shape; set DS_AGENTS_TIMING_TESTS=1",
)

# How much of the timeout a fit may consume before this test calls it a problem. The unit point is
# a yardstick, not a competitor: if it needs a fifth of its own budget, `n_estimators` is the lever
# and `baseline_recipe` is what makes pulling it visible in the data.
HEADROOM = 5.0


def fitted_rows(n_rows: int) -> int:
    """Rows the baseline actually fits on: the train split of the agent frame."""
    agent = n_rows - int(round(n_rows * WITHHELD_FRACTION))
    return agent - int(round(agent * HOLDOUT_FRACTION))


def _synthetic(n_rows: int, n_features: int, seed: int = DEFAULT_RANDOM_SEED):
    """A frame of the right SHAPE. Signal is irrelevant to fit cost; shape is not.

    All-numeric, so the ColumnTransformer would be a no-op imputation -- excluded deliberately,
    because it is linear in cells and the forest is not, and mixing them would make the table
    describe something other than the term that dominates.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features))
    y = (X[:, 0] + rng.standard_normal(n_rows) * 0.5 > 0).astype(int)
    return X, y


def test_the_unit_point_clears_its_timeout_at_every_manifest_shape():
    entries = sorted(load_manifest().datasets, key=lambda e: e.n_rows * e.n_features)
    params = {k: (DEFAULT_RANDOM_SEED if v == "__SEED__" else v) for k, v in _RF.items()}

    print(f"\nunit point `{BASELINE_RECIPE}`: {params}")
    print(f"fitted on 0.64 x n_rows; timeout {BASELINE_TIMEOUT_S}s, headroom {HEADROOM}x\n")
    header = f"{'dataset':24s}{'n_rows':>8s}{'cols':>6s}{'fit rows':>10s}{'seconds':>10s}"
    print(header + f"{'% budget':>10s}")

    slowest = 0.0
    for entry in entries:
        rows = fitted_rows(entry.n_rows)
        X, y = _synthetic(rows, entry.n_features)
        started = time.perf_counter()
        RandomForestClassifier(**params).fit(X, y)
        elapsed = time.perf_counter() - started
        slowest = max(slowest, elapsed)
        share = 100 * elapsed / BASELINE_TIMEOUT_S
        print(
            f"{entry.dataset_id:24s}{entry.n_rows:8d}{entry.n_features:6d}"
            f"{rows:10d}{elapsed:10.2f}{share:9.1f}%"
        )
        assert elapsed * HEADROOM < BASELINE_TIMEOUT_S, (
            f"{entry.dataset_id} fits in {elapsed:.1f}s against a {BASELINE_TIMEOUT_S}s timeout, "
            f"inside the {HEADROOM}x headroom this test requires. Drop n_estimators for large "
            "frames and BUMP BASELINE_RECIPE, or raise the timeout deliberately."
        )

    print(f"\nslowest: {slowest:.2f}s of a {BASELINE_TIMEOUT_S}s budget\n")
