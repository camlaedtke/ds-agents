"""Generate the claims_timing dataset: a leak recorded after the label, not copied from it.

The toy fixture's leak is a noisy copy of the target, which the profiler flags every time -- so the
reviewer has never once been shown a leaky matrix (docs/NEXT.md, 2026-08-27). This fixture is built
to survive the profiler rather than to be caught by it.

The trap is `days_to_close` and `adjuster_touches`. Both are recorded when the claim file is closed,
which happens after the denial decision, so neither could exist at prediction time. Neither is a
copy of the answer: each overlaps heavily across the two classes, so each reads as an ordinary
moderately-predictive number. What makes them a leak is what they MEAN, not what they measure, and
the only place that meaning appears is the column name.

Two trap columns rather than one, on purpose: `leakage_recall` can then take the value 0.5, so a
reviewer that catches one and misses the other is distinguishable from one that catches both.

Regenerate with: uv run python tests/fixtures/claims_timing/generate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260827
N_ROWS = 200
HERE = Path(__file__).parent


def build() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)

    # Legitimate signal, all of it knowable when the claim is filed.
    prior_claims_12m = rng.poisson(lam=0.9, size=N_ROWS)
    claim_amount = np.round(np.exp(rng.normal(7.4, 0.75, size=N_ROWS)), 2)
    days_since_policy_start = rng.integers(30, 2000, size=N_ROWS)

    z_amount = (np.log(claim_amount) - np.log(claim_amount).mean()) / np.log(claim_amount).std()
    z_tenure = (days_since_policy_start - days_since_policy_start.mean()) / (
        days_since_policy_start.std()
    )
    logit = -1.95 + 0.95 * prior_claims_12m + 0.60 * z_amount - 0.40 * z_tenure
    denied = (rng.random(N_ROWS) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    # The trap. Both are stamped on the file at closure, after `denied` is known. A denied claim is
    # investigated longer and touched more often, but the distributions overlap hard on purpose --
    # tuned so each column's mutual information with the target lands near a legitimate strong
    # feature rather than near a copy. See manifest.json for the realised numbers; score against
    # those, never against the intent.
    days_to_close = np.round(np.exp(rng.normal(2.35 + 0.80 * denied, 0.60, size=N_ROWS)), 1)
    adjuster_touches = rng.poisson(lam=1.6 + 2.2 * denied, size=N_ROWS)

    # Pure noise, so the profiler has something to correctly ignore.
    channel = rng.choice(["agent", "web", "phone"], size=N_ROWS, p=[0.4, 0.35, 0.25])
    region = rng.choice(["north", "south", "east", "west"], size=N_ROWS)

    frame = pd.DataFrame(
        {
            "claim_ref": [f"CLM-{i:05d}" for i in range(N_ROWS)],
            "days_since_policy_start": days_since_policy_start,
            "claim_amount": claim_amount,
            "prior_claims_12m": prior_claims_12m,
            "channel": channel,
            "region": region,
            "days_to_close": days_to_close,
            "adjuster_touches": adjuster_touches,
            "claim_denied": denied,
        }
    )

    # Missingness on one benign column, drawn independently of the target. A missing-rate that
    # tracked the label would be a third leak the manifest does not declare, and the profiler bins
    # missingness as its own level, so it would find it.
    missing_idx = rng.choice(N_ROWS, size=int(0.07 * N_ROWS), replace=False)
    frame.loc[missing_idx, "claim_amount"] = np.nan

    manifest = {
        "seed": SEED,
        "n_rows": N_ROWS,
        "target": "claim_denied",
        "task_type": "binary",
        "metric": "roc_auc",
        "planted_leakage": [
            {
                "column": "days_to_close",
                "kind": "recorded_after_label",
                "why": (
                    "Stamped when the claim file is closed, which happens after the denial "
                    "decision. It cannot exist at predict time."
                ),
                "mutual_info_with_target": _nmi(frame, "days_to_close", "claim_denied"),
            },
            {
                "column": "adjuster_touches",
                "kind": "recorded_after_label",
                "why": (
                    "Counted over the life of the claim file, so it is complete only once the "
                    "claim is closed and the decision has already been made."
                ),
                "mutual_info_with_target": _nmi(frame, "adjuster_touches", "claim_denied"),
            },
        ],
        "legit_strong_features": ["prior_claims_12m", "claim_amount"],
        "noise_features": ["channel", "region"],
        "id_columns": ["claim_ref"],
        "positive_rate": round(float(denied.mean()), 4),
    }
    return frame, manifest


def _nmi(frame: pd.DataFrame, column: str, target: str) -> float:
    """The number the profiler will actually hand its model, computed the same way it computes it.

    Recorded in the manifest because it is the entire evidence base for the flag-or-not decision
    this fixture exists to test. If a regenerated fixture moves it, the trap changed difficulty.
    """
    from sklearn.metrics import normalized_mutual_info_score

    return round(float(normalized_mutual_info_score(_bin(frame[target]), _bin(frame[column]))), 4)


def _bin(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 12:
        return pd.qcut(series.rank(method="first"), 10, labels=False, duplicates="drop").fillna(-1)
    return series.astype("string").fillna("<MISSING>")


def main() -> None:
    frame, manifest = build()
    frame.to_csv(HERE / "claims_timing.csv", index=False)
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(frame)} rows -> {HERE / 'claims_timing.csv'}")
    print(f"positive rate {manifest['positive_rate']}")
    for leak in manifest["planted_leakage"]:
        print(f"  {leak['column']:<20} nmi {leak['mutual_info_with_target']}")


if __name__ == "__main__":
    main()
