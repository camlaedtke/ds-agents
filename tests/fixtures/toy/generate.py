"""Generate the toy dataset. Seed is fixed; the CSV it writes is committed alongside it.

The dataset exists to be a trap with a known answer. `manifest.json` records what was planted so
tests and the eval can score the reviewer against ground truth rather than against a vibe.

Regenerate with: uv run python tests/fixtures/toy/generate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260822
N_ROWS = 200
HERE = Path(__file__).parent


def build() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)

    # A legitimately predictive signal: more support tickets -> more likely to churn.
    support_tickets_90d = rng.poisson(lam=1.8, size=N_ROWS)
    tenure_months = rng.integers(1, 72, size=N_ROWS)
    monthly_charges = np.round(rng.normal(70, 22, size=N_ROWS).clip(15, 160), 2)

    logit = -1.4 + 0.62 * support_tickets_90d - 0.03 * tenure_months
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churned = (rng.random(N_ROWS) < churn_prob).astype(int)

    # The planted leak: a status code assigned AFTER the churn decision. It disagrees with the
    # target ~6% of the time by construction, so it reads as a merely excellent feature rather than
    # an obvious copy. The realised agreement at n=200 is recorded in manifest.json; score against
    # that, never against the nominal rate.
    flip = rng.random(N_ROWS) < 0.06
    leaked = np.where(flip, 1 - churned, churned)
    account_status_code = np.where(leaked == 1, "CLOSED_R2", "ACTIVE_S1")

    # Pure noise, to give the profiler something to correctly ignore.
    region = rng.choice(["north", "south", "east", "west"], size=N_ROWS)
    plan_tier = rng.choice(["basic", "plus", "premium"], size=N_ROWS, p=[0.5, 0.35, 0.15])

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(N_ROWS)],
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "support_tickets_90d": support_tickets_90d,
            "region": region,
            "plan_tier": plan_tier,
            "account_status_code": account_status_code,
            "churned": churned,
        }
    )

    # Realistic missingness on one benign column only.
    missing_idx = rng.choice(N_ROWS, size=int(0.08 * N_ROWS), replace=False)
    frame.loc[missing_idx, "monthly_charges"] = np.nan

    agreement = float((leaked == churned).mean())
    manifest = {
        "seed": SEED,
        "n_rows": N_ROWS,
        "target": "churned",
        "task_type": "binary",
        "metric": "roc_auc",
        "planted_leakage": [
            {
                "column": "account_status_code",
                "kind": "noisy_copy_of_target",
                "agreement_with_target": round(agreement, 4),
                "why": (
                    "Status is assigned after the churn decision, "
                    "so it cannot exist at predict time."
                ),
            }
        ],
        "legit_strong_features": ["support_tickets_90d"],
        "noise_features": ["region", "plan_tier"],
        "id_columns": ["customer_id"],
        "positive_rate": round(float(churned.mean()), 4),
    }
    return frame, manifest


def main() -> None:
    frame, manifest = build()
    frame.to_csv(HERE / "toy.csv", index=False)
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(frame)} rows -> {HERE / 'toy.csv'}")
    print(
        f"positive rate {manifest['positive_rate']}, leak agreement "
        f"{manifest['planted_leakage'][0]['agreement_with_target']}"
    )


if __name__ == "__main__":
    main()
