"""Generate the reissued_ids dataset: an identifier that really does encode the answer.

This fixture is aimed at a specific sentence in the profiler's own prompt, which tells the model
that an identifier "shows elevated association purely because it memorises the rows, and predicts
nothing" (src/ds_agents/nodes/profiler.py). That rule is right on the toy fixture, where
`customer_id` is a decoy. Here it is false, and following it costs the run.

Two id-shaped columns carry the contrast:

  application_ref  a genuine identifier: unique per row, no signal. `feature_eng` force-drops it as
                   an identifier before any model sees it. This is the control -- it confirms the
                   id rule still fires while the trap slips past it.
  member_number    the trap. After the book was labelled, defaulted files were migrated into a
                   700000 block and the rest left in the 300000 block, so the number carries the
                   outcome. Members reappear across applications, which puts distinctness under the
                   0.98 threshold `feature_eng` drops on, and the migration is noisy enough that
                   the association reads as a merely-useful column rather than as a copy.

Regenerate with: uv run python tests/fixtures/reissued_ids/generate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260827
N_ROWS = 200
N_REPEAT_MEMBERS = 12
HERE = Path(__file__).parent


def build() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)

    # Legitimate signal, all of it knowable at application time.
    credit_score = rng.integers(520, 830, size=N_ROWS)
    debt_to_income = np.round(rng.beta(2.2, 5.0, size=N_ROWS) * 0.9, 4)
    months_employed = rng.integers(0, 240, size=N_ROWS)

    z_score = (credit_score - credit_score.mean()) / credit_score.std()
    z_dti = (debt_to_income - debt_to_income.mean()) / debt_to_income.std()
    logit = -0.80 - 0.85 * z_score + 0.55 * z_dti - 0.004 * months_employed
    defaulted = (rng.random(N_ROWS) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    # The trap. Migration to the 700000 block happened after labelling, but it is not a clean
    # partition: a third of defaults were never migrated and a quarter of the performing book was
    # migrated for unrelated reasons. That noise is what keeps the association in the range a
    # legitimate strong feature occupies. See manifest.json for the realised number.
    migrate_p = np.where(defaulted == 1, 0.80, 0.20)
    migrated = rng.random(N_ROWS) < migrate_p
    block = np.where(migrated, 700_000, 300_000)
    member_number = block + rng.integers(0, 90_000, size=N_ROWS)

    # Members reappear across applications. This is what real loan books look like, and it is also
    # what puts distinctness below the 0.98 identifier threshold -- above it, `feature_eng` drops
    # the column before any model can lean on it and the trap never reaches the reviewer. A repeat
    # is drawn from a row with the same block, so it does not smuggle in a second, undeclared leak.
    order = rng.permutation(N_ROWS)
    for row in order[:N_REPEAT_MEMBERS]:
        same_block = [i for i in range(N_ROWS) if i != row and migrated[i] == migrated[row]]
        member_number[row] = member_number[rng.choice(same_block)]

    branch = rng.choice(["BR01", "BR02", "BR03", "BR04", "BR05"], size=N_ROWS)
    loan_purpose = rng.choice(
        ["auto", "home_improvement", "debt_consolidation", "other"],
        size=N_ROWS,
        p=[0.3, 0.25, 0.35, 0.1],
    )

    frame = pd.DataFrame(
        {
            "application_ref": [f"APP-2026-{i:05d}" for i in range(N_ROWS)],
            "member_number": member_number,
            "credit_score": credit_score,
            "debt_to_income": debt_to_income,
            "months_employed": months_employed,
            "branch": branch,
            "loan_purpose": loan_purpose,
            "default_90d": defaulted,
        }
    )

    # Missingness on one benign column, drawn independently of the target.
    missing_idx = rng.choice(N_ROWS, size=int(0.06 * N_ROWS), replace=False)
    frame.loc[missing_idx, "months_employed"] = np.nan

    n_unique = int(frame["member_number"].nunique())
    manifest = {
        "seed": SEED,
        "n_rows": N_ROWS,
        "target": "default_90d",
        "task_type": "binary",
        "metric": "roc_auc",
        "planted_leakage": [
            {
                "column": "member_number",
                "kind": "identifier_reissued_by_outcome",
                "why": (
                    "Files were re-keyed into a 700000 block after the outcome was known, so the "
                    f"number range carries the label. {n_unique} distinct values in {N_ROWS} rows, "
                    "which is under the identifier threshold feature_eng drops on."
                ),
                "agreement_with_target": round(
                    float(((member_number >= 700_000).astype(int) == defaulted).mean()), 4
                ),
                "mutual_info_with_target": _nmi(frame, "member_number", "default_90d"),
            }
        ],
        "legit_strong_features": ["credit_score", "debt_to_income"],
        "noise_features": ["branch", "loan_purpose"],
        "id_columns": ["application_ref"],
        "positive_rate": round(float(defaulted.mean()), 4),
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
    frame.to_csv(HERE / "reissued_ids.csv", index=False)
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(frame)} rows -> {HERE / 'reissued_ids.csv'}")
    print(f"positive rate {manifest['positive_rate']}")
    leak = manifest["planted_leakage"][0]
    print(
        f"  {leak['column']:<16} nmi {leak['mutual_info_with_target']}  "
        f"agreement {leak['agreement_with_target']}  "
        f"distinct {frame['member_number'].nunique()}/{N_ROWS}"
    )


if __name__ == "__main__":
    main()
