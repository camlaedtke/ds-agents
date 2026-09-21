"""Capture the walkthrough's example runs. Run with: uv run python docs/explainers/capture_runs.py

Shells out to `ds-agents run --state-json` for each curated example, then prints a one-line
summary per captured replicate so the operator can pick the replicate that actually shows the
story and delete the rest. Model nondeterminism is the whole variance (the seed fixes only the
data split), so an interesting run is found by repeating and choosing -- and the choice is
recorded on the envelope's `note` field, which the operator fills in on the kept file.

This script never touches evals/. Costs real money (~$0.25 for the default set, all haiku).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# slug, dataset, replicates, what the kept replicate should show
EXAMPLES: list[tuple[str, str, int, str]] = [
    ("toy", "toy", 2, "one planted leak; ideally caught=False but remediated=True"),
    ("claims_timing", "claims_timing", 2, "two planted leaks; loops back upstream, 2-3 passes"),
    ("reissued_ids", "reissued_ids", 3, "id-column trap; ideally reaches verdict=exhausted"),
    ("phoneme", "phoneme", 1, "clean benchmark: first-pass pass, normalised score ~0.99"),
    ("kc1", "kc1", 1, "the claim gap: claimed ~0.81 vs verified ~0.77"),
]


def summarize(path: Path) -> str:
    cap = json.loads(path.read_text())
    row = cap["results_row"]
    caught = row.get("leakage_caught")
    remediated = row.get("leakage_remediated")
    leak = (
        f"caught={caught} remediated={remediated}"
        if row.get("leakage_graded")
        else (
            f"claimed={row.get('claimed_holdout_score')} "
            f"verified={row.get('verified_holdout_score')}"
        )
    )
    return (
        f"{path.name:38s} verdict={row.get('review_verdict'):<9s} "
        f"loops={row.get('review_loops')} objections={row.get('objections_raised')} "
        f"{leak}  ${row.get('cost_usd'):.4f}"
    )


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    for slug, dataset, repeat, story in EXAMPLES:
        target = DATA_DIR / f"{slug}.json"
        print(f"\n=== {slug}: {story}", file=sys.stderr)
        cmd = [
            "uv",
            "run",
            "ds-agents",
            "run",
            "--dataset",
            dataset,
            "--repeat",
            str(repeat),
            "--artifacts-dir",
            tempfile.mkdtemp(prefix=f"walkthrough-{slug}-"),
            "--state-json",
            str(target),
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if result.returncode != 0:
            print(
                f"{slug}: run exited {result.returncode}; captures may be partial", file=sys.stderr
            )

    print(
        "\ncaptured replicates -- keep one per slug (rename to <slug>.json), fill in its "
        "`note` with which replicate out of how many and why, delete the rest:\n"
    )
    for path in sorted(DATA_DIR.glob("*.json")):
        try:
            print("  " + summarize(path))
        except (KeyError, json.JSONDecodeError) as exc:
            print(f"  {path.name}: unreadable ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
