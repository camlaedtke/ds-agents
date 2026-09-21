"""Build the walkthrough viewer. Run with: uv run python docs/explainers/build_walkthrough.py

Free and deterministic: reads the committed capture envelopes in docs/explainers/data/, derives
each run's step timeline through `ds_agents.capture.timeline` (so the replay logic stays in
src/ under fast tests, never in the viewer's JS), projects a compact slice of two committed
results files for the overview, and injects the whole payload into
pipeline-walkthrough.template.html at its one sentinel. Re-running the pipeline is never
required to re-render; running this twice must produce a byte-identical file.

Reads evals/results/ but never writes anywhere near it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ds_agents.capture import ARTIFACT_TEXT_LIMIT, SCHEMA_VERSION, Capture, timeline
from ds_agents.provenance import git_commit
from ds_agents.state import PipelineState

HERE = Path(__file__).parent
REPO = HERE.parent.parent
DATA_DIR = HERE / "data"
TEMPLATE = HERE / "pipeline-walkthrough.template.html"
OUTPUT = HERE / "pipeline-walkthrough.html"
SENTINEL = "/*__WALKTHROUGH_DATA__*/"

# Curated order and the one-line story each chip shows. A data file with no entry here is an
# error, not a silent inclusion: the story is part of the deliverable.
STORIES: dict[str, str] = {
    "toy": "one planted leak — removed after an objection, but never named as the leak",
    "claims_timing": "two planted leaks — both removed before the reviewer ever saw them",
    "reissued_ids": "the reviewer names the leak, and still runs the loop to exhaustion",
    "phoneme": "a clean benchmark — first-pass pass, and the harness agrees with the claim",
    "kc1": "the claim gap — the agents' own holdout number vs what the harness verified",
}

BENCHMARK_FILES = [
    REPO / "evals" / "results" / "2026-09-02_full.jsonl",
    REPO / "evals" / "results" / "2026-08-31_ci-baseline.jsonl",
]
BENCHMARK_COLUMNS = [
    "dataset_id",
    "baseline_normalised_score",
    "review_loops",
    "review_verdict",
    "claimed_holdout_score",
    "verified_holdout_score",
    "cost_usd",
    "wall_seconds",
    "objections_raised",
    "leakage_graded",
    "leakage_caught",
    "leakage_remediated",
]


def build_run(path: Path) -> dict[str, Any]:
    slug = path.stem
    if slug not in STORIES:
        raise SystemExit(
            f"{path.name}: no story in STORIES for slug {slug!r}; add one or delete the file"
        )
    cap = Capture.model_validate_json(path.read_text())
    state = PipelineState.from_dump(cap.state)
    steps = timeline(state)
    config = cap.state["config"]
    return {
        "slug": slug,
        "story": STORIES[slug],
        "note": cap.note,
        "dataset_id": cap.dataset_id,
        "source": config["dataset_source"],
        "config": {
            k: config[k]
            for k in (
                "run_id",
                "default_model",
                "reviewer_model",
                "loop_cap",
                "naming",
                "reviewer_enabled",
            )
        },
        "fixture_manifest": cap.fixture_manifest,
        "steps": [s.model_dump(mode="json") for s in steps],
        "results_row": cap.results_row,
        "node_seconds": cap.node_seconds,
        "artifacts": [a.model_dump(mode="json") for a in cap.artifacts],
        "objections": cap.state["objections"],
        "review_passes": cap.state["review_passes"],
    }


def benchmark_slice() -> dict[str, Any]:
    rows: list[list[Any]] = []
    seen_run_ids: set[str] = set()
    for path in BENCHMARK_FILES:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") in seen_run_ids:
                continue
            seen_run_ids.add(row.get("run_id", ""))
            rows.append([row.get(c) for c in BENCHMARK_COLUMNS])
    return {"columns": BENCHMARK_COLUMNS, "rows": rows}


def main() -> int:
    captures = sorted(DATA_DIR.glob("*.json"))
    if not captures:
        raise SystemExit(f"no capture envelopes in {DATA_DIR}; run capture_runs.py first")
    order = {slug: i for i, slug in enumerate(STORIES)}
    captures.sort(key=lambda p: order.get(p.stem, len(order)))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "built_from_commit": git_commit(),
        "artifact_text_limit": ARTIFACT_TEXT_LIMIT,
        "runs": [build_run(p) for p in captures],
        "benchmark": benchmark_slice(),
    }
    # `sort_keys` for byte-stable output; `<\/` so no artifact or evidence string can close the
    # script tag the payload lives in.
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).replace("</", "<\\/")

    template = TEMPLATE.read_text()
    if template.count(SENTINEL) != 1:
        raise SystemExit(
            f"template must contain exactly one {SENTINEL!r}, found {template.count(SENTINEL)}"
        )
    OUTPUT.write_text(template.replace(SENTINEL, blob))
    print(
        f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB, {len(payload['runs'])} runs, "
        f"{len(payload['benchmark']['rows'])} benchmark rows)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
