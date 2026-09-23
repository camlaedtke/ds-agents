"""Re-derive every number in docs/RESULTS.md from the committed results files.

Run from the repo root: `uv run python docs/readme_numbers.py`

Reads only `evals/results/*.jsonl`. Nothing here is a measurement -- it is an aggregation of
measurements already committed, kept beside RESULTS.md so no number in it has to be taken on
trust. Promote it to `src/ds_agents/` if RESULTS.md needs regenerating on a schedule.
"""

import collections
import glob
import json
import statistics

FULL = "evals/results/2026-09-02_full.jsonl"
ALL_FILES = sorted(glob.glob("evals/results/*.jsonl"))


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- provenance
section("PROVENANCE: schema drift across the committed files")
full_keys = set(load(FULL)[0])
print(f"current row: {len(full_keys)} columns\n")
for path in ALL_FILES:
    rows = load(path)
    keys = set().union(*(set(r) for r in rows))
    planted = sum(1 for r in rows if r.get("leakage_planted"))
    graded = sum(1 for r in rows if r.get("leakage_graded") is True)
    print(
        f"{path.split('/')[-1]:40s} {len(rows):3d} rows {len(keys):3d} cols "
        f"missing {len(full_keys - keys):2d} extra {sorted(keys - full_keys)} "
        f"| planted {planted:3d} graded_true {graded:3d}"
    )
FIXTURE_FILES = [p for p in ALL_FILES if any(r.get("leakage_planted") for r in load(p))]
fixture_rows = sum(len(load(p)) for p in FIXTURE_FILES)
graded_anywhere = sum(1 for p in ALL_FILES for r in load(p) if r.get("leakage_graded") is True)
print(f"\nfixture-era rows: {fixture_rows} across {len(FIXTURE_FILES)} files")
print(f"rows anywhere with leakage_graded true: {graded_anywhere}")

# ---------------------------------------------------------------------- the benchmark run
section("RESULT 1: the benchmark run")
rows = load(FULL)
cost = sum(r["cost_usd"] for r in rows)
wall = sum(r["wall_seconds"] for r in rows) / 60
print(f"{len(rows)} rows | ${cost:.4f} | {wall:.1f} min")
print(f"commits {set(r['commit'] for r in rows)}")
print(f"halted_at non-null: {sum(1 for r in rows if r['halted_at'])}")
print(
    "config "
    f"{set((r['naming'], r['reviewer_prompt'], r['objection_routing'], r['objection_closure'], r['loop_cap'], r['default_model'], r['reviewer_model']) for r in rows)}"  # noqa: E501
)

by_dataset: dict[str, list[dict]] = collections.defaultdict(list)
for r in rows:
    by_dataset[r["dataset_id"]].append(r)
header = (
    f"\n{'dataset':24s} {'verified':>8} {'unit':>7} {'sep':>6} {'norm':>6} "
    f"{'nfeat':>6} {'$/run':>7} {'loops':>18} {'obj>0':>6}"
)
print(header)
for name, runs in sorted(by_dataset.items(), key=lambda kv: -mean(r["cost_usd"] for r in kv[1])):
    loops = dict(sorted(collections.Counter(r["review_loops"] for r in runs).items()))
    print(
        f"{name:24s} {mean(r['verified_holdout_score'] for r in runs):8.4f} "
        f"{mean(r['baseline_unit_score'] for r in runs):7.4f} "
        f"{mean(r['baseline_separation'] for r in runs):6.3f} "
        f"{mean(r['baseline_normalised_score'] for r in runs):6.3f} "
        f"{mean(r['n_final_features'] for r in runs):6.1f} "
        f"{mean(r['cost_usd'] for r in runs):7.4f} {str(loops):>18} "
        f"{sum(1 for r in runs if r['objections_raised'] > 0):2d}/{len(runs)}"
    )

gaps = [r["holdout_claim_gap"] for r in rows]
wide = sum(1 for g in gaps if abs(g) > 0.02)
print(
    f"\nclaim gap: mean {mean(gaps):+.4f} median {statistics.median(gaps):+.4f} "
    f"max {max(gaps):+.4f} min {min(gaps):+.4f} |gap|>0.02 {wide}/{len(gaps)}"
)

node: dict[str, list[float]] = collections.defaultdict(list)
for r in rows:
    for name, secs in (r["node_seconds"] or {}).items():
        node[name].append(secs)
total = sum(mean(v) for v in node.values())
print()
for name, secs in sorted(node.items(), key=lambda kv: -mean(kv[1])):
    print(f"{name:14s} {mean(secs):6.2f}s {100 * mean(secs) / total:5.1f}%")

# ------------------------------------------------------------------------- the fixture arms
section("RESULTS 2-3: the fixture arms")
print(
    f"{'file':30s} {'ds':14s} {'naming':11s} {'prompt/model':20s} {'route/cap':18s} "
    f"{'n':>3} {'prof_rec':>8} {'rev_caught':>11} {'remed':>7} {'claimed':>8} {'$/run':>7}"
)
for path in FIXTURE_FILES:
    arms: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in load(path):
        arms[
            (
                r.get("dataset_id"),
                r.get("naming"),
                r.get("reviewer_prompt"),
                r.get("objection_routing"),
                r.get("reviewer_model"),
                r.get("loop_cap"),
                r.get("forced_drop_release"),
            )
        ].append(r)
    for key, runs in sorted(arms.items(), key=str):
        n = len(runs)
        prompt = f"{key[2]}/{key[4]}"
        route = f"{key[3]}/cap{key[5]}"
        print(
            f"{path.split('/')[-1][:30]:30s} {str(key[0]):14s} {str(key[1]):11s} "
            f"{prompt:20s} {route:18s} {n:3d} "
            f"{mean(r['profiler_recall'] for r in runs) or 0:8.3f} "
            f"{sum(1 for r in runs if r.get('reviewer_caught')):6d}/{n:<4d} "
            f"{sum(1 for r in runs if r.get('leakage_remediated')):4d}/{n:<2d} "
            f"{mean(r['claimed_holdout_score'] for r in runs):8.4f} "
            f"{mean(r['cost_usd'] for r in runs):7.4f}"
        )

# ------------------------------------------------------------------------ the loop multiplier
section("RESULT 5: the loop multiplier, within dataset")
first_pass = {
    name: mean([r["cost_usd"] for r in runs if r["review_loops"] == 1])
    for name, runs in by_dataset.items()
    if any(r["review_loops"] == 1 for r in runs)
}
for loops in sorted({r["review_loops"] for r in rows}):
    matching = [r for r in rows if r["review_loops"] == loops and r["dataset_id"] in first_pass]
    if loops == 1:
        print(f"loops=1: n={len(matching)} (reference)")
        continue
    mult = [r["cost_usd"] / first_pass[r["dataset_id"]] for r in matching]
    print(
        f"loops={loops}: n={len(mult)} x{mean(mult):.2f} "
        f"({', '.join(f'{m:.2f}' for m in mult)}) "
        f"datasets={sorted({r['dataset_id'] for r in matching})}"
    )

# ------------------------------------------------------------- objections on the benchmark
section("RESULT 4: objections on the benchmark rows")
objecting = sorted({r["dataset_id"] for r in rows if r["objections_raised"] > 0})
print(
    f"objections raised total {sum(r['objections_raised'] for r in rows)} on "
    f"{sum(1 for r in rows if r['objections_raised'] > 0)}/{len(rows)} rows, datasets {objecting}"
)
print(f"verdicts {dict(collections.Counter(r['review_verdict'] for r in rows))}")
print(
    f"errored {sum(1 for r in rows if r['errored'])}/{len(rows)}; "
    f"fit failures {sum(1 for r in rows if r['n_candidates_failed_to_fit'])}/{len(rows)}"
)
for r in rows:
    if r["objected_columns_unremediated"] or r["errored"]:
        fired = {k: v for k, v in r["objections_by_category"].items() if v}
        print(
            f"  {r['dataset_id']} verdict={r['review_verdict']} loops={r['review_loops']} "
            f"errored={r['errored']} unremediated={r['objected_columns_unremediated']} "
            f"by_target={r['objections_by_target_node']} by_category={fired}"
        )

# ------------------------------------------------- the reissued_ids control column
section("RESULT 3 footnote: reissued_ids nominates the innocent identifier")
# Column order in the fixture CSV, so the opaque rename is positional:
#   var_01 application_ref (genuine unique id, planted as a CONTROL)
#   var_02 member_number   (the TRAP: identifier re-issued by outcome)
#   var_03 credit_score    (a legitimate strong feature)
reissued = [
    r
    for r in load("evals/results/2026-08-31_ci-baseline.jsonl")
    if r["dataset_id"] == "reissued_ids"
]
n = len(reissued)
print(f"n={n}  trap=var_02  control=var_01  legit=var_03")
for col in ("var_01", "var_02", "var_03"):
    nominated = sum(1 for r in reissued if col in r["profiler_nominated"])
    flagged = sum(1 for r in reissued if col in r["leakage_flagged"])
    print(f"  {col}: profiler nominated {nominated}/{n}, reviewer flagged as leakage {flagged}/{n}")
print(
    f"  reviewer_caught {sum(1 for r in reissued if r['reviewer_caught'])}/{n}, "
    f"remediated {sum(1 for r in reissued if r['leakage_remediated'])}/{n}"
)
