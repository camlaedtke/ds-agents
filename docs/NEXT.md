# Next session

## Start here

The claims_timing reproduction closed the largest standing risk to a published number: Result 3
reproduces at HEAD (`evals/results/2026-09-21_claims-repro.jsonl`, n=10, $0.3006, all four primary
endpoints matching, `review_loops` to the second decimal). It also surfaced a new one -- the
profiler's opaque-arm recall on the same fixture reads 2/10, 5/10, then 8/10 across three dates
(chi-square(2) = 7.20, p = 0.027), a flag and not yet a finding. This session did no new
measurement; it shortened the docs. The module and CLI flags that captured a run's final state for
the walkthrough, and the scripts that rebuilt the walkthrough page from it, are gone;
`docs/explainers/pipeline-walkthrough.html` stays as a static page. `docs/DECISIONS.md` is now an
index of one-line summaries; the full text of every decision is in `docs/decisions-archive.md`.

## First prompt

Read CLAUDE.md, `docs/RESULTS.md`, and the "Start here" above.

Fund the naming-ablation replication: two cells differing only in `naming`, `claims_timing`, n=10
each, at one commit, with `DS_AGENTS_GIT` set so the rows carry a commit hash (the system `git` is
blocked by an unaccepted Xcode license on this machine -- `sudo xcodebuild -license` fixes it, or
prefix invocations with `DS_AGENTS_GIT=/Users/cameronlaedtke/anaconda3/bin/git`). Pre-register
`profiler_recall` per arm: the 2026-08-27 values are descriptive 0.950 and opaque 0.100, and the
open prediction is that descriptive holds while opaque lands near 0.40. Add the cells as a
`naming-repro` subset the same way `claims-repro` was added, sharing objects where a cell already
exists. Do not edit RESULTS.md Result 2 until this runs -- about $0.30, and the caveat already
there is honest; a rewritten number on n=10 post-hoc would not be.

Note what it also tests: if the opaque arm comes back near 0.10, the 8/10 from 2026-09-21 was
chance. If it comes back near 0.40, the profiler's behavior has changed under a fixed prompt on a
fixed fixture -- a finding about agent pipelines generally, since nothing in the repo pins or
tracks the dated model id behind `"haiku"` (see Open questions).

If more is funded after that, the redundancy ablation is next: H_concentration's falsification
test, about $0.40, fixture variants of `australian` and `sylvine` with near-duplicate copies of
the dominant column so permutation importance collapses while per-column NMI holds steady.

## Open questions

- **Is the profiler's opaque recall drifting, or was 8/10 chance?** The first prompt. The
  asymmetry in what the two outcomes cost is worth keeping in mind: "chance" costs nothing and
  "drift" invalidates the comparability of every cross-date comparison in the repo.
- **Nothing in the repo pins the model behind `"haiku"`.** `RunConfig.default_model` carries an
  undated family alias (`tools/pricing.py`'s `ALIASES` resolves it), which is the right call for a
  pricing table and the wrong one for a provenance record. If the drift question resolves to
  "drift", this is the mechanism: no committed row can say which dated model id produced it.
- **The fixture arms other than `claims-opaque-which` are still unreproduced at HEAD.**
  `reissued_ids` and the toy cell were dropped from the 2026-09-21 run to buy n=10 on the cell that
  carried Result 3. `reissued_ids` carries Result 3's second half (the innocent-identifier finding)
  and hasn't run since 2026-08-31.
- **H_concentration is untested.** See the first prompt. What it doesn't explain: only 6 of 12
  runs on the three objecting datasets actually objected, and concentration is dataset-level, so
  the within-dataset coin flip is unaccounted for.
- **`objections_raised` is 0 on 36 of 52 benchmark rows.** Consistent with both H_concentration
  (nothing admissible to object with) and simply "the reviewer isn't looking" -- the two readings
  aren't yet separated.
- **A run produced no model at all and was recorded `review_verdict: "pass"`.** Still open whether
  that's a scoring bug.
- **No CI job and no thresholds.** The reason is policy, not code -- a gate that spends real API
  money on every push needs a decision from Cameron about whether that's worth it.

## Parking lot

- `docs/readme_numbers.py` is a kept tool. Promote it into `src/ds_agents/` if it earns a CLI entry
  and tests.
- `benchmark.py`'s `BASELINE_DEFINITION.note` and `benchmark_build.py`'s `HEADER` still name
  `baseline_score`; the rename must ride the next `datasets refresh`.
- `ModelResult.model_artifact` is still declared and never written.
- `.mcp.json` still hardcodes the toy dataset on argv.
- `csv_sha256` mismatches warn rather than refuse; decide which.
- `ds-agents eval` appends to a same-day, same-name file rather than refusing, and `cmd_run` has no
  per-run `try/except`. Both want a decision.
- `harness.py` and `cli.py` still import each other inside functions.
- `test_baseline_cost.py`'s synthetic bound is loose by about 1.6x on real data.
- `register_dataset`'s two grader calls are unbounded under both transports (pinned by test);
  needs a watchdog thread or a signal, exposure is 1.2% of budget.
- The AMLB self-signed certificate still blocks vendoring AMLB's own per-dataset numbers; two
  candidates (`guillermo`, `Robert`) couldn't be fetched at all.
- LangSmith is wired but never exercised; unverified until a key exists.
- The `higgs` unforced drop (1 event in 8) hasn't reproduced; needs its own cell before it's quoted.
- `published_reference` picks the max over uploaded runs, not a protocol-stable anchor.
- The 2026-09-09 walkthrough's two replicates ran under the descriptive naming arm while Result 3's
  rows are opaque -- check which naming arm produced a number before treating an old walkthrough
  run as contradicting a published result.
