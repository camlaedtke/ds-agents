# Next session

## Start here
**The reviewer is live and the loop closes.** `modeler -> reviewer -> router` is wired, the router
mints the whole `ReviewPass`, and runs cycle back to `feature_eng`/`modeler` and terminate at the
cap. Phase 3 boxes 1 to 3 are ticked. Two things are worth knowing before touching anything.

First, the ownership question that NEXT.md has been carrying since Phase 1 is settled and it moved
a contract. The reviewer writes `objections`, `reviewer_claim` and a new field,
`reviewer_dispositions`; it never writes `review_passes`, `review_verdict` or `review_iterations`.
The router folds the dispositions into the `ReviewPass` it mints, and it now also owns `routed_to`,
which `route_target` reads back instead of re-deriving — the two see the state on opposite sides of
the merge and genuinely disagree on a pass that resolves one node's objection while raising
another's. See DECISIONS.md 2026-08-27.

Second, and this is the real finding of the session: **the reviewer never saw the planted leak.**
In 7 live Haiku runs it raised zero leakage objections, because the profiler flags
`account_status_code` and `feature_eng` drops it before the reviewer is called. The toy fixture
cannot currently demonstrate the thing Phase 3 exists to measure. That is not a bug in the node; it
is the fixture being too easy, and it makes the leakage-trap variants the next task rather than a
nice-to-have. What the reviewer did do, in 2 of 7 runs, was raise `metric_mismatch` against the
modeler and never withdraw it, ending `exhausted` at `loop_cap=3`. I did not capture the evidence
text and have not judged whether that objection is legitimate; it is the first question below.

The floor: 251 tests pass in 37.0s full, 230 in 6.9s fast. Live Haiku toy runs are now ~$0.013 and
~18s when the reviewer passes on the first look (5 of 7), ~$0.030 and ~40s when it blocks to the cap
(2 of 7), against a pre-reviewer floor of $0.0098 and 15.9s. All 7 exited 0 with `publishable: yes`,
`errors` empty, roc_auc claimed 0.843, and `account_status_code` and `customer_id` dropped.

## First prompt
Read CLAUDE.md, docs/PLAN.md and the "Start here" above, then continue Phase 3 in plan mode: the
leakage-trap variants. Build 2 to 3 fixtures alongside `tests/fixtures/toy/` whose leak survives the
profiler and `feature_eng` and therefore actually reaches the reviewer — timestamp-after-label,
ID-encoded target, duplicate rows across the split. The generator scripts get committed with fixed
seeds like the toy one. The point is to make PLAN.md's "reviewer catches the toy leakage" box
testable at all: right now the upstream nodes clean the matrix first and the reviewer has nothing to
catch. Decide per fixture what `planted_leakage_columns` should say, since duplicate-rows-across-the-
split has no single guilty column and the contamination category may need a different ground truth
shape than a column list.

## Open questions
- **Is the `metric_mismatch` objection real?** The reviewer raised it against the modeler in 2 of 7
  live runs and never withdrew it, which is what produced both `exhausted` outcomes. The spec asks
  for roc_auc on a binary task, which looks correct on its face, so either the reviewer is wrong in
  a way worth measuring or the prompt is showing it something misleading. Read the evidence text
  from a blocking run before deciding. If it is a false objection, it is also the first natural
  candidate for a false-alarm metric that counts non-column objections, which `results_row()` does
  not currently do — `false_alarm` only counts columns.
- **A refused model call loses its token accounting.** `AnthropicModel.generate` raises
  `ModelRefusal` before it builds the `Completion`, so `NodeRun` books nothing and the node event
  reads $0.00 and 0/0 tokens for a call that was made and billed. Observed live this session: a
  crashed reviewer pass showed a 3.98s event costing nothing. `_run.py` exists precisely so failed
  runs do not read cheap, and this is the same failure one level lower. The fix is to carry usage on
  the exception and record it in each node's `except` block, which touches `llm.py`, `_run.py` and
  all six nodes — too wide to bolt onto this session, but it biases the cost table of exactly the
  runs Phase 5 will care about.
- **`customer_id` as a false positive.** Unchanged for five sessions, and now concrete rather than
  hypothetical: the reviewer has its own opinion about it and there is still no rule saying who is
  right. Every run flags it, correctly on the merits, and scores it as a false alarm because the
  manifest files it under `id_columns` rather than `planted_leakage`, pinning `leakage_precision` at
  0.5. Options unchanged: an `acceptable_flags` set the metric forgives, count id columns as planted
  leakage, or publish 0.5 and explain it. It changes a published column, which is why it keeps not
  being decided in passing.
- **Does `feature_eng` deserve its own reviewer-independent retry?** Still open, still argues against
  itself: measuring the reviewer's effect is cleaner if the baseline is left alone.
- **LangSmith is wired but never exercised.** `@traceable` on `AnthropicModel.generate`, inert with
  no key. Treat "tracing works" as unverified until someone adds the key.
- Which OpenML suite has citable published baselines. Open since session 0.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.

## Parking lot
- **A per-item rule on a response schema is a whole-response rule.** The reviewer's first
  implementation put `Objection`'s "column-scoped categories need a column" validator on the
  model-facing `ProposedObjection` too. Haiku raised one `implausible_importance` with no columns,
  the entire `ReviewFinding` failed to parse, and a pass carrying a real claim and real dispositions
  was recorded as a crash. Fixed by filtering one objection at a time inside the node. Worth
  checking the other nodes' LLM-facing schemas for the same shape — anywhere a validator on a list
  element can take down the response that contains it. `intake` and `modeler` are the candidates.
- **The static review did not catch that bug and the live run did.** The diff review checked every
  contract invariant and passed it clean; the failure only appeared when a real model produced a
  shape the tests had not imagined. Worth remembering when deciding how much a green suite is worth
  on a node whose input is a model.
- **The split manifest is still embedded in snippet text**, and the 1 MiB read cap makes it urgent
  rather than merely wasteful. Every artifact carries `sandbox_path` in `ArtifactMeta.extra` and
  `ArtifactStore.path_of()` returns it, so a snippet can `open()` the manifest instead of having it
  rendered into its source. A Phase 4 dataset around 100k rows produces roughly 7 MB of manifest,
  which trips the cap and fails the run loudly in `feature_eng` and `modeler`. Loud is better, but
  the fix is the same fix: `nodes/feature_eng.py`, `nodes/modeler.py` and their tests.
- **Docker is deferred, not rejected, and `SandboxPool` is the seam.** What it buys that the current
  sandbox does not is a memory cap and a network block. Neither matters while every snippet is
  templated by us; both matter the moment model-authored feature code runs.
- **The reviewer-off arm still runs the reviewer node**, as a zero-cost no-op that writes no claim.
  So `node_trace` contains a `reviewer` row in both arms and anything comparing them must read
  `model`/`cost_usd`, not the presence of the node name. Deliberate — a graph conditional would put
  logic in `graph.py` — but it is a trap for whoever writes the ablation table.
- **`--reviewer-model` exists on the CLI** and binds a second client to the reviewer node only. It
  is the Haiku-vs-Sonnet ablation's whole mechanism, unexercised so far.
- `permutation_importance` costs `n_source_columns x n_repeats` scoring passes per candidate.
  Trivial on toy; at Phase 4 sizes restrict it to the best-by-CV candidate or drop `n_repeats` to 5.
- `_strip_value` in `state.py` returns on the first `BaseModel` in `get_args`, so a future
  `dict[str, SomeModel]` field would round-trip wrong. `reviewer_dispositions` is `dict[str,
  Disposition]` and `Disposition` is a `Literal`, so it does not trip this; the fix is still to
  dispatch on `get_origin`.
- `ArtifactStore` copies the dataset per run and chmods it 0444. Fine for 200 rows, wasteful for a
  benchmark set.
- `feature_eng` picks columns but does not write code. Revisit once a Docker backend gives
  model-authored feature code a smaller blast radius.
- **`.mcp.json` hardcodes the toy dataset on argv**, right for today and wrong for Phase 5, where
  the generalist arm has to face each benchmark dataset in turn. The server already takes
  `--dataset` and `--dataset-id`, so the fix belongs to whatever launches the single-agent arm.
- Hint-injection ablation: tell the reviewer which categories of failure exist vs not.
- Cost-per-caught-leak as a headline metric.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
  Harmless but surprising; scope the hook to `src/ tests/ mcp_server/` if it becomes annoying.
