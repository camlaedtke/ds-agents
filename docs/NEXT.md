# Next session

## Start here
**Phase 1 is done and the pipeline produced its first real numbers.** All six nodes run —
`intake -> profiler -> feature_eng -> modeler -> router -> reporter` — against a real Haiku client,
with per-token cost accounting. The floor: 165 fast tests in ~1.6s, 177 in the full suite,
`uv run ds-agents run --dataset toy` exits 0 in ~16s and costs about $0.009.

What landed: `AnthropicModel` (direct Anthropic SDK, `messages.parse` constrained decoding),
`tools/pricing.py` with published per-model rates that *raise* on an unknown model rather than
costing zero, `.env` loading at the CLI only, `PipelineState.publishable()` as the guard that
refuses a placeholder run, the four remaining nodes with 61 unit tests between them, and the graph
wired with the Phase 3 cycle edges already in place and tested.

**The first real finding.** Over nine live Haiku runs on the toy fixture, the profiler flagged the
planted leak `account_status_code` at `high` suspicion in **9 of 9** runs, with reasoning that
matched the planted mechanism ("an account status code is likely determined after churn occurs" —
the generator does exactly that). It cleanly separated the leak (MI 0.518) from the legitimately
strong `support_tickets_90d` (0.094), which it never flagged. But `feature_eng` acted on that flag
in only **8 of 9** runs. When it does not, `account_status_code` survives into the matrix and the
claimed score jumps from **0.843 to 0.983**. That is `caught` and `remediated` coming apart in
practice, on a real model, and it is precisely the gap the Phase 3 reviewer exists to close: an open
objection *forces* the drop, so the reviewer should take remediation to 9 of 9.

The profiler also flags `customer_id` at `high` every run. That is defensible data science — a
per-row identifier with MI 0.194 by memorization — but it scores as a **false positive** against
ground truth, because the toy manifest lists `customer_id` under `id_columns`, not under
`planted_leakage`. So `leakage_precision` reads 0.5 on every toy run. Decide what that should mean
before Phase 4 publishes a precision column.

## First prompt
Read CLAUDE.md, docs/ARCHITECTURE.md, and docs/DECISIONS.md from 2026-08-26 down (there are twelve
new entries; the ones on the `neg_*` scorer sign convention and the source-name vocabulary are the
load-bearing ones). Then start Phase 2: `mcp_server/` with a real `run_python` sandbox — Docker or a
long-lived subprocess, with a timeout and no network — plus the artifact store and `log_metric`,
then swap `tools/local.py` for MCP client wrappers and confirm the toy run is still green. Settle
Docker-per-run vs one long-lived container first and record it in DECISIONS.md; the profiler and the
two new snippets each pay a fresh pandas and scikit-learn import (~0.8s), which is most of the toy
run's 16 seconds, and a long-lived process is the only thing that fixes it.

Before writing the server, read `nodes/feature_eng.py` and `nodes/modeler.py` for what the sandbox
contract actually has to support now: snippets read `os.environ["DS_DATASET"]`, write to
`os.environ["DS_ARTIFACTS"]`, and the modeler `exec`s the feature-transform artifact *inside* the
sandbox. Any MCP server has to keep all three working.

## Open questions
- **`customer_id` as a false positive.** Every toy run flags it, correctly on the merits, and scores
  it as a false alarm. Options: add an `acceptable_flags` set to the fixture manifest that
  precision forgives; count id columns as planted leakage; or accept 0.5 precision as the honest
  number and say so in the writeup. This changes a published column, so it is a real decision.
- **Does `feature_eng` deserve its own reviewer-independent retry?** Remediation failing 1 in 9 runs
  with no reviewer is the Phase 3 motivation, but it is also an argument for making the node's
  prompt harder to ignore. Measuring the reviewer's effect is cleaner if the baseline is left alone.
- Docker-per-run vs one long-lived container. Now the biggest lever on wall time.
- **LangSmith is wired but never exercised** — `@traceable` on `AnthropicModel.generate`, inert with
  no key. Add `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true` to `.env` to turn it on; nothing in
  the code changes. Until someone does, treat "tracing works" as unverified.
- Which OpenML suite has citable published baselines. Open since session 0.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`. Both
  currently reach only `state.errors` or vanish. Phase 3's reviewer will want the rationale to
  critique the model choice; decide then whether it earns a field.

## Parking lot
- **`ReviewPass` has a split ownership problem, and Phase 3 will hit it immediately.** It carries
  `dispositions` (only the reviewer knows them) and `routed_to` (only the router knows it), and
  `review_passes` is `operator.add` — so if both nodes append, every iteration produces two records.
  Recommended fix when the reviewer lands: the router writes the whole `ReviewPass`, and the
  reviewer hands dispositions across on a new narrow field. Do not add that field before it has a
  writer.
- **The split manifest is embedded in snippet text, which is O(n_rows).** Fine at 200 rows, roughly
  7 MB of snippet on a 100k-row Phase 4 dataset. The fix is exposing `sandbox_path` in
  `ArtifactMeta.extra` so a snippet can `open()` the artifact instead — and requiring the Phase 2
  MCP server to do the same. Worth doing *during* Phase 2, not after.
- `permutation_importance` costs `n_source_columns x n_repeats` scoring passes per candidate.
  Trivial on toy; at Phase 4 sizes restrict it to the best-by-CV candidate or drop `n_repeats` to 5.
- `_strip_value` in `state.py` returns on the first `BaseModel` in `get_args`, so a future
  `dict[str, SomeModel]` field would round-trip wrong. No such field exists; fix is to dispatch on
  `get_origin`.
- `LocalTools` copies the dataset per run and chmods it 0444. Fine for 200 rows, wasteful for a
  benchmark set.
- `feature_eng` picks columns but does not write code. Revisit in Phase 2 once `run_python` is
  genuinely isolated and model-authored feature code has a smaller blast radius.
- Hint-injection ablation: tell the reviewer which categories of failure exist vs not.
- Cost-per-caught-leak as a headline metric. Now computable — a toy run is ~$0.009.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
  Harmless but surprising; scope the hook to `src/ tests/` if it becomes annoying.
