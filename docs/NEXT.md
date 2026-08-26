# Next session

## Start here
Phase 1 is roughly half done and the floor is green: 67 fast tests in ~1.3s, 73 in the full suite,
`ds-agents run --dataset toy` exits 0 and prints a two-node trace. What landed: `tools/local.py`
(the four MCP signatures behind a `Tools` Protocol, subprocess execution, stripped environment,
artifact store), `tools/llm.py` (a `StructuredModel` Protocol plus `StubModel`), the `intake` and
`profiler` nodes with unit tests, and `graph.py` wiring `intake -> profiler -> END`.

The leftover reviewer pass from session 0 finally ran and it was worth it: 8 blocking findings,
three of which would have made published numbers wrong (an empty feature matrix counting as
leakage remediation, `score_ratio` raising on reachable zero denominators and taking the whole
results row with it, `holdout_claim_gap` unsigned so overclaims on rmse and roc_auc cancel when
averaged). All fixed, all now covered by branch tests in `tests/test_state.py`. Read the
2026-08-26 entries in docs/DECISIONS.md before changing anything in `results_row()`.

**The one thing to know:** there is no real model client. `StubModel` answers intake from column
conventions and nominates zero leakage candidates by design. Every event it produces is stamped
`model="stub"` and the CLI warns on every run. No number from this repo means anything yet.

## First prompt
Read CLAUDE.md, docs/ARCHITECTURE.md, and docs/DECISIONS.md from 2026-08-26 down. Then write the
real model client in `src/ds_agents/tools/llm.py`: an adapter satisfying the `StructuredModel`
Protocol, with per-model token pricing so `NodeEvent.cost_usd` stops being 0.0. Decide the
LangChain-vs-Anthropic fork first and record it in DECISIONS.md — LangChain buys LangSmith tracing
and the model-swap ablation for free, and `langgraph` is already a dependency, so the burden of
proof is on going direct. Then run the toy pipeline live against Haiku and report what the
profiler actually flags: whether it separates `account_status_code` (mutual information 0.518)
from `support_tickets_90d` (0.094) and `customer_id` (0.194) is the first real signal this project
has produced. After that, continue Phase 1 with `feature_eng` and `modeler` per /add-node.

The key lives in `.env` at the repo root (gitignored, and denied to Claude Code in
`.claude/settings.json`). **Nothing loads it yet** -- there is no `python-dotenv` dependency and no
code that reads it -- so the client work has to add the loader too, at the CLI entry point rather
than inside a node. Nodes never read the environment; that rule is in `state.py`'s docstring and is
what lets `tools/local.py` hand the sandbox a stripped env with no key in it.

## Open questions
- The LangChain-vs-Anthropic fork is now blocking, not theoretical. Everything downstream of it —
  cost accounting, LangSmith, the reviewer ablation — is waiting.
- The harness must refuse to write a results row when any `NodeEvent.model` is `"stub"`. Not built.
  Until it is, nothing structurally prevents a placeholder run from being published.
- `severity` on `Objection` is still written and never read. Either the router blocks only on
  `high`, or delete it rather than paying prompt tokens for it. Unanswered since session 0.
- `NodeEvent` still records no tool calls — not how many, not which, not the code passed to
  `run_python`. No committed run can show what the agents executed. Decide before Phase 2 wires the
  real server.
- `score_ratio` returns `None` for an r2 row, because the predict-the-mean baseline is exactly
  0.0 and a ratio against zero has no meaning. Those rows need a difference column instead. Real
  gap, not a guard.
- The reviewer flagged two contract holes left unfixed, both judged Phase 3 work: nothing rejects
  a `ReviewPass.dispositions` entry naming an objection id that does not exist, and nothing
  notices a pass that omits an objection open on entry.
- Docker-per-run vs one long-lived container, and which OpenML suite has citable published
  baselines. Both still open from session 0.

## Parking lot
- `_strip_value` in `state.py` returns on the first `BaseModel` it finds in `get_args`, so a
  future `dict[str, SomeModel]` field would round-trip wrong. No such field exists today; the fix
  is to dispatch on `get_origin`. Verified harmless for every shape currently on the contract.
- `LocalTools` copies the dataset per run and chmods it 0444. Fine for a 200-row fixture, wasteful
  for a Phase 4 benchmark set.
- The profiler pays a fresh pandas and scikit-learn import per snippet (~0.8s each once warm). A
  long-lived sandbox process would fix it, and that is the same question as Docker-per-run.
- Hint-injection ablation: tell the reviewer which categories of failure exist vs not.
- Cost-per-caught-leak as a headline metric.
- The gap between `claimed_holdout_score` and `verified_holdout_score` may deserve its own table
  in the writeup — "how often does the system overstate itself" is a finding independent of
  leakage.
- ruff formats Python code blocks inside `docs/*.md`, so the hook rewrites design docs on every
  edit. Harmless but surprising; scope the hook to `src/ tests/` if it becomes annoying.
