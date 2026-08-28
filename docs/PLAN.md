# Plan

Budget: roughly 15 to 25 Claude Code sessions. Phases 3 and 4 get the deep sessions.
Phases 0 to 2 are plumbing; keep prompts tight and do not let sessions explore.

## Phase 0: Session 0 (this one)
- [x] Pressure-test docs/ARCHITECTURE.md. Done as an adversarial review rather than an interview;
      8 findings, all absorbed. See docs/DECISIONS.md 2026-08-22 entries.
- [x] `uv init`, pyproject with ruff + pytest config, `fast` marker, `ds-agents` CLI entry point.
- [x] `state.py` from the draft schema, with tests that a fixture round-trips.
- [x] `tests/fixtures/toy/`: 200-row synthetic tabular dataset, binary target, one planted leakage
      column (noisy copy of target), one legit strong feature. Generator script committed, seed fixed.
- [x] Confirm the hook in .claude/settings.json fires and is fast (<10s). Measured 0.44s. NOTE:
      the hook needs a PATH export to find uv in a non-login shell; fix is staged, not applied.
- [x] First DECISIONS.md entries.

## Phase 1: Linear skeleton (2 to 3 sessions)
Scope note: the router turned out to need `NodeName` and a real seat in the graph, so it is listed
here rather than appearing for the first time in Phase 3. `verified_holdout_score` requires the
harness to withhold rows before the graph starts and re-apply `feature_code_artifact` to score
them; that is Phase 4 scope that PLAN.md did not previously name.
- [x] `tools/local.py` shim with the four MCP signatures, behind a `Tools` Protocol. Subprocess
      execution, stripped environment, artifact store, both asserted in tests.
- [x] `tools/llm.py`: `StructuredModel` Protocol plus `StubModel`, so the graph runs with no API
      key. The LangChain-vs-Anthropic fork is now decided (direct SDK) and the adapter is written.
- [x] intake node (spec from the dataset artifact's schema) and profiler node (deterministic stats
      via `run_python`, model nominates leakage candidates, pins `split_artifact`), unit tests each
- [x] graph.py wiring for intake -> profiler, `ds-agents run --dataset toy` prints a node trace
- [x] modeler, reporter nodes. Modeler fits a fixed candidate set on the pinned folds and asks the
      model only which to promote; reporter is deterministic markdown and calls no model.
- [x] feature_eng node. Model proposes a drop plan, a templated snippet does the transform. The
      artifact is a *fitted transform as code*, so Phase 4 re-applies the same code path.
- [x] router as a real node on a conditional edge, with the cycle back to feature_eng/modeler
      wired and tested even though nothing takes it until Phase 3.
- [x] LangSmith tracing wired (`@traceable`, inert without a key -- untested end to end, no key
      yet). NodeEvent cost accounting live: a real toy run reports ~$0.009.
- [x] real model client (`AnthropicModel`, direct SDK), per-model pricing, and the guard --
      `PipelineState.publishable()`, which Phase 4's harness must call before writing a row.

Scope note: the guard landed as a method on `PipelineState` rather than inside `harness.py`, so the
CLI and the not-yet-written harness share one implementation. The CLI prints it on every run.
First live Haiku runs done: the profiler flagged the planted leak in 9 of 9 runs, `feature_eng`
acted on it in 8 of 9.

## Phase 2: MCP server (2 to 3 sessions)
Scope note: the Docker-versus-long-lived-container question resolved to neither. The sandbox is a
warm worker process that forks per snippet — same isolation per snippet, one twentieth the cost,
and it does not make `pytest -m fast` depend on a running Docker daemon. Docker is deferred behind
the `SandboxPool` seam until model-authored feature code needs a memory cap and a network block.
See DECISIONS.md 2026-08-27.
- [x] `mcp_server/` sandbox and store: `_worker.py` (fork-per-snippet, stripped env, pruned
      `sys.path`, process-group timeout), `sandbox.py` (worker lifetime, request protocol,
      outer deadline), `store.py` (artifact store, `sandbox_path` in metadata, metric log to
      JSONL). 25 tests. `tools/local.py` rewritten as a thin binding of the same two objects.
- [x] `mcp_server/server.py`: the four tools over the MCP protocol, wrapping the objects above.
      It constructs a `LocalTools` and exposes its four methods, so the protocol layer decides
      nothing; one server process per run, because keying stores by `run_id` would need a fifth
      tool and the tool surface is what the Phase 5 ablation holds constant.
- [x] swap `tools/local.py` for MCP client wrappers; toy run still green. `tools/mcp_client.py`
      satisfies the same Protocol, `ds-agents run` defaults to `--tools mcp`, `--tools local`
      stays for debugging. Three live Haiku runs through MCP: leak dropped 3 of 3, roc_auc 0.843,
      15.4s and $0.0098, all matching the in-process numbers.
- [x] verify Claude Code can connect to the same server (this is the generalist-agent baseline
      later). Done from inside Claude Code itself: the four tools resolve as
      `mcp__ds-agents-tools__*` and all four were exercised against the toy dataset by an agent
      that shares no code with `tools/mcp_client.py`. `run_python` saw the dataset at
      `$DS_DATASET`, the repo off `sys.path` and no API key in the environment; `write_artifact`
      returned an id and a `sandbox_path`; `read_artifact` with `max_bytes=40` returned
      `truncated: true` and the same artifact opened whole from inside a snippet at its
      `sandbox_path`; `log_metric` accepted a number. Residual, and it is a CLI flag rather than a
      capability: `claude mcp list` still prints "pending approval" for the project-scoped entry,
      so a fresh terminal session prompts once. See DECISIONS.md 2026-08-27.

## Phase 3: Adversarial reviewer (4 to 6 sessions, plan mode)
Scope note: the `ReviewPass` split-ownership problem was settled before the node was written, and
it changed who writes what. The router now mints the whole `ReviewPass` and the reviewer hands its
dispositions across on a new narrow field, `reviewer_dispositions`. `routed_to` got a single
authority at the same time: the router computes the destination and `route_target` reads it back.
See DECISIONS.md 2026-08-27.

Scope note: fixtures needed a registry before a second one could be run at all -- `cmd_run` rejected
every `--dataset` but `toy` and hardcoded the CSV path. `src/ds_agents/fixtures.py` types the
manifest and resolves a fixture by name; nothing in `nodes/` imports it, because a node that could
read a manifest could read the answer key.
- [x] reviewer node with structured Objection output and conditional routing
- [x] loop cap, `exhausted` verdict. Both reached live, not just in tests: 2 of 7 live Haiku runs
      ended `exhausted` at `loop_cap=3` on a `metric_mismatch` objection the modeler never
      satisfied.
- [x] feature_eng and modeler consume objections. Wired in Phase 1; this session is the first
      evidence it works end to end, in `tests/test_review_loop.py` (block once, drop the objected
      column, pass) and in the live runs that took the cycle edge back to `modeler` twice.
- [x] reviewer catches the toy leakage; test that it does. **Answered, and the answer is no.** It
      took three fixtures to get a leaky matrix in front of the reviewer at all. In the 3 runs where
      a planted trap did survive upstream and ranked first or second by permutation importance, at a
      claimed roc_auc of 0.954 to 0.986 against a legitimate ceiling near 0.82, the reviewer returned
      `pass` and raised no leakage or contamination objection in 3 of 3. It never mentioned the
      columns. Needs a larger n and a Sonnet arm before publication, but the box is no longer
      untestable. See DECISIONS.md 2026-08-27.
- [~] 2 to 3 more leakage-trap variants. Two built: `claims_timing` (timestamp after label, two trap
      columns) and `reissued_ids` (ID-encoded target, with a genuine unique id alongside as a
      control). Both ship a seeded generator, a committed CSV and a manifest, and both are proved by
      test to reach `final_features` and `top_importances` under `StubModel`. Duplicate-rows-across-
      split is deferred, not skipped: the reviewer is never shown the split or any rows, so it is
      structurally uncatchable today, and `results_row()` scores leakage as a set comparison over
      columns and cannot score a trap with no guilty column.
- [x] Name transparency as an explicit ablation axis. Landed as a load-time rename
      (`src/ds_agents/naming.py`) rather than a manifest field or a paired fixture: the header line
      is rewritten and every other byte of the CSV is copied through, so the arms differ in one line
      of one file and nothing else. All non-target columns are renamed, not just the traps -- an
      opaque name that only traps carry is a replacement cue, not the absence of one. `naming` is on
      the frozen `RunConfig` and in `results_row()`. Nothing in `nodes/`, `tools/` or `mcp_server/`
      changed; the rename lives above the tools boundary and the only thing that reaches them is a
      different CSV path. See DECISIONS.md 2026-08-27.

- [x] Reviewer prompt as an explicit ablation axis, and the reviewer's columns on the results row.
      `reviewer_prompt` is a `Literal` on the frozen `RunConfig`; `which_column` appends one rule to
      `REVIEWER_SYSTEM` asking which top-importance column explains an implausible score, and `base`
      is byte-identical to every earlier run. `results_row()` gained
      `reviewer_nominated/caught/recall/false_alarm` over all column-scoped categories plus
      `objections_by_category`; `leakage_*` keeps its narrower two-category definition so the
      committed naming-ablation rows stay comparable. See DECISIONS.md 2026-08-28.
- [x] Reviewer 2x2 run live: `{haiku, sonnet}` x `{base, which_column}` on `claims_timing --naming
      opaque`, 27 rows at `evals/results/2026-08-28_reviewer-ablation.jsonl`, $1.04. **The prompt is
      the binding constraint and the model is not**: Haiku goes from 1/10 to 9/10 runs naming a
      planted trap, while Sonnet under the base prompt is 0/4. Scope note: the Haiku-versus-Sonnet
      arm listed under Phase 5 was pulled forward, because the opaque arm is the only configuration
      that reliably puts a leaky matrix in front of the reviewer and the prompt was a confound under
      any model number. The Sonnet cells are underpowered (n=4 and n=3) -- Sonnet cost $0.078-0.093
      per run against a budgeted $0.035 and the pre-registered $1.20 cap bound them. **Topped up
      2026-08-28 to n=7 per Sonnet cell** ($0.613). The prompt effect holds (sonnet/base 0 of 7).
      The model effect separated from it and is not about detection: under `which_column` Haiku
      names a trap more often (9/10 vs 5/7) and Sonnet remediates more often (3/7 vs 1/10), because
      Sonnet addresses its objection to `feature_eng` rather than `modeler`. Directional at this n.
- [x] Diagnose the remediation bottleneck. **The loop cap was never it.** A pre-registered sweep at
      `loop_cap` 1/3/5, n=10 each on `claims_timing --naming opaque --reviewer-prompt which_column`,
      moves remediation 0, 1, 1 out of 10 while quadrupling cost per run --
      `evals/results/2026-08-28_loop-cap-sweep.jsonl`, 20 new rows at $0.694, with the cap=3 arm
      reused from the reviewer ablation. Three diagnostic runs ($0.10) named two causes instead, and
      both are independent of the cap: the reviewer addresses `implausible_importance` to `modeler`,
      which has no column lever, so `feature_eng` never sees the objection; and it never dispositions
      an objection `resolved`, even in a run that dropped both traps and watched the claimed score
      fall to the legitimate ceiling. `exhausted` therefore is not evidence the trap shipped --
      `leakage_remediated` is. See DECISIONS.md 2026-08-28 (second entry).
- [x] `results_row()` carries why a caught leak was not fixed: `objections_by_target_node`,
      `objected_columns_unremediated`, `route_sequence`, `new_objections_per_pass`, all derived from
      existing state, no node changes. `--loop-cap` became a real CLI flag at the same time -- it had
      been on the frozen `RunConfig` and on every results row since the first one, with no way to
      set it.
- [ ] Fix the remediation path. Deliberately deferred so it gets a controlled before/after against
      the sweep above. Routing is the leading candidate (`implausible_importance` naming a column is
      a `feature_eng` problem whatever the prompt calls it); a closure condition the reviewer can
      actually observe is a second, separate change.

## Phase 4: Eval harness (3 to 5 sessions, plan mode)
- [ ] evals/datasets/manifest.yaml with 10 to 15 OpenML / Kaggle playground datasets and
      published baselines, sources cited
- [ ] harness.py, results JSONL, eval-diff command
- [ ] `ci` subset, GitHub Actions job with thresholds
- [ ] LOG.md running

## Phase 5: Ablations and writeup (3 to 4 sessions)
- [~] reviewer on/off, Haiku/Sonnet reviewer, single agent vs team, loop cap 1/3. Two of these ran
      early, in Phase 3. The Haiku/Sonnet reviewer arm ran crossed with a prompt condition --
      `evals/results/2026-08-28_reviewer-ablation.jsonl`; its Sonnet cells were topped up to n=7
      each on 2026-08-28, short of the pre-registered n=8 because the loop-cap sweep overran its
      cost estimate, so the model main effect is better powered but still not at n=10. The loop-cap
      ablation ran at 1/3/5 rather than 1/3 -- `evals/results/2026-08-28_loop-cap-sweep.jsonl`, a
      null result on remediation. Reviewer on/off and single-agent-vs-team are untouched.
- [ ] README as a short paper: thesis, setup, results tables, failure analysis, design section
      lifted from DECISIONS.md
- [ ] resume bullet with real numbers

## Out of scope unless the core is done and credits remain
- non-tabular data, hyperparameter search, a web UI, TypeScript anything, Kubernetes anything
