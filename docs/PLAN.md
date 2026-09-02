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
- [~] Fix the remediation path. **Routing half done; closure half still open.** `objection_routing`
      is a `Literal["as_addressed", "by_category"]` on the frozen `RunConfig`, defaulting to the
      arm that reproduces every committed row byte for byte. Under `by_category` a column-scoped
      objection has an *effective* target of `feature_eng` whatever the reviewer wrote, applied in
      one place (`PipelineState.effective_target`, read only by `open_objections`), so the router
      and `feature_eng` stopped being two independent answers to "who acts on this".
      `Objection.target_node` is never rewritten, so `objections_by_target_node` still measures the
      reviewer's dispatch judgement in the arm that overrides it -- which is what makes this a
      recorded condition rather than a thumb on the scale. n=10 live at $0.272:
      **`leakage_remediated` 5/10 against the control cell's 1/10**, exactly the pre-registered
      ">=5/10 is shown" threshold, with 8/10 runs reaching `feature_eng`.
      `evals/results/2026-08-28_objection-routing.jsonl`. Two costs recorded rather than buried:
      2 of 10 runs dropped `prior_claims_12m`, a legitimate strong feature, because `by_category`
      turns a reviewer false positive into a real drop (hence `n_final_features` on the row); and
      the arm got *cheaper*, not dearer, which contradicted the pre-registration. Three residual
      failures named the trap on the **final** pass, which the cap routes straight to the reporter
      -- so the effective number of actionable passes is `loop_cap - 1`. See DECISIONS.md
      2026-08-28 (third entry).
- [x] Closure: give the reviewer a termination condition it can observe. **Built, pre-registered,
      and the arm was not run -- a recorded null.** `objection_closure` is a `Literal["off","on"]`
      on the frozen `RunConfig`, its own axis rather than a third `reviewer_prompt` value so the
      two prompt rules stay independently attributable; `CLOSURE_RULE` points only at
      `final_features`, a field the reviewer already sees. The control cell was pre-registered as a
      **decision gate** rather than a baseline, because re-reading the committed rows first showed
      the premise did not hold: 8 of 10 routing rows already closed an objection, and all 4
      `exhausted` runs raised a NEW objection on their final pass rather than refusing to close.
      The gate closed at exactly its line (`objections_resolved > 0` in 6/10), and two other
      pre-registered numbers settle it -- `objections_falsely_resolved` is 0 in every row, and
      `leakage_remediated` is 9/10 with the sole failure being the unrelated zero-objection `block`
      bug, so there is no headroom for a prompt rule to buy.
      `evals/results/2026-08-28_objection-closure.jsonl`, n=10, $0.2938. See DECISIONS.md
      2026-08-28 (fourth entry).
- [x] Decide whether a forced drop is sticky. **Decided: only `withdrawn` releases a column.**
      `resolved` means the problem is fixed and on this pipeline the fix IS the drop, so resolution
      must not un-fix itself; `withdrawn` is the opposite claim and is the only route back for a
      reviewer false positive. `PipelineState.binding_objections` is a second named question
      alongside `open_objections`, with exactly one caller in the graph -- the router and the
      reviewer keep asking `open_objections`, or nothing terminates. Applied unconditionally, not
      as an axis. **Unpredicted and the largest remediation effect measured so far: this alone
      moved `leakage_remediated` 5/10 to 9/10** against the routing cell, with 6 of 6 resolving
      runs remediating. It has no pre-registration of its own and needs its own cell before it goes
      in a results table.
- [x] Confirm the sticky-drop effect against a same-commit control. **Run, and it refuted the
      headline rather than confirming it.** `forced_drop_release` is a `Literal["withdrawn_only",
      "resolved_or_withdrawn"]` on the frozen `RunConfig` -- the one condition here whose default is
      NOT the old behaviour, because the off value reproduces a defect rather than offering a second
      design. Applied in one place, and under the off value `binding_objections` is provably
      `open_objections` again. Both arms n=10 at one commit in one file,
      `evals/results/2026-08-28_forced-drop-release.jsonl`, $0.5420 against a $0.85 cap.
      **The primary failed with the wrong sign: control 7/10, sticky 6/10, a difference of -1/10
      against a pre-registered +4/10, and the sticky arm did not reproduce its own 9/10.** The
      5/10 -> 9/10 line is retired. The finding that replaced it is bigger: four cells now exist at
      the same nominal config and **two running identical behaviour returned 9/10 and 6/10**, so
      nondeterminism alone moves a 10-run count on `claims_timing` by about 3 -- which is most of
      the original effect, and puts error bars on every 10-run comparison in this project including
      the routing arm's 1/10 -> 5/10. The mechanism is separately confirmed and the fix stays in:
      `objections_falsely_resolved` is 3/10 in the control and 0/10 in the sticky arm, so
      resurrection is real and the rule prevents it -- it just costs a loop rather than an outcome,
      because the reviewer re-objects to the re-admitted column on a later pass. See DECISIONS.md
      2026-08-28 (fifth entry).

Scope note: Phase 3 reopened for one session to pay the pre-registration debt on its own headline,
and the debt turned out to be larger than recorded -- the headline was wrong. That is the phase
working as intended rather than a setback, but it means the Phase 5 writeup inherits one fewer
result and one more caveat, and that **no 10-run count from this project should be quoted as an
effect without a replicate**.

Scope note: Phase 3 closes with the closure arm as a recorded null and with the reviewer's
resolution behaviour measurable for the first time (`objections_resolved`, `objections_withdrawn`,
`objections_falsely_resolved`). The remaining reviewer defects are both structural rather than
prompt-shaped, and both are Phase 4/5 work: the zero-objection `block` bug, now 1-2 runs in every
10, and late detection against the cap, which is what every remaining `exhausted` run is.

## Phase 4: Eval harness (3 to 5 sessions, plan mode)
Scope note: the harness landed before the manifest, reversing the order listed here. The manifest is
blocked on an open question carried since session 0 (which OpenML suite has citable baselines) and
the harness was not, but the deciding reason is that replicate-awareness is a harness *design*
constraint rather than a feature: a single 10-run cell cannot resolve a 4/10 difference, and
retrofitting that onto a harness built without it would have meant rewriting it. Also folded in:
the zero-objection `block` fix NEXT.md named as the first code change, and three new results-row
columns (`errors`, `commit`, `default_model`) the harness needed in order to be worth running.

Scope note (2026-08-31, third): the phase's last box is closed and `--subset full` now waits on
money rather than on code. The decision it needed turned out to be a *retirement*: `score_ratio`
computed `verified / baseline` where AMLB's convention is `(x - zero) / (unit - zero)`, so both it
and `baseline_score` are gone, replaced by two raw points and one derived column. The pooling hazard
that motivated the deferral is closed by a gate on `planted_leakage_columns` -- written now even
though it cannot fire today, because fixtures withhold nothing and the contradiction only becomes
reachable the day the carve widens. One thing the plan did not anticipate and it re-prices the rest
of the phase: the unit point costs 0.20s on `credit_g` and **72s on `higgs`**, so the yardstick is
invisible at the cheapest dataset and dominant at the largest, and the per-dataset cost estimate
`full` still needs has to be measured at two sizes rather than extrapolated from one.

Scope note (2026-08-31, second): the re-scorer landed and `--subset full` is one blocker lighter,
not zero. The run path exists and is exercised by `--subset bench-smoke`; what remains is
`baseline_score` and a measured cost for the other twelve datasets. `bench-smoke`'s own estimate
was wrong by more than a factor of two (0.040 guessed, 0.017 measured), which is the argument for
measuring rather than extrapolating before thirteen real datasets are paid for.

Scope note (2026-08-31): the manifest landed and that session-0 question is answered -- AMLB,
OpenML suite 271. The phase does NOT close with it, because building the manifest revealed that
the box was two boxes: a registry of datasets, and a path by which a dataset without a planted
answer key can be run and scored at all. The first is done. The second is listed below as its own
box, and it is what `--subset full` now waits on.
- [~] evals/datasets/manifest.yaml with 10 to 15 OpenML / Kaggle playground datasets and
      published baselines, sources cited. **The file exists and the session-0 blocker is closed.**
      13 binary datasets from the AutoML Benchmark (Gijsbers et al., JMLR 25(101) 2024; OpenML
      suite 271), chosen by a recorded rule applied to measurements rather than by hand, each
      pinned by OpenML data id + task id + upstream md5 and each carrying a `published_reference`
      citable to a stable OpenML run id. The manifest is GENERATED -- `.claude/settings.json`
      denies Edit/Write under `evals/datasets/`, so `ds-agents datasets refresh` is the only
      writer, which is what makes "no number in it was typed" a property rather than a promise.
      Verified against the live APIs by 25 opt-in `network` tests and by `datasets verify
      --online`, which re-fetches and diffs clean. `[~]` and not `[x]` for one honest reason:
      AMLB's OWN per-dataset numbers are not vendored, because its raw-results store presents a
      self-signed certificate and a number this repo cannot re-fetch is one it will not publish --
      what is cited instead is OpenML's evaluation API. Kaggle is declined, not deferred: no
      reproducible published protocol. **No dataset here can be run yet** -- see the box below.
      See DECISIONS.md 2026-08-31 (second entry).
- [x] harness.py, results JSONL, eval-diff command. `src/ds_agents/harness.py` (not `evals/`, which
      is not a package -- see DECISIONS.md 2026-08-29 third entry) and `src/ds_agents/evaldiff.py`,
      behind `ds-agents eval` and `ds-agents eval-diff`. Replicate-major planning, an
      invocation-level cost cap, per-run failure isolation, and an injected `Runner` seam that makes
      the whole thing testable with no API key. `eval-diff` refuses to call an underpowered
      difference an effect, excludes `None` metrics from denominators, and reports per-replicate
      counts beside pooled Wilson intervals. Smoked live at n=1 on `toy` for $0.0135 --
      `evals/results/2026-08-29_harness-smoke.jsonl`, which is a write-path proof and not a cell.
- [~] `ci` subset, GitHub Actions job with thresholds. **The subset has now been run live**, 3
      cells x 2 replicates x n=5, 30 rows at $0.7291, 0 refused and 0 failed --
      `evals/results/2026-08-31_ci-baseline.jsonl`. It bought the first live evidence that the
      zero-objection `block` retry fires (7 of 30 runs, 4 of those 7 rescued by the retry), measured
      per-run costs that replaced the three guesses in `SUBSETS`, and one defect the harness could
      only find by running more than once: `_run_once` read `git_commit()` per run, so writing the
      results file dirtied the tree and split `toy-default` across two `commit` values. Fixed to a
      once-per-invocation read. Still no CI job and no thresholds, and the reason has changed: a
      threshold is now affordable to set, but a gate that spends real API money on every push needs
      a policy nobody has written, there is no `.github/` in this repo, and no key is available to
      Actions. Scope note: this run does NOT serve as the comparand for later ablations. `commit` is
      an `eval-diff` condition field and a new arm is a new commit, so both arms of a comparison
      have to run in one invocation -- see DECISIONS.md 2026-08-31.
- [x] LOG.md running. Seven entries; the newest is the harness smoke, labelled as not-a-cell.
- [~] **The re-scorer. Built and run; `baseline_score` is the one piece deliberately deferred.**
      A manifest dataset is runnable end to end: `src/ds_agents/runnable.py` (a concrete adapter
      over `Fixture | DatasetEntry` rather than the `typing.Protocol` this line asked for -- there
      is no type checker here, so a structural protocol would document the guardrail instead of
      being it), `holdout.py` (20% carved before the graph starts, stratified, numpy-only, seeded
      from `RunConfig.random_seed`, `credit_g`'s indices pinned by test), `rescore.py` (refit from
      `CANDIDATE_SPECS` -- never from `ModelResult.params`, which is a flat merge across steps and
      is lossy -- scored in a sandbox through the grader's own `LocalTools`). `_fixture_state`
      became `_run_state`. Seven new results columns including `rescore_status`, a 13-value enum
      that is `leakage_graded`'s fix applied to a score.
      **The self-check is what makes the number a measurement rather than a number**: the same
      pipeline is fit on the agents' train split and scored on the agents' own holdout first, and
      compared to what the modeler claimed there. `refit_claim_gap` is exactly 0.0 on every run so
      far, offline and live. Sensitivity is proved by construction rather than by the live rows --
      `tests/test_rescore.py` builds a dataset whose only signal is absent from the withheld rows
      and measures claimed 1.0 against verified 0.45.
      Live: `evals/results/2026-08-31_credit-g-smoke.jsonl`, 4 rows at $0.0673, `rescore_status`
      ok 4/4, claimed 0.7520 against verified 0.7476. **All four rows are numerically identical,
      so the replicates bought no variance estimate** -- `credit_g` has no leak, the reviewer
      raised nothing and `feature_eng` dropped nothing, so no decision was available to be made
      differently. `[~]` for one reason: `baseline_score` is not built, so `score_ratio` is null on
      every benchmark row. Deferred rather than rushed -- the unit point's recipe would be ours and
      not AMLB's, and the baseline sees every column including a leak, so `score_ratio` inverts its
      meaning between labelled and unlabelled datasets. See DECISIONS.md 2026-08-31 (third entry).
- [x] **The baseline, and the pooling hazard it carried. `score_ratio` is retired.** The answer to
      "can `score_ratio` be published at all" was no: it computed `verified / baseline`, which is
      not AMLB's normalisation and disagrees with it about what 1.0 means. What ships instead is
      two raw points plus one derived column -- `baseline_zero_score` (a constant class-prior
      predictor, exactly 0.5 for roc_auc by construction and therefore a correctness assertion on
      positive class, scorer sign and row selection at once), `baseline_unit_score` (a RandomForest
      whose grid is OURS and is versioned on the row as `baseline_recipe`), and
      `baseline_normalised_score = (verified - zero) / (unit - zero)`, which returns `None` on any
      dataset with a planted leak because the baseline is fit on every raw column including the
      trap. Both points fit on the agents' `split["train"]` and scored on the same withheld holdout
      as the run, in their OWN sandbox process with their own timeout and their own ten-value
      `BaselineStatus` -- a RandomForest that dies must not take `verified_holdout_score` with it,
      and `unit_point_failed` keeps the zero point. Retiring two published columns was free and
      that is checked rather than claimed: all 149 committed rows carry both as `null`, pinned by
      test, and no committed results file was edited. Sensitivity is proved by construction, as it
      was for the re-scorer: `tests/test_rescore.py` builds a 60-level string column `feature_eng`
      skips and the grader's encoder keeps, so the pipeline scores the class prior and the baseline
      scores near 1.0 -- a grader reporting the pipeline's number twice would fail it. Live on
      `credit_g`: `baseline_status` ok, zero exactly 0.5, unit 0.7660 against a verified 0.7476,
      normalised 0.9309. See DECISIONS.md 2026-08-31 (fourth entry).

Scope note (2026-09-01): the phase does not close, and the reason changed shape. `--subset full`
was down to money; it is now down to CODE again, and the money question is answered. Nine of the
thirteen datasets are priced (`bench-mid`, 8 rows, $0.2120); the other four -- `adult`,
`bank_marketing`, `higgs`, `numerai28_6` -- **cannot complete a run at all**, because the profiler
writes the split as a JSON list of every row index and `read_artifact` caps reads at 1 MiB. The
failure is silent: `feature_eng` refuses with `recoverable=False`, nothing branches on
`recoverable`, so the run spends full tokens and writes a publishable row with no model in it. The
cost model itself came out simpler than planned and for an unwelcome reason -- the pre-registered
"tokens track columns, wall clock tracks rows" prediction was half wrong. Both track columns, because
the modeler's `permutation_importance` is `10 x n_columns` scoring passes per candidate and
dominates everything: 108s at 144 columns against 8s at 9. The baseline, this phase's whole worry,
is the one row-driven term and is never more than 4% of a run.
- [x] **The grader's own wall cost, and the 2x2 that priced nine datasets.** `wall_seconds` stops
      when the graph returns and the grader runs after it, so the term `full` was blocked on pricing
      was structurally invisible in every committed row -- a run WITH two extra fits read faster
      than one without. Fixed with `rescore_seconds`, `baseline_seconds` and `node_seconds`, all
      derived from existing state, no node changes. `bench-mid` crosses few/many rows with few/many
      columns (`phoneme`, `jasmine`, `amazon_employee_access`, `nomao`) so the axes could be
      separated rather than assumed. Endpoints: `rescore_status` and `baseline_status` ok 8/8,
      `baseline_zero_score` exactly 0.5 on 8/8, `refit_claim_gap` exactly 0.0 on 8/8, every cell
      under its estimate. `evals/results/2026-09-01_bench-mid.jsonl`. See DECISIONS.md 2026-09-01.
- [x] **Make the split manifest fit, so the other four datasets can be run. Done, and `full` is
      back to waiting on money alone.** The manifest is now one character per agent row -- `h` for
      holdout, `0`..`4` for the fold that row validates in -- and every other partition is derived.
      `higgs` goes 2,690,410 B to 78,831 B, 7.5% of the read cap, with headroom to roughly a million
      rows. All four formerly-unrunnable datasets complete end to end, proved offline for $0 with
      `--no-live` before anything was funded, and `adult` produced the first graded row of the four
      live at $0.0158 (`rescore_status` ok, `baseline_status` ok, `refit_claim_gap` exactly 0.0,
      `baseline_normalised_score` 1.054 -- the first row in this project above the raw-column floor).
      Not a cap bump, for the reason the box said: the JSON text is substituted into four snippet
      sources, not just read. Run-length was costed and is 8.3x WORSE than explicit lists (mean run
      length 1.21 on a shuffled assignment); base64 is 1.33x worse than the digit string; seed
      re-derivation is refused because the split stops being a recorded object and would depend on
      the installed sklearn version. `evals/results/2026-09-01_adult-smoke.jsonl`. See DECISIONS.md
      2026-09-01 (second entry).
- [x] **`recoverable` is read. A fatal refusal halts the run and the row says so.** Every
      straight-line edge in `graph.py` is now `halt_or(destination)`; a `recoverable=False` error
      routes straight to `reporter` and no further node spends. The row is still written --
      deliberately, because refusing it would delete the hardest datasets from the results file --
      and carries `halted_at`, the node that refused. Fourth instance of the companion-column fix
      after `leakage_graded`, `rescore_status` and `baseline_status`, which closes the standing
      "`errored` needs a companion column" item. `publishable()` is unchanged. See DECISIONS.md
      2026-09-01 (third entry).

- [x] **The last four datasets priced, and `--subset full` built. The phase closes on money being a
      decision rather than a blocker.** `bench-tall` -- `adult`, `bank_marketing`, `numerai28_6`,
      `higgs` -- 16 rows at n=4 a cell, $0.3021, `evals/results/2026-09-02_bench-tall.jsonl`.
      Measured $0.0159 / $0.0178 / $0.0172 / $0.0246. `SUBSETS["full"]` now exists with 13 cells (9
      measured, 4 modelled) and `_resolve_subset`'s bespoke message is deleted after four
      corrections. Endpoints: `rescore_status` and `baseline_status` ok 16/16, `baseline_zero_score`
      exactly 0.5 16/16, `refit_claim_gap` exactly 0.0 16/16, `halted_at` null 16/16, every timing
      inside its budget with room (`higgs` modeler 23.6s of 240s).
      **The arm was pre-registered as a hypothesis test and it returned the null option.** Neither
      "the model needs a row term" nor "the model was fitted on datasets with no categorical
      columns" produced its predicted ordering; what happened instead is that all four residuals
      were positive, making 5 of 5 out-of-sample datasets under-predicted. Refitting on nine
      datasets, a row term helps and a categorical term makes things worse, so **H_categorical is
      refuted despite a real mechanism**. Two defects found that nobody was looking for: `errored` is
      `true` on four completely healthy `adult` runs because `feature_eng` logs a routine
      high-cardinality skip as a `PipelineError`, which means the 2026-09-01 claim that `halted_at`
      CLOSED the "`errored` needs a companion column" item was too strong; and
      `baseline_normalised_score` returns 2.089 on `numerai28_6` because it divides by the unit
      point's span, which on a near-chance dataset is 0.0101. Both reopened in NEXT.md rather than
      patched. See DECISIONS.md 2026-09-02.

Scope note (2026-09-02): Phase 4 closes. Two boxes stay `[~]` and both for reasons outside the
code: the manifest, because AMLB's raw-results store presents a self-signed certificate and a number
this repo cannot re-fetch is one it will not publish; and the `ci` subset, because a GitHub Actions
job that spends real API money on every push needs a policy nobody has written, there is no
`.github/` here, and no key is available to Actions. `--subset full` is runnable and priced at about
$1.10 for 52 runs; it has not been run, because that is a budget decision and it has never been put
to anyone. That is the phase's actual exit state and it is not a blocker of the kind the previous
four scope notes described.

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
