# Learning ledger

Concepts this project depends on, flagged as they come up, so that not understanding something
never blocks a session. Claude appends entries here instead of stopping to explain. Read it when
you want a break from building; run `/learn <slug>` to go deep on one.

Status: `flagged` (noted, not explained) -> `explained` (we did a session on it) -> `owned`
(you could rebuild it without help).

Priority: **load-bearing** means the thesis breaks if this is wrong and you cannot tell.
**useful** means it changes how you read the code. **trivia** means it can stay delegated forever.

## Format

```
### <slug> -- <one-line plain-language description>
- Priority: load-bearing | useful | trivia
- Came up: <date>, <where>
- Status: flagged
- Why it matters here: <one or two sentences, project-specific, not a textbook definition>
- What clicked: optional note manually entered by me
```

## Entries

### langgraph-state-machine -- how LangGraph turns nodes and edges into a running loop
- Priority: load-bearing
- Came up: 2026-08-22, session 0 setup
- Status: owned
- Why it matters here: the reviewer's ability to block and route back is the whole thesis. If the
  conditional-edge and loop-cap semantics are not clear to you, you cannot tell a real "exhausted"
  verdict from a wiring bug, and the eval numbers become uninterpretable.
- What clicked: I understand the basic state machine concept. When it gets to a node, you pull a few fields off the state, build a prompt, call LLM with a Pydantic model, maybe call a tool (i.e. run code through run_python), return a small dict, add to the state and pass it on. All defined with StateGraph. 

### target-leakage -- a feature that encodes the answer, so the model scores well and is useless
- Priority: load-bearing
- Came up: 2026-08-22, session 0 setup
- Status: owned
- Why it matters here: this is what the adversarial reviewer is being measured on. You need to be
  able to look at a planted trap and say why it is a trap, or you cannot grade the reviewer.
- What clicked: totally makes sense. A garbage feature which has a very high correlation with the target. 

### mcp-tool-boundary -- why tools are a separate server rather than Python functions
- Priority: useful
- Came up: 2026-08-22, docs/DECISIONS.md
- Status: explained
- Covered: 2026-08-24, `docs/explainers/mcp-tool-boundary.html` -- traces one profiler call across
  the process boundary and contrasts it with in-process `exec()`, where the answer key would be
  reachable and `leakage_recall` would stop meaning anything.
- Why it matters here: it costs a phase of plumbing (PLAN.md Phase 2), but it is what lets the
  single-agent-vs-team ablation give both sides the same tool surface.

### structured-output -- forcing a model to return a typed object instead of prose
- Priority: useful
- Came up: 2026-08-22, docs/NEXT.md open question
- Status: owned
- Why it matters here: the reviewer's `Objection` has to be a parseable object for routing to work
  at all. The open question in NEXT.md (LangChain vs the Anthropic API directly) is a real fork.
- What clicked: makes sense. It is a boundary between non deterministic LLM output and callable code. Raises errors loudly. Enforces structure to the LLM output. 

### eval-baselines -- why published numbers are the ground truth and get no hand edits
- Priority: load-bearing
- Came up: 2026-08-22, CLAUDE.md rules
- Status: flagged (acted on 2026-08-31 -- the rule is now enforced by tooling, not by discipline)
- Why it matters here: findings are the product. A baseline quietly adjusted to make a run look
  good destroys the only thing this repo is for. `evals/datasets/manifest.yaml` is now generated-only
  and Edit/Write are denied there, so the no-hand-edit rule is enforced by tooling, not discipline.

### measurement-independence -- why the thing being tested cannot report its own score
- Priority: load-bearing
- Came up: 2026-08-22, adversarial review of the state contract
- Status: explained
- Why it matters here: `holdout_score` is currently written by the modeler, which also chose the
  model and did the split. Every headline number and every ablation rests on a figure the agent
  writes about itself. The fix (harness re-scores on a holdout the agents never see) also creates
  the project's best single finding: the gap between what the agent claimed and what it actually got.
- What clicked: it is a test set. The harness holds back a portion of the data as a test set that the agents don't see. 

### langgraph-reducers -- why appending to a list in LangGraph needs an annotation
- Priority: load-bearing
- Came up: 2026-08-22, adversarial review of the state contract
- Status: owned
- Covered: 2026-08-22, nodes return partial dicts and LangGraph folds each key -- replace by
  default, `operator.add` to append. History fields append; current-state fields overwrite.
- Covered: 2026-08-24, `docs/explainers/langgraph-reducers.html` -- toggles the `operator.add`
  annotation on and off on a toy run to show the trace collapsing, then shows the reverse mistake
  of adding it to `final_features` and `review_iterations`.
- Why it matters here: a node returning `{"node_trace": [event]}` without a reducer REPLACES the
  list instead of appending, so the trace and cost accounting silently collapse to whatever ran last.
- What clicked: The node returns a small dict with maybe a few fields. For each field, there is a merge rule. A reducer is this merge rule. Selecting the right merge rule is important - are we adding history, or are we making a correction? For things like tracing the cost, we definitely want to be appending to a list. We are adding history. For things like the final features, we want to be replacing because we are sometimes making corrections. 

### caught-vs-remediated -- noticing a problem and actually fixing it are different events
- Priority: load-bearing
- Came up: 2026-08-22, adversarial review of the state contract
- Status: flagged
- Why it matters here: "reviewer correctly flagged the planted leak, feature_eng ignored it, the
  model shipped with the leak anyway" is the most interesting failure this project could report,
  and the current contract cannot see it -- it looks identical to a clean catch.

### metric-direction -- higher-is-better vs lower-is-better breaks any ratio
- Priority: useful
- Came up: 2026-08-22, adversarial review of the state contract
- Status: flagged
- Why it matters here: `score_ratio = holdout / baseline` means "good" for AUC and "bad" for RMSE.
  With `metric` as a free string, a results table mixing classification and regression rows would
  be silently self-contradictory.

### mutual-information -- the one number the profiler hands the model as evidence
- Priority: load-bearing
- Came up: 2026-08-26, `src/ds_agents/nodes/profiler.py`
- Status: flagged
- Why it matters here: the profiler computes normalized mutual information between every column and
  the target, but this is not the whole evidence base as originally thought -- the prompt also carries
  every column name, and the name does most of the work. In `reissued_ids`, `member_number` (NMI
  0.0945) and `credit_score` (NMI 0.0952) are nearly identical by this number, yet the profiler flags
  the first and not the second in 3 of 3 runs, a call the numbers alone cannot support.

### two-splits -- the agents get a holdout, and the harness keeps a different one
- Priority: load-bearing
- Came up: 2026-08-26, `split_artifact` landing in the profiler
- Status: flagged
- Why it matters here: there are two holdouts in this project. `split_artifact` is pinned by the
  profiler and partitions the rows the agents were given; the harness's holdout is withheld before
  the graph ever starts. The gap between `claimed_holdout_score` and `verified_holdout_score` only
  exists because they are different data -- collapsing this to one split deletes the project's best
  finding without the tests noticing.

### protocol-not-implementation -- why nodes are handed their tools instead of importing them
- Priority: useful
- Came up: 2026-08-26, `src/ds_agents/tools/protocol.py`
- Status: flagged
- Why it matters here: nodes type-hint against a `Tools` Protocol and `graph.py` binds the real
  objects with `functools.partial`. This is what makes swapping the Phase 1 shim for the MCP server,
  testing a node against canned tool results, and running the Haiku-vs-Sonnet reviewer ablation all
  cheap, without touching `nodes/`.

### constrained-decoding -- making the model physically unable to return the wrong shape
- Priority: useful
- Came up: 2026-08-26, writing the real model client
- Status: flagged
- Why it matters here: we use constrained decoding (`messages.parse(output_format=Schema)`) rather
  than asking nicely and parsing prose. Without it, a reviewer objection missing its `columns` list
  reads identically in the results table to "the reviewer found no leaky columns" -- a much
  better-looking result than the truth. The client raises on a refusal or validation failure rather
  than retrying with a nudge, for the same reason.

### token-pricing-and-cost-accounting -- where the dollar figures in the README come from
- Priority: load-bearing
- Came up: 2026-08-26, `tools/pricing.py`
- Status: flagged
- Why it matters here: `cost_usd` is a published number that the cost-per-caught-leak headline and
  the Haiku-vs-Sonnet comparison read straight off `NodeEvent.cost_usd`, summed from token usage
  times a hand-maintained rate table. An unknown model id raises instead of pricing at zero, and
  prices are a snapshot -- a rate change invalidates old rows rather than backfilling them.

### permutation-importance -- how we ask "did the model actually lean on this column?"
- Priority: useful
- Came up: 2026-08-26, the modeler node
- Status: flagged
- Why it matters here: this is the number that catches a leak after the model is fit -- shuffle a
  column's values, re-score, and see how much worse the model gets. On the toy set the planted leak
  scores 0.367 against 0.034 for the legitimate strong feature, a 10x gap; it's computed on the
  holdout (not train, to avoid measuring memorisation) with shuffling on source columns rather than
  one-hot expansions, and it isn't SHAP.

### fit-on-train-only -- why the median gets computed on 160 rows and not 200
- Priority: load-bearing
- Came up: 2026-08-26, the feature_eng node
- Status: flagged
- Why it matters here: filling missing values with a median computed over the whole dataset lets
  holdout rows influence the training preparation, quietly inflating the score. This is the
  `contamination` objection category the reviewer is meant to catch in the pipeline's own output, so
  having it in our scaffolding would make every contamination finding unfalsifiable -- which is also
  why a missing `split_artifact` kills `feature_eng` outright rather than falling back to the full
  frame.

### loop-cap-and-verdict-derivation -- who decides the run is over, and why it is not the reviewer
- Priority: load-bearing
- Came up: 2026-08-26, the router node
- Status: flagged
- Why it matters here: the reviewer only says `pass` or `block`; the router counts rounds and turns
  a `block` at the cap into `exhausted`. Collapsing those -- letting the reviewer write the verdict,
  or scoring `exhausted` as `pass` -- would make a reviewer that ran out of patience look identical to
  one that was satisfied, which are opposite findings.

### warm-fork-sandbox -- why the sandbox forks instead of starting a new interpreter
- Priority: useful
- Came up: 2026-08-27, `mcp_server/sandbox.py`
- Status: flagged
- Why it matters here: `run_python` used to start a fresh interpreter per snippet (0.898s to import
  pandas/scikit-learn vs 0.013s bare); it now keeps one warmed worker and forks per snippet (0.04s).
  It forks rather than reusing the interpreter for isolation: a fork is a genuinely separate process,
  so nothing one snippet does to its globals or `sys.path` can reach the next snippet.

### sys-path-is-not-the-environment -- why stripping env vars did not stop the sandbox reading the repo
- Priority: load-bearing
- Came up: 2026-08-27, building the Phase 2 sandbox
- Status: flagged
- Why it matters here: `tools/local.py`'s docstring claimed a stripped environment (no `PYTHONPATH`,
  no API key) meant agent code could not see the repo -- that was wrong. An editable install writes a
  `.pth` file that puts the repo on `sys.path` regardless of environment variables, so snippets could
  `import ds_agents`. No published number was wrong (nothing handed snippets the live
  `PipelineState`), but an isolation argument phrased only in terms of environment variables is
  incomplete by construction.

### one-error-channel -- a transport with one failure type decides who gets blamed
- Priority: load-bearing
- Came up: 2026-08-27, `mcp_server/server.py` and `tools/mcp_client.py`
- Status: flagged
- Why it matters here: in-process there are two failure types that mean opposite things: a
  `ToolError` is a finding about the agent (recorded into the run), a `SandboxError` is a finding
  about us (the worker died) and should propagate rather than be recorded as the agent's mistake.
  MCP has exactly one error channel, so sent naively every broken sandbox would arrive as a refused
  tool call and inflate the published failure rate. The fix is a prefix on the message, stripped on
  the way back.

### sync-over-async-bridge -- why the MCP client owns a thread and an event loop
- Priority: useful
- Came up: 2026-08-27, `src/ds_agents/tools/mcp_client.py`
- Status: flagged
- Why it matters here: the MCP SDK is async and LangGraph calls nodes synchronously, so `MCPTools`
  runs an asyncio event loop on a private thread and every tool call blocks on a future from it. The
  connection must be opened and closed by the same coroutine on that loop, because the SDK's session
  is an anyio task group that can only be exited by the task that entered it -- closing it from the
  caller's thread doesn't disconnect, it raises, and it raises at the end of the run, after the
  numbers are computed.

### adversarial-review-loop -- one model grading another model's work, and being allowed to send it back around the graph
- Priority: load-bearing
- Came up: 2026-08-27, `src/ds_agents/nodes/reviewer.py`
- Status: flagged
- Why it matters here: this is the project's central claim under test. The reviewer is a model under
  test, not a trusted judge, which is why it cannot write its own verdict or count its own loops.
  The interesting failure mode is an unfalsifiable answer rather than a wrong one -- a claim the
  contract has no way to check against anything.

### dispositions-as-audit-trail -- why every pass records a verdict on every objection open at entry, including the one the model never says
- Priority: useful
- Came up: 2026-08-27, `src/ds_agents/state.py`
- Status: flagged
- Why it matters here: `ReviewPass.dispositions` fills every open objection the model's list left
  silent with `not_reviewed`, a label the model itself is never allowed to produce. The point is
  that silence and agreement are different events, and a record that only stores what the model
  said cannot tell them apart -- an objection the reviewer never looked at again would otherwise
  read identically to one it actively decided was fine.

### semantic-vs-statistical-leakage -- the agents flag a leak by what the column is called, not by what it correlates with
- Priority: load-bearing
- Came up: 2026-08-27, building the `claims_timing` and `reissued_ids` trap fixtures
- Status: flagged (acted on 2026-08-27 -- it is now a recorded run condition)
- Why it matters here: two fixtures were tuned so the planted column's NMI with the target sits
  exactly where a legitimate feature's does (`member_number` 0.0945 vs `credit_score` 0.0952), and
  the profiler flagged the planted one 6 of 6 live runs; renamed to `metric_a7`/`metric_b3` it was
  flagged only 1 of 3. So "the profiler caught the leak" mostly measures whether the column was
  named like a leak -- a real behavior and also an uncontrolled variable underneath every leakage
  number the project publishes.

### evidence-surface -- a reviewer can only object to what its prompt contains
- Priority: load-bearing
- Came up: 2026-08-27, deciding which trap variants were buildable
- Status: flagged
- Why it matters here: `reviewer._user_message` shows a fixed list -- feature summary, final/dropped
  features, per-candidate scores, `top_importances`, open objections -- and nothing else; it never
  sees the split manifest, the profile, mutual-information numbers, or any data rows, so a column
  silently dropped upstream is invisible to the reviewer and duplicate rows across train/holdout are
  structurally uncatchable. Correction, 2026-08-28: the reviewer's actual leakage miss was not
  evidence absence -- `top_importances` was in the prompt with the planted traps at the top all
  along, and one added prompt rule moved detection from 1/10 to 9/10. "Not shown" and "not asked"
  are different diagnoses with different fixes.

### fixture-difficulty -- a benchmark whose trap gets cleaned upstream measures nothing
- Priority: useful
- Came up: 2026-08-27, the second attempt at a trap the reviewer can see
- Status: flagged
- Why it matters here: the reviewer had been handed a clean matrix on three fixtures in a row,
  because the profiler flags the leak and `feature_eng` drops it before the reviewer ever runs --
  producing a `leakage_recall` of 0.0 that reads like a miss but is really a fixture that never
  posed the question. Tuning the fixture until the model stops catching it fails the other way:
  you'd be measuring the tuner, and the benchmark ends up fitted to whichever model it was tuned
  against. The fix is to make difficulty an explicit, recorded condition.

### controlled-ablation -- what makes two runs comparable
- Priority: load-bearing
- Came up: 2026-08-27, turning name transparency into a recorded condition
- Status: flagged
- Why it matters here: an ablation is a claim that one thing differed, and the value of the claim
  depends on how few other things did. Of three ways to build the naming arm, a rename applied at
  load time (rewriting one header line, copying every other byte through) is the only one where
  "the profiler behaved differently because of the names" has one candidate explanation -- a paired
  fixture or a duplicated manifest field would each let something else drift too.
  `test_only_the_header_line_differs` and `test_the_two_arms_produce_the_same_importances` are what
  make that assertion checkable rather than assumed. The condition also has to be recorded on the
  frozen `RunConfig` and land in `results_row()`, or two rows run under different conditions get
  silently averaged together.

### factorial-design -- why two suspected causes get tested together, not one after the other
- Priority: load-bearing
- Came up: 2026-08-28, choosing how to spend the reviewer ablation budget
- Status: flagged
- Why it matters here: going into this session the reviewer missed the leak in 12 of 12 runs, and
  there were two live explanations -- the model (Haiku) isn't strong enough, or the prompt never
  asked which column produced the score. A factorial design crosses both conditions instead of
  testing one at a time, for the same cost per run, and it separates the two: the prompt moved the
  catch rate from 1/10 to 9/10 while the model arm alone moved it from 1/10 to 0/4. Had the model arm
  run alone under the base prompt, the conclusion would have been "Sonnet is no better at catching
  leakage," which is true and profoundly misleading. The Sonnet cells are unequal-n (n=4, n=3)
  because Sonnet cost 5x the budgeted amount, which is why the model main effect is reported as
  unresolved rather than absent.

### actionable-objection -- a finding nobody can act on is indistinguishable from a wrong finding
- Priority: load-bearing
- Came up: 2026-08-28, diagnosing why 9-of-10 caught became 1-of-10 remediated
- Status: flagged
- Why it matters here: once the reviewer started noticing the leak (9/10), the pipeline still
  shipped it in 9 of 10 runs, because an objection has a lifecycle -- raised, routed, acted on,
  closed -- and two links in that chain were broken. Routing: in 2 of 3 runs the reviewer addressed
  `implausible_importance` to `modeler`, a node with no column lever at all. Closure: in 3 of 3 runs
  the reviewer never dispositioned an objection `resolved`, holding an unfalsifiable standard
  ("never validated as non-leaking, only removed") that no fix could ever satisfy. An adversarial
  reviewer needs a termination condition it can observe as much as it needs a detection rule, or
  catch rate and remediation rate come apart and only the flattering half gets measured.

### effective-target-vs-recorded-choice -- overriding a model's decision without deleting the evidence that you had to
- Priority: load-bearing
- Came up: 2026-08-28, fixing the remediation path
- Status: flagged
- Why it matters here: the reviewer kept addressing column-scoped objections to `modeler`, which has
  no column lever, so a correct catch and a hallucination produced the same results row. The fix
  (route to `feature_eng` regardless) is also how you quietly stop measuring reviewer dispatch
  judgement, so `Objection.target_node` still holds exactly what the reviewer said,
  `PipelineState.effective_target` is the one place the override applies, and `objections_rerouted`
  counts how often they disagreed. An override is only honest if you also measure its cost: the
  reroute dropped a legitimate feature (`prior_claims_12m`) in 2 of 10 runs, which is why
  `n_final_features` is on the results row -- otherwise a run that "remediated" by gutting the
  fixture reads as a clean success.

### primary-endpoint-vs-guardrail -- the headline metric of a project is not automatically the endpoint of an arm
- Priority: load-bearing
- Came up: 2026-08-28, designing the closure arm
- Status: flagged
- Why it matters here: `leakage_remediated` looked like the natural primary endpoint for the closure
  arm, but closure makes a run terminate earlier, and `claims_timing` plants two traps that often
  need two `feature_eng` passes -- so closure could honestly push remediation down for reasons that
  have nothing to do with whether closure works. The fix was splitting roles: the primary endpoint
  is what the intervention claims to do (does the reviewer close anything), the falsifier is the
  specific new harm it risks (`objections_falsely_resolved` -- closing an objection whose column is
  still in the matrix), and a guardrail is a non-inferiority bound you check hasn't collapsed, never
  quoted as success if it rises. A metric that conflates two opposite claims can't gate anything:
  `objections_open_at_end` used to merge `resolved` with `withdrawn`, so no row could answer the
  question the gate turned on.

### replication-before-attribution -- a 10-run count is not an effect until something has been run twice
- Priority: load-bearing
- Came up: 2026-08-28, confirming the sticky-drop headline
- Status: flagged
- Why it matters here: the project's largest number -- `leakage_remediated` 5/10 to 9/10, attributed
  to the sticky-drop fix -- did not survive a same-commit control: two cells running identical
  behaviour under identical config returned 9/10 and 6/10, so model nondeterminism alone moves a
  10-run count by about 3, which is most of the reported effect. The mechanism prediction still held
  (`objections_falsely_resolved` fired 3/10 in the buggy arm, 0/10 in the fixed one), so the fix does
  prevent resurrection -- what failed was the link from that event to the outcome, because the
  pipeline recovers on its own a pass later. The general lesson: a control arm's job is to be the
  counterfactual for a claim, not a design you'd ship, and a count out of 10 is not an effect until
  something has been run twice.

### binomial-variance-and-wilson-intervals -- how wide the error bars on "9 out of 10" actually are
- Priority: load-bearing
- Came up: 2026-08-29, building `evaldiff.py`
- Status: flagged
- Why it matters here: every headline this project produces is a count out of 10, which is a
  proportion estimate with an interval about half the scale wide -- a Wilson interval on 9/10 is
  [0.596, 0.982] and on 5/10 is [0.237, 0.763], and those overlap. The pooled interval alone would
  have refused the retired sticky-drop headline before any runs were spent confirming it, which is
  why `eval-diff` prints intervals rather than deltas. Replicates are still required on top of the
  interval because a Wilson interval assumes independent draws with one fixed probability, and that
  assumption is exactly what a repeat of the same cell tests.

### structured-output-repair -- asking the model again, with the reasons its last answer failed
- Priority: useful
- Came up: 2026-08-29, fixing the zero-objection `block` bug
- Status: flagged
- Why it matters here: constrained decoding guarantees the shape of a response, never its
  usefulness -- the reviewer returned well-formed `ReviewFinding`s that were dead ends
  (`claim: block` with every objection filtered away, nothing to route). Of the three fixes (repair
  in code, change the prompt, or ask the model again with its own rejection reasons), only the third
  neither hides the failure from the metric nor breaks byte-identity with already-published rows.
  The retry is capped at one: a model that answers empty twice has told you something, and the
  second empty answer is let through as the block it actually claimed.

### regression-gate-vs-baseline -- a threshold is a coin flip until something has been run twice
- Priority: load-bearing
- Came up: 2026-08-31, running the `ci` subset live for the first time
- Status: flagged
- Why it matters here: a regression gate is a threshold on a noisy measurement, and it inherits the
  measurement's noise -- model nondeterminism alone moves a 10-run count by about 3, so a threshold
  set from one cell fires on runs where nothing changed, and a gate that cries wolf gets disabled
  within a week. `leakage_remediated` at 8/10 carries a Wilson interval of [0.490, 0.943], so an
  honest threshold is roughly "fail under 4/10" -- it catches only outright breakage. This makes
  cheap deterministic checks (pytest, ruff, the toy pipeline) more valuable than an expensive
  stochastic one: the benchmark subset is better run deliberately at a decision point than
  automatically on every commit.

### instrument-contaminates-measurement -- the harness wrote a file, and the file changed what the harness recorded
- Priority: load-bearing
- Came up: 2026-08-31, reading the `commit` column of the first 30-row eval file
- Status: flagged
- Why it matters here: `commit` is computed as `git rev-parse HEAD` plus a `-dirty` suffix from
  `git status`, but the results file it's written into is untracked -- so after run 0 writes its row,
  the tree is dirty, and every run after it records `-dirty`, splitting one cell into n=1 clean and
  n=9 dirty. The measuring act changed the measurement. The fix is to read the commit once per
  invocation, before runs start changing the thing being read. The already-committed rows were not
  back-fixed -- a results file edited to say something other than what the run recorded is the one
  thing this project won't do.

### published-vs-computed-baseline -- a cited number and a comparand are different objects
- Priority: load-bearing
- Came up: 2026-08-31, building evals/datasets/manifest.yaml
- Status: flagged
- Why it matters here: `evals/datasets/manifest.yaml` records the best published AUC per dataset,
  and dividing our `verified_holdout_score` by it would be a plausible-looking error: the published
  number comes from OpenML's own 10-fold CV run by a third party, ours from a single holdout this
  repo withholds, so the ratio would attribute a protocol difference to a system difference. The
  manifest instead carries the published number as `published_reference` (informational, labelled
  with its protocol) and the baseline as a definition with no number in it, which the harness must
  fit on this repo's own train split and score on the same holdout as the run.

### complete-list-or-nothing -- an empty answer key is not an answer of zero
- Priority: load-bearing
- Came up: 2026-08-31, external datasets arriving with no planted leak
- Status: flagged
- Why it matters here: `planted_leakage_columns` is read as a complete enumeration of a dataset's
  leaks -- true by construction for fixtures, but on a real dataset the list is empty because it's
  unknown, not because the answer is none. Left alone, columns like `leakage_caught` would read that
  emptiness as fact and count every reviewer flag a false alarm, so the fix is a named gate
  (`graded_for_leakage`) that returns `None` rather than a wrong zero. Sharpened 2026-09-02: this
  rule was paid for three times (leakage grading, `baseline_separation`, `n_candidates_failed_to_fit`),
  giving the compressed version -- whenever a numerator can be zero, put its denominator on the row
  next to it, because "measured as none" and "never measured" must stay distinguishable.

### withheld-holdout-vs-cv-fold -- who drew the partition decides what it can grade
- Priority: load-bearing
- Came up: 2026-08-31, building the re-scorer
- Status: flagged
- Why it matters here: `split_artifact` is drawn by the profiler, inside the graph, so it can't
  grade the node that decided it -- a pipeline that leaked a column scores beautifully on it because
  the leak sits on both sides of the split, and cross-validation has the same problem since its
  folds are still partitions of rows the agents can see. `verified_holdout_score` differs only in
  when its rows were removed: before `run_pipeline` is called, above the tools boundary, never
  entering `$DS_DATASET`. The honest limit: the sandbox has no filesystem namespace, so this buys an
  assertable property (no file the run's store holds contains a withheld row), not real isolation.

### normalisation-is-not-a-ratio -- why `baseline_score` was deferred rather than guessed
- Priority: load-bearing
- Came up: 2026-08-31, deciding not to ship `baseline_score`
- Status: flagged (resolved 2026-08-31 -- two raw points ship, the ratio does not)
- Why it matters here: AMLB's convention is a normalisation, `(it - zero)/(unit - zero)`;
  `score_ratio` was a ratio, `verified/baseline`, and they don't agree about what 1.0 means. Worse,
  the baseline is fit on every column including a planted leak, so a pipeline that correctly drops
  the trap scores below a baseline that kept it -- `score_ratio < 1` is good behaviour on a labelled
  dataset and bad behaviour on an unlabelled one, in the same column, which can't be pooled. The
  field was deferred rather than shipped.

### reference-system-independence -- a yardstick that inherits what it measures is not a yardstick
- Priority: load-bearing
- Came up: 2026-08-31, deciding what `baseline_unit_score` is allowed to see
- Status: flagged
- Why it matters here: the baseline could have been fit on the agents' post-`feature_eng` matrix,
  which would answer a modeler question ("was the promoted model the right choice") instead of the
  thesis question ("did this team beat a reference system it never saw"). Independence has a real
  cost: the reference system sees a column the agents were supposed to drop, so on a dataset with a
  planted trap the baseline *should* beat a pipeline that behaved correctly -- that's paid for by
  suppressing the normalised column on such datasets, not by weakening the baseline.

### failure-domain-separation -- the instrument may fail without taking the measurement with it
- Priority: load-bearing
- Came up: 2026-08-31, deciding whether the baseline extends the re-scorer's snippet
- Status: flagged
- Why it matters here: the obvious build appended the baseline fits to the end of the re-scorer's
  snippet, same process -- which means a RandomForest OOM and a refit failure become the same exit
  code, and `verified_holdout_score` (the number this half of the project exists to produce) would
  be destroyed by a failure in the optional thing measured beside it. The baseline got its own
  process, timeout, and status enum (`unit_point_failed`) instead. General shape: whenever two
  computations of different importance share a failure domain, the less important one can take the
  more important one down, and a shared timeout budget also makes the same code pass on a fast
  machine and fail on a slow one.

### a-metric-floor-is-not-zero -- the bottom of a scale is measured, not assumed
- Priority: useful
- Came up: 2026-08-31, replacing `score_ratio` with `baseline_normalised_score`
- Status: flagged
- Why it matters here: the retired `score_ratio`'s guard assumed r2's baseline is exactly 0.0 -- it
  isn't, because `DummyRegressor` predicts the train mean while r2's denominator is variance about
  the holdout mean, so a constant predictor scores slightly negative on held-out rows. Measured
  rather than assumed, only roc_auc has a real constant floor (0.5); f1, accuracy, log_loss, rmse and
  mae are all dataset-dependent or artifacts. The fix is to normalise from a measured zero point,
  `(v-z)/(u-z)`, rather than assuming where the floor is.

### cost-hides-in-the-cheapest-case -- the smoke test is the one shape that cannot show you the bill
- Priority: load-bearing
- Came up: 2026-08-31, measuring what the baseline costs before `--subset full` is funded
- Status: flagged
- Why it matters here: `bench-smoke`'s cost estimate was guessed at $0.040 and measured at $0.017 --
  wrong by 2x, on the cheapest dataset in the manifest. The baseline's wall cost is 0.20s there,
  invisible against LLM latency spread, but 72s on `higgs`, roughly quadrupling a run -- a smoke test
  on the cheapest case is uninformative about cost precisely because the cheapest case has the least
  of whatever scales. Measure the cost term at the size that will actually be paid, or at two sizes
  so you can see the slope.

### silent-refusal-looks-like-a-result -- the failure that writes a row
- Priority: load-bearing
- Came up: 2026-09-01, pricing `--subset full` and discovering four datasets cannot be run
- Status: flagged
- Why it matters here: above ~36k rows the split artifact collides with the 1 MiB `read_artifact`
  cap, and `feature_eng` correctly refuses with `recoverable=False` -- but nothing in the graph or
  router read that field, so the run continued through reviewer and reporter and `publishable()`
  accepted the result. Four of thirteen benchmark datasets produced a row with no model and no
  verified score, indistinguishable at a glance from a real measurement. FIXED 2026-09-01: the graph
  now halts to the reporter on an unrecoverable error, but the row is still written, carrying
  `halted_at` -- refusing to write it would have deleted the hardest datasets from the results file
  instead.

### two-axes-treated-as-one -- when "how big is it" has more than one answer
- Priority: useful
- Came up: 2026-09-01, designing `bench-mid` as a 2x2 rather than a line
- Status: flagged
- Why it matters here: datasets had been sized by rows, but an agent pipeline pays on two
  independent axes: wall clock tracks rows (fits, permutation importance) while token cost tracks
  columns (the schema and profile get rendered into a prompt, re-rendered on every review pass).
  `jasmine` is 144 columns on 2984 rows and `nomao` is 118 columns on 34465 -- a table sampled only
  along the row axis had missed that `nomao` has more cells than `higgs`. Before extrapolating cost
  from one measurement, sample a cross of both axes, not a line along one.

### encoding-as-a-contract -- a representation decides what can be said, including what can be lied about
- Priority: load-bearing
- Came up: 2026-09-01, replacing the split manifest's explicit index lists with an assignment string
- Status: flagged
- Why it matters here: the split manifest stores one character per row and derives every partition,
  including each fold's training set, from a rule -- a fold trains on rows it doesn't validate on --
  that's true of `KFold` and false of `TimeSeriesSplit`. The old form wrote fold training rows down
  explicitly and couldn't have this bug; the new form is smaller precisely because it doesn't, so a
  correct encoder and decoder can still produce a wrong split the day the splitter changes, with
  nothing raising because nothing is broken. Compression removes redundancy, and redundancy is what
  catches errors -- the defence is to put the rule in the artifact itself (`fold_train: complement`)
  and check it against reality at write time.

### a-guard-that-stops-being-reachable -- the checks a change quietly disarms
- Priority: useful
- Came up: 2026-09-01, shrinking the split manifest below the read cap
- Status: flagged
- Why it matters here: shrinking the split manifest 40x disarmed two guards nobody mentioned in the
  change: `truncated` was a real backstop while the artifact could exceed the read cap and now never
  fires, and each consumer's `0 <= i < len(df)` filter was a real bound under explicit indices but
  became a no-op under an assignment string -- a manifest written for a different, larger frame would
  now decode silently, dropping the tail of the data everywhere while reporting a plausible row
  count. Fixing the thing a guard was watching can disarm the guard; the fix here was making
  `decode_split` require the frame length as an argument, so no caller can forget to prove it.

### a-fit-is-not-a-model -- four points, two parameters, and the residual nobody looked at
- Priority: load-bearing
- Came up: 2026-09-02, deciding whether `adult`'s $0.0158 refuted the `bench-mid` cost model
- Status: flagged
- Why it matters here: `bench-mid` fitted `cost ≈ a + b·n_features` on 4 points and 2 parameters,
  and `docs/NEXT.md` quoted the next `adult` run as "13% high" against it -- but the fit's own
  in-sample residual sd is $0.0026 on 2 degrees of freedom, and `adult`'s residual (+$0.0019) is
  smaller than the error the model already makes on the data it was fit to. The honest 95%
  prediction interval, [$0.0004, $0.0274], is wider than any cost this project has ever measured, so
  the point constrained nothing. Before calling a new observation a miss, compute what the fit
  already fails to explain -- the tell is degrees of freedom at n-p ≤ 2.

### confounded-by-what-you-did-not-vary -- the fitted set had one dtype, and nobody chose that
- Priority: load-bearing
- Came up: 2026-09-02, pricing four datasets outside the cost model's fitted range
- Status: flagged
- Why it matters here: `bench-mid`'s careful rows-by-columns 2x2 had all four datasets with zero
  categorical columns, an axis nobody was thinking about, and both out-of-sample points measured
  since were categorical and ran high, with a real mechanism (the reviewer's prompt grows with
  one-hot level count, not rows). A designed experiment controls the axes you named and silently
  confounds every axis you didn't. The hypothesis was ultimately refuted, though -- adding a
  categorical-count term to the refit made the residual worse -- which is the point worth keeping: a
  confound tells you a result is unproven, never that the alternative is true.

### is-it-an-error-or-a-decision -- two identical-looking defects, one session apart, took opposite fixes
- Priority: load-bearing
- Came up: 2026-09-02, adding `ModelResult.fit_error` before `--subset full`
- Status: flagged
- Why it matters here: the same diagnosis ("`errored` is uninformative") got opposite fixes twice in
  one day. A routine one-hot cardinality skip was reclassified out of `errors` entirely into a count
  (`n_skipped_high_cardinality`), because it was a decision, not an error; a real fit failure in
  `modeler.py` kept its `PipelineError` and got a companion column instead, because it was a genuine
  anomaly. Ask what the underlying event is before touching the column: a routine decision belongs
  in its own count, a real anomaly needs a way to say which anomaly fired. Sharpened same day: a
  third case, a zero-objection block that was retried and succeeded, showed the taxonomy has a third
  value too -- `errored` can say something happened but not whether it was handled, and that gap was
  left open rather than answered.

### the-assumption-beside-the-estimate -- the printed caveat was the whole error, and the model was fine
- Priority: load-bearing
- Came up: 2026-09-02, `--subset full` overrunning its estimate by 10.2%
- Status: flagged
- Why it matters here: `--subset full` overran its estimate by 10.2%, and the reflexive read (the
  cost model under-predicted again) was wrong: all of the miss sat in 2 of 13 cells that looped
  2-3x, and the 46 runs that reviewed exactly once landed within 0.95x-1.04x of their price. The
  error lived in one sentence in `SUBSETS` -- every price assumes a run that passes first time -- that
  had been stated correctly for two sessions and never costed. A documented assumption behaves
  exactly like an undocumented one until someone puts a number on it. The refit needed next is a
  loop-rate term, not a bigger size fit, and it was declined until there was enough looping data to
  estimate one from.

### replay-from-a-final-state -- what a single-writer contract lets you reconstruct, and what it cannot
- Priority: useful
- Came up: 2026-09-09, building the pipeline walkthrough viewer
- Status: explained
- Covered: 2026-09-09, `docs/explainers/pipeline-walkthrough.html` -- replays five captured runs node
  by node, each step labelled with a fidelity level.
- Why it matters here: the pipeline keeps no checkpointer, yet the walkthrough replays runs node by
  node because every state field has one writer and `node_trace` appends in execution order. The
  limit shows the same property from the other side: when `feature_eng` runs twice in a review loop,
  the first run's outputs are overwritten and gone, so those steps are labelled `not_recorded`
  rather than shown from the final value.

### two-statistics-one-intuition -- NMI and permutation importance agree until they do not
- Priority: load-bearing
- Came up: 2026-09-21, working out why three datasets drew every objection on the benchmark run
- Status: flagged
- Why it matters here: the profiler's mutual information is univariate and blind to redundancy; the
  reviewer's permutation importance is multivariate and measures what breaks when one column is
  shuffled. They diverge when two columns carry the same signal: shuffling either does nothing
  because the model reads the other, so importance collapses toward zero while NMI stays put. This
  is why `bank_marketing` is concentrated in importance and flat in NMI while `nomao` and `kc1` are
  NMI outliers and flat in importance -- the profiler nominated columns there and the reviewer never
  objected, because the pipeline has two leak detectors that fail on opposite data.

### a-clustered-rate-is-not-a-rate -- six events in three datasets, and why they do not average
- Priority: useful
- Came up: 2026-09-21, declining a loop-rate term for the cost estimate
- Status: flagged
- Why it matters here: 6 of 52 runs looped, which reads like an 11.5% rate that could be multiplied
  into every cell's price -- except the events are piled into 3 of 13 datasets, not spread across
  them, and a goodness-of-fit test against one constant probability rejects the pooled rate
  (chi-square(12)=27.5, p=0.0065). Averaging anyway charges the ten datasets that never loop a 17.3%
  premium while still under-pricing the three that do. A mean is only a summary when the thing being
  averaged is homogeneous, and that's testable, not a matter of taste.

### pre-registered-vs-found -- the same number means two different things depending on when you named it
- Priority: load-bearing
- Came up: 2026-09-21, the `claims-repro` run producing one of each in the same file
- Status: flagged
- Why it matters here: the `claims-repro` run produced two results with different standing from the
  same ten rows. The four endpoints written into `LOG.md` before the run are believable at face
  value, because the test was chosen without knowing the answer. The profiler's 2/10-to-8/10 shift
  was noticed afterward while scanning a table of everything that might have moved, so its p=0.027
  isn't really 0.027 -- it's the smallest of however many comparisons the eye ran over. The fix isn't
  to discard it, it's to treat it as a hypothesis for the next experiment, not a result that settles
  this one.

### the-conditions-are-part-of-the-number -- an arm mismatch reads exactly like a regression
- Priority: useful
- Came up: 2026-09-21, the walkthrough capture that appeared to contradict README Result 3
- Status: flagged
- Why it matters here: a session recorded that `claims_timing` "no longer loops," and it sat in
  NEXT.md for twelve days as the largest risk to a published number -- but it was never a
  contradiction. The capture script runs the fixture under descriptive column names and the
  published rows are under opaque ones, and the pipeline behaves completely differently between the
  two because the profiler reads names. A measurement is a number plus the conditions that produced
  it; read the invocation that produced a surprising number before pricing a run to chase it.
