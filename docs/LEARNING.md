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
### <slug> — <one-line plain-language description>
- Priority: load-bearing | useful | trivia
- Came up: <date>, <where>
- Status: flagged
- Why it matters here: <one or two sentences, project-specific, not a textbook definition>
- What clicked: optional note manually entered by me
```

## Entries

### langgraph-state-machine — how LangGraph turns nodes and edges into a running loop
- Priority: load-bearing
- Came up: 2026-08-22, session 0 setup
- Status: owned
- Why it matters here: the reviewer's ability to block and route back is the whole thesis. If the
  conditional-edge and loop-cap semantics are not clear to you, you cannot tell a real "exhausted"
  verdict from a wiring bug, and the eval numbers become uninterpretable.
- What clicked: I understand the basic state machine concept. When it gets to a node, you pull a few fields off the state, build a prompt, call LLM with a Pydantic model, maybe call a tool (i.e. run code through run_python), return a small dict, add to the state and pass it on. All defined with StateGraph. 

### target-leakage — a feature that encodes the answer, so the model scores well and is useless
- Priority: load-bearing
- Came up: 2026-08-22, session 0 setup
- Status: owned
- Why it matters here: this is what the adversarial reviewer is being measured on. You need to be
  able to look at a planted trap and say why it is a trap, or you cannot grade the reviewer.
- What clicked: totally makes sense. A garbage feature which has a very high correlation with the target. 

### mcp-tool-boundary — why tools are a separate server rather than Python functions
- Priority: useful
- Came up: 2026-08-22, docs/DECISIONS.md
- Status: explained
- Covered: 2026-08-24, `docs/explainers/mcp-tool-boundary.html` — steps one real profiler call
  (`run_python` measuring that `account_status_code` agrees with `churned` on 91% of the 200 toy
  rows) across the process boundary: the server advertising its four tools, the model writing
  arguments, the JSON-only crossing, the sandbox with `manifest.json` deliberately not mounted, and
  the typed `LeakageCandidate` the node writes from `stdout`. Then toggles to `exec()` in-process,
  where `state.planted_leakage_columns` (set at `cli.py:29`) is in reach and a `leakage_recall` of
  1.0 no longer distinguishes a reviewer that reasoned from code that read the answer key.
- Why it matters here: it costs a phase of plumbing (PLAN.md Phase 2). The payoff is that the
  single-agent-vs-team ablation gives both sides the same tool surface. Worth understanding before
  you are tempted to skip it.

### structured-output — forcing a model to return a typed object instead of prose
- Priority: useful
- Came up: 2026-08-22, docs/NEXT.md open question
- Status: owned
- Why it matters here: the reviewer's `Objection` has to be a parseable object for routing to work
  at all. The open question in NEXT.md (LangChain vs the Anthropic API directly) is a real fork.
- What clicked: makes sense. It is a boundary between non deterministic LLM output and callable code. Raises errors loudly. Enforces structure to the LLM output. 

### eval-baselines — why published numbers are the ground truth and get no hand edits
- Priority: load-bearing
- Came up: 2026-08-22, CLAUDE.md rules
- Status: flagged (acted on 2026-08-31 — the rule is now enforced by tooling, not by discipline)
- Why it matters here: findings are the product. A baseline quietly adjusted to make a run look
  good destroys the only thing this repo is for.
- Acted on 2026-08-31: `evals/datasets/manifest.yaml` is generated-only, because
  `.claude/settings.json` denies Edit and Write there. So "no baseline was hand-edited" stopped
  being a promise and became a property of the tooling. Still `flagged` because the concept has
  not been sat with — see [[published-vs-computed-baseline]] for the distinction the manifest
  turns on.

### measurement-independence — why the thing being tested cannot report its own score
- Priority: load-bearing
- Came up: 2026-08-22, adversarial review of the state contract
- Status: explained
- Why it matters here: `holdout_score` is currently written by the modeler, which also chose the
  model and did the split. Every headline number and every ablation rests on a figure the agent
  writes about itself. The fix (harness re-scores on a holdout the agents never see) also creates
  the project's best single finding: the gap between what the agent claimed and what it actually got.
- What clicked: it is a test set. The harness holds back a portion of the data as a test set that the agents don't see. 

### langgraph-reducers — why appending to a list in LangGraph needs an annotation
- Priority: load-bearing
- Came up: 2026-08-22, adversarial review of the state contract
- Status: owned
- Covered: 2026-08-22, nodes return partial dicts and LangGraph folds each key — replace by
  default, `operator.add` to append. The four history fields append; current-state fields like
  `final_features` overwrite on purpose.
- Covered: 2026-08-24, `docs/explainers/langgraph-reducers.html` — steps a toy run node by node with
  the `operator.add` annotation on and off, so the discarded trace and the wrong `total_cost_usd`
  are visible, then shows the reverse mistake: `operator.add` on `final_features` inverts
  `leakage_remediated`, and on `review_iterations` turns `loop_cap=3` into 2. This closes the open
  check from the first pass — `review_iterations` is current truth with a single owner (the router),
  not history, and a reducer would give the reviewer a share of its own cap.
- Why it matters here: a node returning `{"node_trace": [event]}` REPLACES the list rather than
  appending, so the trace and the cost accounting silently collapse to whatever ran last. This is
  the difference between a cost table that is real and one that is quietly wrong.
- What clicked: The node returns a small dict with maybe a few fields. For each field, there is a merge rule. A reducer is this merge rule. Selecting the right merge rule is important - are we adding history, or are we making a correction? For things like tracing the cost, we definitely want to be appending to a list. We are adding history. For things like the final features, we want to be replacing because we are sometimes making corrections. 

### caught-vs-remediated — noticing a problem and actually fixing it are different events
- Priority: load-bearing
- Came up: 2026-08-22, adversarial review of the state contract
- Status: flagged
- Why it matters here: "reviewer correctly flagged the planted leak, feature_eng ignored it, the
  model shipped with the leak anyway" is the most interesting failure this project could report,
  and the current contract cannot see it — it looks identical to a clean catch.

### metric-direction — higher-is-better vs lower-is-better breaks any ratio
- Priority: useful
- Came up: 2026-08-22, adversarial review of the state contract
- Status: flagged
- Why it matters here: `score_ratio = holdout / baseline` means "good" for AUC and "bad" for RMSE.
  With `metric` as a free string, a results table mixing classification and regression rows would
  be silently self-contradictory.

### mutual-information — the one number the profiler hands the model as evidence
- Priority: load-bearing
- Came up: 2026-08-26, `src/ds_agents/nodes/profiler.py`
- Status: flagged
- Why it matters here: the profiler computes normalized mutual information between every column
  and the target, and that number is the entire *numeric* evidence base the model gets for deciding
  what to flag. On the toy fixture the planted leak reads 0.518, the legitimate strong feature 0.094,
  and the meaningless id column 0.194. If you cannot read those three numbers and say which pattern
  is which, you cannot tell a profiler that reasoned well from one that guessed and got lucky, and
  the leakage_recall column stops being interpretable. The distinction it forces — high
  association is not the same as leakage — is the whole task.
- Correction, 2026-08-27: this entry used to say the number was "the entire evidence base", and that
  is wrong in a way that matters. The prompt also carries every column *name*, and the name is doing
  most of the work. `member_number` reads 0.0945 and `credit_score` 0.0952 in `reissued_ids`, and the
  profiler flags the first and not the second in 3 of 3 runs — a decision the numbers cannot support.
  See [[semantic-vs-statistical-leakage]] and DECISIONS.md 2026-08-27.

### two-splits — the agents get a holdout, and the harness keeps a different one
- Priority: load-bearing
- Came up: 2026-08-26, `split_artifact` landing in the profiler
- Status: flagged
- Why it matters here: there are now two holdouts in this project and they are easy to conflate.
  `split_artifact` is pinned by the profiler and partitions the rows the agents were given; the
  harness's holdout is withheld before the graph ever starts. The gap between
  `claimed_holdout_score` and `verified_holdout_score` only exists because they are different
  data. Anyone who "simplifies" this to one split deletes the project's best finding without the
  tests noticing. Builds on [[measurement-independence]], which you already own.

### protocol-not-implementation — why nodes are handed their tools instead of importing them
- Priority: useful
- Came up: 2026-08-26, `src/ds_agents/tools/protocol.py`
- Status: flagged
- Why it matters here: nodes type-hint against a `Tools` Protocol and a `StructuredModel`
  Protocol, and `graph.py` binds the real objects in with `functools.partial`. This is what makes
  three separate things cheap: swapping the Phase 1 shim for the MCP server without touching
  `nodes/`, testing a node against canned tool results in milliseconds, and running the
  Haiku-vs-Sonnet reviewer ablation as a config change rather than an edit. It also answers the
  structured-output fork in one file instead of six. Related: [[structured-output]],
  [[mcp-tool-boundary]].

### constrained-decoding — making the model physically unable to return the wrong shape
- Priority: useful
- Came up: 2026-08-26, writing the real model client
- Status: flagged
- Why it matters here: there are two ways to get JSON out of a model. Ask nicely in the prompt and
  parse what comes back, or hand the API a schema it enforces while generating. We use the second
  (`messages.parse(output_format=Schema)`). The difference shows up in the eval, not the code: if a
  reviewer objection can come back missing its `columns` list, "the model could not fill in the
  field" and "the reviewer found no leaky columns" become the same row in the results table, and
  the second one is a much better-looking result than the truth. The client raises on a refusal or
  a validation failure rather than retrying with a nudge, for the same reason.

### token-pricing-and-cost-accounting — where the dollar figures in the README come from
- Priority: load-bearing
- Came up: 2026-08-26, `tools/pricing.py`
- Status: flagged
- Why it matters here: `cost_usd` is a published number — the cost-per-caught-leak headline and the
  Haiku-vs-Sonnet comparison are both read straight off `NodeEvent.cost_usd`, which is summed from
  the API's own reported token usage times a rate table we maintain by hand. Two consequences worth
  understanding before trusting any cost number: an unknown model id raises instead of pricing at
  zero (a $0.00 row is indistinguishable from a genuinely cheap one), and prices are a snapshot, so
  a rate change invalidates old rows rather than being backfilled.

### permutation-importance — how we ask "did the model actually lean on this column?"
- Priority: useful
- Came up: 2026-08-26, the modeler node
- Status: flagged
- Why it matters here: this is the number that catches a leak *after* the model is fit. Shuffle one
  column's values, re-score, and see how much worse the model gets; a column the model leaned on
  hard makes the score collapse. On the toy set the planted leak scores 0.367 and the legitimately
  strong feature 0.034 — a 10x gap, which is what makes `implausible_importance` a real judgement
  call for the reviewer rather than a threshold. Two details that are not cosmetic: it is computed
  on the holdout, not on the training rows (permuting training rows measures memorisation, not
  usefulness), and the feature transform sits inside the sklearn pipeline so the shuffling happens
  on *source* columns, not one-hot expansions. It is not SHAP; the field is named
  `importance_artifact` for that reason.

### fit-on-train-only — why the median gets computed on 160 rows and not 200
- Priority: load-bearing
- Came up: 2026-08-26, the feature_eng node
- Status: flagged
- Why it matters here: filling missing values with a median computed over the whole dataset means
  the holdout rows helped choose the number used to prepare the training rows. The model has then
  seen a smudge of its own test set, and the score comes out slightly too good for reasons nothing
  in the code says out loud. This is the `contamination` objection category — one of the failures we
  are asking the reviewer to catch in *its own* pipeline — so having it in our scaffolding would
  make every contamination finding unfalsifiable. It is also why a missing `split_artifact` kills
  `feature_eng` outright instead of falling back to the full frame.

### loop-cap-and-verdict-derivation — who decides the run is over, and why it is not the reviewer
- Priority: load-bearing
- Came up: 2026-08-26, the router node
- Status: flagged
- Why it matters here: the reviewer says `pass` or `block` and nothing else. The router counts the
  rounds and turns a `block` at the cap into `exhausted`. Collapsing those — letting the reviewer
  write the verdict, or scoring `exhausted` as `pass` — would make a reviewer that ran out of
  patience look identical in a results table to one that was satisfied, and those are opposite
  findings. The cap is enforced by the router and the graph edge together rather than by a guard
  inside the router, deliberately: see docs/DECISIONS.md 2026-08-26.

### warm-fork-sandbox — why the sandbox forks instead of starting a new interpreter
- Priority: useful
- Came up: 2026-08-27, `mcp_server/sandbox.py`
- Status: flagged
- Why it matters here: `run_python` used to start a whole new Python for each snippet, and each
  one paid 0.898s to import pandas and scikit-learn against 0.013s for a bare interpreter. The
  sandbox now keeps one worker process that has already done those imports and calls `fork()` for
  each snippet, so the child starts with the libraries already in memory and costs 0.04s. The part
  that is worth understanding rather than trusting is *why forking rather than just reusing the
  interpreter*: a fork produces a genuinely separate process, so nothing the profiler's snippet did
  to its globals, its imported modules, or `sys.path` can reach the modeler's. Reusing one
  interpreter would save the same second and quietly make runs depend on what ran before them.
  Related: [[mcp-tool-boundary]], [[protocol-not-implementation]].

### sys-path-is-not-the-environment — why stripping env vars did not stop the sandbox reading the repo
- Priority: load-bearing
- Came up: 2026-08-27, building the Phase 2 sandbox
- Status: flagged
- Why it matters here: `tools/local.py` handed each snippet an environment built from scratch — no
  `PYTHONPATH`, no API key — and its docstring said agent code therefore could not see the repo it
  is being graded in. That was wrong, and the new sandbox tests prove it: an editable install
  writes a `.pth` file into site-packages that puts `/ds-agents` and `/ds-agents/src` on `sys.path`
  at startup, and a `.pth` is read by the interpreter regardless of the environment. Snippets could
  `import ds_agents`. Nothing handed them the live `PipelineState`, so no number published so far
  is wrong — but "the agent read the grading code" and "the agent reasoned about the columns" are
  the same `leakage_recall` in a results table, and that is the distinction this whole project
  exists to make. The transferable idea is that an isolation argument phrased entirely in terms of
  environment variables is incomplete by construction, because import paths, the current working
  directory, and inherited file descriptors are all reachable without an env var. Builds on
  [[mcp-tool-boundary]].

### one-error-channel — a transport with one failure type decides who gets blamed
- Priority: load-bearing
- Came up: 2026-08-27, `mcp_server/server.py` and `tools/mcp_client.py`
- Status: flagged
- Why it matters here: in-process there are two failure types and they mean opposite things. A
  `ToolError` is a finding about the agent — it asked for an artifact that does not exist, or its
  snippet was rejected — and a node catches it and writes a recoverable `PipelineError` into the
  run. A `SandboxError` is a finding about us: the worker died, the machine is broken, and it
  propagates rather than being recorded, because a run that failed for our reasons is not a data
  point about the agent. MCP has exactly one failure channel: an error result carrying a string.
  Sent naively, every broken sandbox would arrive as a refused tool call and land in the results
  table as the agent's mistake, quietly inflating whatever failure rate we publish. The fix is
  small (a prefix on the message, stripped on the way back) and the transferable idea is not: any
  time a boundary has fewer error types than the code on either side of it, the collapse happens
  silently and shows up as a wrong number rather than as an exception. Related:
  [[measurement-independence]], [[caught-vs-remediated]].

### sync-over-async-bridge — why the MCP client owns a thread and an event loop
- Priority: useful
- Came up: 2026-08-27, `src/ds_agents/tools/mcp_client.py`
- Status: flagged
- Why it matters here: the MCP SDK is async and LangGraph calls our nodes synchronously, so
  `MCPTools` runs an asyncio event loop on a private thread and every tool call blocks on a
  future handed back from it. The part worth understanding rather than trusting is why the
  connection is opened *and closed* by one long-lived coroutine on that loop instead of by the
  calling thread: the SDK's session is an anyio task group, and a task group must be exited by
  the same task that entered it. Closing it from the caller's thread does not disconnect, it
  raises — and it raises at the end of the run, after the numbers are computed, which is the
  worst possible time to discover a lifetime bug. Related: [[protocol-not-implementation]].

### adversarial-review-loop — one model grading another model's work, and being allowed to send it back around the graph
- Priority: load-bearing
- Came up: 2026-08-27, `src/ds_agents/nodes/reviewer.py`
- Status: flagged
- Why it matters here: this is the project's central claim under test. The reviewer is a model
  under test, not a trusted judge — which is exactly why it cannot write its own verdict or count
  its own loops, and why the router exists at all. The interesting failure mode is not the reviewer
  giving a wrong answer, it's giving an unfalsifiable one: a claim the contract has no way to check
  against anything. Related: [[measurement-independence]], [[caught-vs-remediated]].

### dispositions-as-audit-trail — why every pass records a verdict on every objection open at entry, including the one the model never says
- Priority: useful
- Came up: 2026-08-27, `src/ds_agents/state.py`
- Status: flagged
- Why it matters here: `ReviewPass.dispositions` fills every open objection the model's list left
  silent with `not_reviewed`, a label the model itself is never allowed to produce. The point is
  that silence and agreement are different events, and a record that only stores what the model
  said cannot tell them apart — an objection the reviewer never looked at again would otherwise
  read identically to one it actively decided was fine.

### semantic-vs-statistical-leakage — the agents flag a leak by what the column is called, not by what it correlates with
- Priority: load-bearing
- Came up: 2026-08-27, building the `claims_timing` and `reissued_ids` trap fixtures
- Status: flagged (acted on 2026-08-27: it is now a recorded run condition, see [[controlled-ablation]])
- Why it matters here: two fixtures were tuned so the planted column's association with the target
  sits exactly where a legitimate strong feature sits — `member_number` at 0.0945 against
  `credit_score` at 0.0952 — and the profiler flagged the planted one anyway, 6 of 6 live runs. The
  same CSV with the trap columns renamed to `metric_a7` and `metric_b3` was flagged in 1 of 3. So
  when we say "the profiler caught the leak" we are mostly measuring whether the column was named
  like a leak. That is not a bug — a real data scientist reads names too — but it is an
  uncontrolled variable sitting underneath every leakage number the project plans to publish, and it
  means a benchmark of "hard" traps can be built or defeated by renaming columns. Related:
  [[mutual-information]], [[target-leakage]], [[fixture-difficulty]].

### evidence-surface — a reviewer can only object to what its prompt contains
- Priority: load-bearing
- Came up: 2026-08-27, deciding which trap variants were buildable
- Status: flagged
- Why it matters here: `reviewer._user_message` puts seven things in front of the model — the spec,
  `feature_summary`, `final_features`, `dropped_features`, per-candidate `cv_mean` and claimed score,
  `top_importances`, open objections, and optionally the feature code. That list is the reviewer's
  entire world. It never sees the split manifest, the profile, the mutual-information numbers, or a
  single row of data. Two consequences that look like model failures and are not: a column the
  feature snippet skipped as high-cardinality is invisible, so the reviewer cannot object to it; and
  duplicate rows across the train/holdout boundary are structurally uncatchable, because nothing in
  the prompt could distinguish a contaminated split from a clean one. Before grading the reviewer on
  a failure, check whether the evidence for it was in the prompt at all. Related:
  [[adversarial-review-loop]], [[measurement-independence]], [[caught-vs-remediated]].
- Sharpened 2026-08-28: the reviewer's leakage miss is **not** an instance of this. `top_importances`
  was in the prompt all along, ranked, with the planted traps at the top of it, and one appended
  rule asking which column explains an implausible score moved Haiku from 1 of 10 to 9 of 10. "Was
  not shown" and "was not asked" are different diagnoses with different fixes, and this entry only
  covers the first. Related: [[factorial-design]].

### fixture-difficulty — a benchmark whose trap gets cleaned upstream measures nothing
- Priority: useful
- Came up: 2026-08-27, the second attempt at a trap the reviewer can see
- Status: flagged
- Why it matters here: the reviewer has now been handed a clean matrix on three fixtures in a row,
  because the profiler flags the leak and `feature_eng` drops it before the reviewer is called. A
  fixture like that produces a `leakage_recall` of 0.0 for the reviewer that reads exactly like a
  miss and is actually a fixture that never posed the question. The tempting fix is to tune the
  fixture until the model stops catching it, and that fails in the other direction: you are then
  measuring the tuner, and the benchmark is fitted to whichever model it was tuned against, which
  quietly corrupts the Phase 5 ablations it exists to support. The way out is to make difficulty an
  explicit, recorded condition rather than a property of how hard the author tried. Related:
  [[semantic-vs-statistical-leakage]], [[eval-baselines]].

### controlled-ablation — what makes two runs comparable
- Priority: load-bearing
- Came up: 2026-08-27, turning name transparency into a recorded condition
- Status: flagged
- Why it matters here: an ablation is a claim that one thing differed. The value of the claim is
  entirely in how few other things did. Three shapes were available for the naming arm and they are
  not equally good: a *paired fixture* (two CSVs, one descriptive and one opaque) would differ in
  the names and also in whatever the second generator's noise drew; a *manifest field* would mean
  maintaining two committed files that are supposed to be identical and eventually will not be; a
  *rename applied at load time* rewrites one line of one file and copies every remaining byte
  through, so the arms differ in the header row and provably nowhere else. Only the third makes
  "the profiler behaved differently because of the names" a statement with one candidate
  explanation. The assertion that buys this is not a comment, it is
  `test_only_the_header_line_differs` and its sibling `test_the_two_arms_produce_the_same
  _importances`: identical rows must produce identical permutation importances, and if they ever
  stop doing so, something other than the names moved and every number in the arm is suspect. The
  second half is bookkeeping and matters just as much: the condition has to be recorded on the
  frozen `RunConfig` and land in `results_row()`, because two rows that ran under different
  conditions and do not say so are worse than no rows -- they will be averaged together by someone
  who has forgotten, including by us. Related: [[semantic-vs-statistical-leakage]],
  [[fixture-difficulty]], [[measurement-independence]], [[eval-baselines]].

### factorial-design — why two suspected causes get tested together, not one after the other
- Priority: load-bearing
- Came up: 2026-08-28, choosing how to spend the reviewer ablation budget
- Status: flagged
- Why it matters here: going into this session the reviewer had missed a leak in 12 of 12 runs where
  it had one in front of it, and there were two live explanations -- Haiku is not strong enough to
  see it, or nothing in the prompt ever asked it which column produced the score. The obvious plan
  was to run the model arm (Haiku versus Sonnet) because `--reviewer-model` already existed. That
  plan had a hole: whichever number it produced would have been conditional on whichever prompt
  happened to be in `reviewer.py` that week, the prompt was not a recorded condition, and so the
  dependency would have been invisible to anyone reading the results file six weeks later. A
  *factorial* design crosses both conditions -- every combination of model and prompt -- instead of
  varying one and holding the other at whatever it happened to be. It costs the same per run, and it
  buys three answers instead of one: each main effect (does the model matter? does the prompt
  matter?) and the interaction (does the prompt help only the stronger model?). It also gives you a
  built-in validity check, because everything upstream of the manipulated node is identical across
  all four cells, so `profiler_recall` and trap survival must agree cell to cell -- if they do not,
  something other than the reviewer moved and no comparison in the table means anything. Here the
  answer was lopsided: the prompt moved the catch rate from 1/10 to 9/10 and the model moved it
  from 1/10 to 0/4. Had the model arm run alone under the base prompt it would have concluded
  "Sonnet is no better at catching leakage", which is true and profoundly misleading. Two practical
  notes. Decide the decision rule before seeing the numbers -- ours was ">=5/10 difference is shown,
  3-4 is directional, less is not resolved at this n" -- because a rule chosen afterwards is a
  rule chosen to fit. And cells must be equal-n by intent: ours were not, because Sonnet cost 5x
  what was budgeted and the pre-registered spending cap bound the Sonnet half to n=4 and n=3, which
  is why the model main effect is reported as unresolved rather than as absent. Related:
  [[controlled-ablation]], [[evidence-surface]], [[caught-vs-remediated]].

### actionable-objection — a finding nobody can act on is indistinguishable from a wrong finding
- Priority: load-bearing
- Came up: 2026-08-28, diagnosing why 9-of-10 caught became 1-of-10 remediated
- Status: flagged
- Why it matters here: the reviewer's job looks like "notice the leak", and once it started noticing
  the leak the pipeline still shipped it in 9 runs out of 10. Three live diagnostic runs show the
  reason is not that the reviewer was wrong and not that it ran out of turns. It is that an
  objection has a *lifecycle* -- raised, routed, acted on, closed -- and two separate links in that
  chain were broken, each of which is invisible in a results table that only records what was
  raised. First, **routing**: in 2 of 3 runs the reviewer raised `implausible_importance` and
  addressed it to `modeler`, which is a defensible reading of its own prompt (the objection is about
  a score) and is also a node with no column lever at all -- every candidate is fit on the one
  frozen transform `feature_eng` already produced. `feature_eng` force-drops an objected column
  before its model is even consulted, but only for objections addressed to it, so a correct
  objection sent one node sideways produces exactly the same table row as a hallucinated one.
  Second, **closure**: in 3 of 3 runs the reviewer never dispositioned a single objection
  `resolved`. The most instructive run dropped both traps, watched the claimed score fall from 0.986
  to 0.823, wrote that the drop "is consistent with removing leakage" -- and marked the objection
  `still_open` anyway, on the grounds that the columns "were never validated as non-leaking, only
  removed". That is an unfalsifiable standard, and a reviewer holding one can never let a run pass,
  so `exhausted` stops meaning "the fix did not land". The general lesson is that an adversarial
  reviewer needs a *termination condition* as much as it needs a detection rule: something it can
  observe that discharges its own objection. Without one, catch rate and remediation rate come apart
  and only the flattering half is easy to measure. Related: [[caught-vs-remediated]],
  [[adversarial-review-loop]], [[loop-cap-and-verdict-derivation]],
  [[dispositions-as-audit-trail]], [[evidence-surface]].

### effective-target-vs-recorded-choice — overriding a model's decision without deleting the evidence that you had to
- Priority: load-bearing
- Came up: 2026-08-28, fixing the remediation path
- Status: flagged
- Why it matters here: the reviewer kept addressing column-scoped objections to `modeler`, a node
  with no column lever, so a correct catch produced the same results row as a hallucination. The
  fix is obvious — route those to `feature_eng` regardless — and the obvious fix is also how you
  quietly stop measuring the thing you built the project to measure. The move that keeps both is to
  separate the *recorded* choice from the *effective* one: `Objection.target_node` still holds
  exactly what the reviewer said and is never rewritten, `PipelineState.effective_target` is the
  single place the override is applied, and `objections_by_target_node` keeps counting the raw
  field so the reviewer's dispatch judgement stays measurable *in the arm that overrides it*.
  `objections_rerouted` beside it counts how often the graph disagreed. That is the difference
  between a recorded condition and a thumb on the scale, and it is checkable rather than assertable:
  `test_the_raw_target_node_survives_the_reroute` fails if anyone folds the effective target into
  the counter. Two corollaries worth as much as the main idea. First, applying the condition in one
  place is not tidiness — the router and `feature_eng` had been two independent answers to "who acts
  on this", and under the new arm they would have disagreed, with the router sending a run to
  `modeler` while `feature_eng` was the only node that could act. Second, an override is only honest
  if you also measure what it costs: `by_category` converts a reviewer false positive from something
  inert into a really dropped column, and it dropped `prior_claims_12m` — a legitimate strong
  feature — in 2 of 10 runs. `n_final_features` exists on the results row for exactly that reason,
  because `leakage_remediated` is `None` on an empty matrix but `True` on a one-column one, so a run
  that "remediated" by gutting the fixture would otherwise read as a clean success. Related:
  [[actionable-objection]], [[controlled-ablation]], [[caught-vs-remediated]],
  [[measurement-independence]].

### primary-endpoint-vs-guardrail — the headline metric of a project is not automatically the endpoint of an arm
- Priority: load-bearing
- Came up: 2026-08-28, designing the closure arm
- Status: flagged
- Why it matters here: `leakage_remediated` is what this project is about, so it was the natural
  primary endpoint for the closure arm -- and it was the wrong one, for a reason you can only see by
  thinking about the mechanism first. Closure makes a run terminate *earlier*; `claims_timing`
  plants two traps and four of the control cell's runs needed two `feature_eng` returns, so a
  reviewer that resolves after the first drop never reaches the pass in which it would have named
  the second. Closure could honestly push remediation *down*, while the sticky-drop fix in the same
  commit pushed it *up*. Pre-registering a number whose direction you cannot predict is
  pre-registering a coin flip: whichever way it lands you will have a story, which is another way of
  saying it tests nothing. The fix is to split the roles. The **primary endpoint** is the thing the
  intervention is claimed to do (here: does the reviewer close anything). The **falsifier** is the
  specific new harm the intervention risks (here: `objections_falsely_resolved` -- closing an
  objection whose column is still in the matrix, which is what "buying termination by teaching the
  reviewer to say fixed" would look like in a column). A **guardrail** is a metric you do not expect
  to improve and are checking has not collapsed -- pre-registered as a non-inferiority bound
  (">=4/10 acceptable, <=3/10 kills the arm"), never quoted as success if it happens to rise.
  Three corollaries this session paid for. First, a **pre-registered stopping rule** is worth as
  much as a pre-registered threshold: the control cell was written up as a decision gate ("if the
  reviewer already resolves honestly, do not run the arm") and it closed, which saved the arm's
  budget and turned a non-result into a recorded null instead of a quiet abandonment. Second,
  **check the premise against data you already have before funding the arm** -- eight of ten
  committed rows already closed an objection, which contradicted the diagnosis in NEXT.md that the
  whole arm was designed against, and that re-read cost nothing. Third, **a metric that conflates
  two opposite claims cannot gate anything**: `objections_open_at_end` merged `resolved` ("the fix
  landed") with `withdrawn` ("I was wrong"), so no committed row could answer the question the gate
  turned on, and splitting them was a prerequisite rather than a nicety. Related:
  [[caught-vs-remediated]], [[controlled-ablation]], [[effective-target-vs-recorded-choice]],
  [[measurement-independence]], [[dispositions-as-audit-trail]].

### replication-before-attribution — a 10-run count is not an effect until something has been run twice
- Priority: load-bearing
- Came up: 2026-08-28, confirming the sticky-drop headline
- Status: flagged
- Why it matters here: the project's largest number -- `leakage_remediated` 5/10 -> 9/10, attributed
  to the sticky-drop fix -- did not survive a same-commit control. The paired cell came back at 7/10
  against 6/10, a difference of -1. The reason is not subtle and it is the thing worth keeping: two
  cells running **identical behaviour under an identical config** returned 9/10 and 6/10. Model
  nondeterminism alone moves a 10-run count on `claims_timing` by about 3, which is most of the
  effect the original comparison reported. Nothing about that was visible from either cell alone,
  and it was not discoverable by thinking harder about the design -- only by running the same thing
  twice. It also applies backwards: the routing arm's pre-registered 1/10 -> 5/10 is the same size
  of difference and now carries the same error bars.
  Three things follow that generalise past this repo. **A control arm's job is to be the
  counterfactual for a claim, not a design you would ship.** "The fix moved remediation 5/10 -> 9/10"
  is a claim *about the buggy pipeline*, which makes the buggy pipeline the denominator; the instinct
  that measuring a known bug "would measure nothing" is right about design and wrong about evidence,
  and it is what DECISIONS.md 2026-08-28 (fourth entry) got wrong and the fifth overturned.
  **A mechanism prediction is what separates a replicated number from a confirmed cause** -- and here
  it separated an *unreplicated* number from a *real* mechanism, which is the more instructive
  split. The fingerprint fired exactly as predicted (`objections_falsely_resolved` 3/10 in the buggy
  arm, 0/10 in the fixed one): the resurrection is real and the fix does prevent it. What failed was
  the link from the event to the outcome, because the pipeline recovers on its own -- the reviewer
  re-objects to the re-admitted column on a later pass, so resurrection costs a loop, not a result.
  Without the mechanism endpoints the cell would have read as "the fix does nothing", which is
  false. **And a metric built for one purpose is often the sharpest instrument for another**:
  `objections_falsely_resolved` was built to catch a *dishonest reviewer* under the closure arm, and
  it turned out to be the cleanest available detector of an *honest* reviewer whose pipeline
  un-fixed itself.
  The bookkeeping half matters as much: the condition had to go on the frozen `RunConfig` and into
  `results_row()`, and its default is the one in this repo that is deliberately NOT the old
  behaviour, because the old behaviour is a defect rather than a design fork -- pinned by a test with
  the reasoning in its docstring, since it reads as an inconsistency and tidying it would silently
  ship the bug. Related: [[controlled-ablation]], [[primary-endpoint-vs-guardrail]],
  [[caught-vs-remediated]], [[effective-target-vs-recorded-choice]], [[measurement-independence]],
  [[eval-baselines]].

### binomial-variance-and-wilson-intervals — how wide the error bars on "9 out of 10" actually are
- Priority: load-bearing
- Came up: 2026-08-29, building `evaldiff.py`
- Status: flagged
- Why it matters here: every headline this project has produced is a count out of 10, and a count
  out of 10 is a *proportion estimate* with an interval about half the scale wide. A Wilson interval
  is the standard way to put bounds on one: 9/10 is [0.596, 0.982] and 5/10 is [0.237, 0.763]. Those
  overlap, which means **the pooled interval alone would already have refused the retired sticky-drop
  headline** -- the tool that costs nothing would have caught what $0.54 of live runs caught, before
  the runs. That is the practical payoff and it is why `eval-diff` prints intervals rather than
  deltas. The subtler half is why replicates are still required on top of the interval. A Wilson
  interval assumes the 10 runs are independent Bernoulli draws with one fixed success probability.
  That assumption is exactly what a repeat of the same cell tests, and it is not obviously true here:
  a model whose behaviour drifts, a fixture whose difficulty depends on one split, or a bug that
  fires in bursts would all break it, and a broken assumption makes the interval a lie rather than
  merely wide. So the replicate requirement is not about shrinking the interval -- it is about
  earning the right to compute one. That is why `Count` carries `per_replicate` beside the pooled
  number: if between-replicate spread ever exceeds what binomial noise allows, the pooled interval
  has to be thrown out rather than narrowed. Related: [[replication-before-attribution]],
  [[controlled-ablation]], [[primary-endpoint-vs-guardrail]], [[eval-baselines]].

### structured-output-repair — asking the model again, with the reasons its last answer failed
- Priority: useful
- Came up: 2026-08-29, fixing the zero-objection `block` bug
- Status: flagged
- Why it matters here: constrained decoding guarantees the *shape* of a response, never its
  *usefulness*. The reviewer returned well-formed `ReviewFinding`s that were nonetheless dead ends:
  `claim: "block"` with every objection filtered away, so nothing could be routed anywhere. There
  are only three things you can do about that -- repair it in code (the node overrules the model,
  and the failure disappears from the metric that was supposed to measure it), change the prompt
  (which breaks byte-identity with every row already published, and here would only restate a rule
  the model already had), or hand the model its own rejection reasons and ask once more. The third
  is the only one that neither hides the failure nor invalidates the archive, and it pairs with the
  2026-08-27 lesson that a per-item rule on a response schema is really a whole-response rule: the
  filter that rejects one bad objection is also what leaves the claim stranded, so the repair has to
  happen after filtering, not in the schema. Two details are load-bearing in the implementation. The
  correction travels *inside* the JSON facts block, because `_payload` finds the block with
  `find("{")`/`rfind("}")` and anything after it reaches nobody. And the retry is capped at one:
  a model that answers empty twice has told you something, and the second empty answer is allowed
  through to the router as the block the model actually claimed. Related: [[structured-output]],
  [[constrained-decoding]], [[actionable-objection]], [[measurement-independence]].

### regression-gate-vs-baseline — a threshold is a coin flip until something has been run twice
- Priority: load-bearing
- Came up: 2026-08-31, running the `ci` subset live for the first time
- Status: flagged
- Why it matters here: PLAN.md has wanted a CI gate since Phase 4 opened, and the reason it kept not
  happening is the useful part. A regression gate is a *threshold on a noisy measurement*, so it
  inherits every property of the measurement -- and on this repo's own evidence, model
  nondeterminism alone moves a 10-run count by about 3. A threshold set from one cell therefore
  fires on runs where nothing changed. That is not a small annoyance: a gate that cries wolf gets
  disabled or clicked through within a week, at which point it is strictly worse than no gate,
  because the repo now believes it has coverage it does not have. The order is forced: measure the
  baseline, replicate it to find out how much it moves on its own, and only then set a line far
  enough outside that spread to mean something. This session bought step one and two of three -- and
  the numbers say a useful gate here has to be wide. `leakage_remediated` at 8/10 carries a Wilson
  interval of [0.490, 0.943], so an honest threshold on that column is roughly "fail under 4/10",
  which catches a pipeline that has broken outright and nothing subtler. The second half of the
  lesson is that this makes a *cheap deterministic* gate more valuable than an expensive stochastic
  one: `pytest`, `ruff`, and the toy pipeline running green catch real breakage per push at zero
  dollars and zero false alarms, while the benchmark subset is better run deliberately, at a
  decision point, than automatically on every commit. Related: [[eval-baselines]],
  [[replication-before-attribution]], [[binomial-variance-and-wilson-intervals]],
  [[primary-endpoint-vs-guardrail]].

### instrument-contaminates-measurement — the harness wrote a file, and the file changed what the harness recorded
- Priority: load-bearing
- Came up: 2026-08-31, reading the `commit` column of the first 30-row eval file
- Status: flagged
- Why it matters here: `commit` exists so that rows pooled into one cell provably came from one tree,
  and it is computed as `git rev-parse HEAD` plus a `-dirty` suffix when `git status` is non-empty.
  The results file the harness writes is untracked. So run 0 recorded a clean hash, run 0's row
  landed on disk, and every run after it recorded `-dirty` -- **the act of measuring changed the
  measurement, and the field designed to guarantee comparability is what broke it**, splitting one
  cell into n=1 and n=9. The general shape is worth carrying: any instrument that writes into the
  environment it observes will eventually observe its own writing, and the failure is invisible
  because every individual value is *correct*. Run 17's tree really was dirty. Nothing was buggy in
  the sense of computing a wrong answer; the design was wrong about *when* to ask. The fix is the
  same one that already applied to `materialize` a few lines away -- read once per invocation,
  before the runs start changing the thing being read -- which is the tell that the rule was already
  known here and simply not applied twice. Two smaller lessons ride along. **An n=1 smoke test
  structurally cannot find this**: with one run there is no second read to disagree with the first,
  so the 2026-08-29 write-path proof was clean and gave false confidence. And the committed rows
  were **not** back-fixed, because a results file edited to say something other than what the run
  recorded is the one thing this project will not do -- the defect is documented in place instead.
  Related: [[measurement-independence]], [[controlled-ablation]], [[eval-baselines]],
  [[caught-vs-remediated]].

### published-vs-computed-baseline — a cited number and a comparand are different objects
- Priority: load-bearing
- Came up: 2026-08-31, building evals/datasets/manifest.yaml
- Status: flagged
- Why it matters here: `evals/datasets/manifest.yaml` records, per dataset, the best AUC anyone has
  uploaded to that OpenML task. It is tempting to divide this pipeline's `verified_holdout_score`
  by it and call the result `score_ratio` — the field already exists and would populate. That would
  be wrong, and wrong in the worst way: it produces a plausible number rather than an error. The
  published score came from OpenML's own 10-fold cross-validation run by a third-party flow; ours
  will come from a single holdout this repo withholds. The ratio would attribute the difference
  between two *protocols* to the difference between two *systems*, and no amount of replication
  would reveal it, because the bias is constant. So the manifest carries the published number as
  `published_reference` (informational, labelled with its protocol on the value itself) and carries
  the baseline separately as a *definition* with no number in it — AMLB's constant class-prior
  predictor and tuned RandomForest — which the harness must fit on this repo's own train split and
  score on the same holdout as the run. A comparand has to be measured alongside the thing it is
  compared to. The rule is enforced by an AST test rather than a comment, because the failure it
  prevents is silent. Related: [[eval-baselines]], [[measurement-independence]], [[two-splits]],
  [[controlled-ablation]].

### complete-list-or-nothing — an empty answer key is not an answer of zero
- Priority: load-bearing
- Came up: 2026-08-31, external datasets arriving with no planted leak
- Status: flagged
- Why it matters here: `planted_leakage_columns` is read as a *complete* enumeration of a dataset's
  leaks. On a fixture that is true by construction — the generator writes the manifest as it writes
  the CSV. On a real dataset nobody has enumerated anything, and the list is empty because it is
  unknown, not because the answer is none. Left alone, nine results-row columns read that emptiness
  as fact: `leakage_caught` says the reviewer missed something, and every column the reviewer
  flagged is counted a false alarm. Both are claims with no evidence, and both pool straight into a
  published rate. The fix is one named gate (`graded_for_leakage`) returning `None` — not measured
  — which is what `leakage_remediated` and the `*_recall` columns already did for the same reason.
  The general shape: whenever a denominator can be empty, decide whether empty means *zero* or
  *unknown*, and make the type say which. The same question is why `known_leakage` in the manifest
  is a documented observation and never ground truth — one cited leak in `bank_marketing` does not
  make the list complete, and pretending it does would turn every other suspicious column into a
  scored mistake. Related: [[eval-baselines]], [[caught-vs-remediated]], [[fixture-difficulty]],
  [[measurement-independence]].
- Sharpened 2026-09-02: this repo has now paid for the rule three times, and the third payment
  named the cheap version of it. `leakage_graded` was the gate on an empty answer key;
  `baseline_separation` was the denominator behind a quotient that could approach zero;
  `n_candidates` is the denominator behind `n_candidates_failed_to_fit`, because a run halted at
  `feature_eng` writes a row with no candidates at all and its zero numerator would otherwise read
  as "every candidate fit fine" on a run where none was ever attempted. The compressed rule:
  **whenever a numerator can be zero, put the denominator on the row next to it.** It costs one
  integer, it is derived from state that already exists, and it is the difference between "measured
  as none" and "never measured". Note what the three have in common -- each failure is invisible,
  because each produces a perfectly plausible number rather than a missing one. Related:
  [[is-it-an-error-or-a-decision]].

### withheld-holdout-vs-cv-fold — who drew the partition decides what it can grade
- Priority: load-bearing
- Came up: 2026-08-31, building the re-scorer
- Status: flagged
- Why it matters here: this pipeline already had a holdout before this session, and it could not be
  used to grade anything. `split_artifact` is drawn by the profiler, using a strategy intake chose,
  over the frame the agents were mounted on — every part of it decided inside the graph. A split a
  node decided cannot grade the node that decided it, and the failure is not hypothetical: the
  modeler reports `claimed_holdout_score` against exactly that partition, and a pipeline that
  leaked a column into its features scores beautifully there because the leak is present on both
  sides of it. Cross-validation has the same property and is not a fix — 5 pinned folds inside the
  train split are five more partitions of rows the agents can see. What makes
  `verified_holdout_score` a different kind of number is only *when* its rows were removed: before
  `run_pipeline` was called, above the tools boundary, never entering `$DS_DATASET` or the run's
  artifact store. The general shape: an evaluation split is only independent of a decision if it
  existed before the decision was made, and "the model never saw these rows" is a claim about
  chronology, not about row counts. Note the honest limit recorded alongside it — the sandbox has no
  filesystem namespace, so what the layout buys is an assertable property (no file the run's store
  holds contains a withheld row) rather than isolation. Related: [[two-splits]],
  [[measurement-independence]], [[published-vs-baseline]], [[caught-vs-remediated]].

### normalisation-is-not-a-ratio — why `baseline_score` was deferred rather than guessed
- Priority: load-bearing
- Came up: 2026-08-31, deciding not to ship `baseline_score`
- Status: flagged (resolved 2026-08-31 — two raw points ship, the ratio does not; see [[reference-system-independence]])
- Why it matters here: AMLB's convention is a *normalisation* — score a system as
  `(it − zero) / (unit − zero)`, where zero is a constant class-prior predictor and unit is a tuned
  RandomForest. `PipelineState.score_ratio` is a *ratio*: `verified / baseline`. Those are not the
  same function and they do not agree about what "1.0" means, which is the first thing to know
  before the field is populated. The second thing is worse and is the actual reason the work was
  deferred: the baseline is fit on every column, **including a leak**. On a dataset with a planted
  trap, a pipeline that correctly drops the trap scores *below* a baseline that kept it — so
  `score_ratio < 1` is evidence of good behaviour on a labelled dataset and of bad behaviour on an
  unlabelled one, in the same column, with nothing on the row to distinguish them. A headline
  column that inverts its meaning depending on a property of the dataset cannot be pooled, and
  discovering that after the numbers exist is how a results table becomes wrong. The general shape:
  before computing a normalised score, ask what the reference system was allowed to see, and
  whether the thing being measured was allowed to see the same. Related: [[published-vs-baseline]],
  [[eval-baselines]], [[complete-list-or-nothing]], [[two-splits]].

### reference-system-independence — a yardstick that inherits what it measures is not a yardstick
- Priority: load-bearing
- Came up: 2026-08-31, deciding what `baseline_unit_score` is allowed to see
- Status: flagged
- Why it matters here: the baseline could have been fit on the agents' post-`feature_eng` matrix.
  It would have been cheaper, reused machinery that already exists, and produced a perfectly
  reasonable-looking number. It would also have answered a different question. A RandomForest on
  the agents' own columns asks "was the promoted model the right choice", which is a *modeler*
  question; a RandomForest on the raw frame asks "did this team beat a reference system", which is
  the thesis. The independence is what makes the second question askable at all, and independence
  is expensive in exactly one way: the reference system now sees a column the agents were supposed
  to drop, so on a dataset with a planted trap the baseline **should** beat a pipeline that behaved
  correctly. That is not a bug to be tuned away, it is the price of the property, and it is paid by
  suppressing the normalised column on such datasets rather than by weakening the baseline. The
  general shape: when you build a comparand, the first question is not "is it strong enough" but
  "what was it allowed to see, and was the thing being measured allowed to see the same". A
  comparand that saw *less* is unfair; one that saw *more* is unfair in the other direction and
  much harder to notice, because it makes your own system look worse and nobody audits a
  disappointing number. Related: [[normalisation-is-not-a-ratio]], [[published-vs-computed-baseline]],
  [[measurement-independence]], [[eval-baselines]], [[two-splits]].

### failure-domain-separation — the instrument may fail without taking the measurement with it
- Priority: load-bearing
- Came up: 2026-08-31, deciding whether the baseline extends the re-scorer's snippet
- Status: flagged
- Why it matters here: the obvious build was to add the two baseline fits to the end of
  `RESCORE_SNIPPET` — same process, same sandbox call, one JSON line out. It works, and it has one
  property that disqualifies it: a RandomForest that runs out of memory on a 98k-row frame and a
  refit that fails are then **the same process exit code**. `verified_holdout_score` is the number
  this half of the project exists to produce, and it would be destroyed by a failure in the
  optional thing measured beside it. So the baseline got its own process, its own timeout, and its
  own status enum, and `unit_point_failed` is a real value that keeps `baseline_zero_score`
  alongside it. The reasoning generalises past sandboxes: whenever two computations of different
  importance share a failure domain — one process, one transaction, one request, one try block —
  the less important one can take the more important one down, and no amount of care inside the
  code changes that, because the sharing is what does it. The tell that you have this problem is
  having to *reconstruct* which half failed from tagged output; if you are parsing your own logs to
  recover a distinction, the distinction should have been structural. There is a second, quieter
  reason here too: one timeout budget covering two fits makes the same code return `ok` on a fast
  machine and `snippet_failed` on a slow one, which is a reproducibility bug with no wrong line in
  it. Related: [[instrument-contaminates-measurement]], [[reference-system-independence]],
  [[complete-list-or-nothing]].

### a-metric-floor-is-not-zero — the bottom of a scale is measured, not assumed
- Priority: useful
- Came up: 2026-08-31, replacing `score_ratio` with `baseline_normalised_score`
- Status: flagged
- Why it matters here: the retired `score_ratio` divided by a baseline and carried a guard whose
  comment said "the predict-the-mean baseline for r2 is exactly 0.0". It is not. A
  `DummyRegressor(strategy="mean")` predicts the **train** mean, while r2's denominator is the
  holdout's variance about the **holdout** mean, so a constant predictor scores slightly *negative*
  on held-out rows. The number was small enough that nothing would have crashed and wrong enough
  that a ratio built on it would have been quietly meaningless. Measured rather than recalled, the
  floors are: roc_auc exactly 0.5 (every pair is a tie, so this doubles as a correctness assertion
  on the whole grading chain — positive class, scorer sign, row selection); f1 exactly 0.0 when
  positive is the minority, which is an artifact of never predicting positive rather than a floor;
  accuracy the majority-class rate; log_loss, rmse and mae all non-zero and dataset-dependent. Only
  one of those is a constant. The fix is not a better guard, it is a better *function*: subtracting
  a measured zero point, `(v − z) / (u − z)`, needs no assumption about where the floor is, and it
  is direction-invariant for free because both differences flip sign together on a lower-is-better
  metric. The general shape: "normalise by the baseline" and "normalise from the baseline" sound
  alike and only the second one survives a metric whose floor is not zero.
  Related: [[normalisation-is-not-a-ratio]], [[eval-baselines]], [[reference-system-independence]].

### cost-hides-in-the-cheapest-case — the smoke test is the one shape that cannot show you the bill
- Priority: load-bearing
- Came up: 2026-08-31, measuring what the baseline costs before `--subset full` is funded
- Status: flagged
- Why it matters here: this project now has the same lesson twice from two different quantities.
  `bench-smoke`'s cost estimate was guessed at $0.040 and measured at $0.017 — wrong by more than a
  factor of two, on the cheapest dataset in the manifest. Then the baseline shipped, and its wall
  cost on that same dataset is **0.20 seconds**, which is invisible against a ~4s run-to-run spread
  in LLM latency; on `higgs` the identical code takes **72 seconds**, roughly quadrupling a run. In
  both cases the smoke test was green, correct, and completely uninformative about the thing it
  was about to authorise spending on — not because the measurement was sloppy but because the
  cheapest case is chosen precisely for having the least of whatever scales. A smoke test answers
  "does the path work", and it is worth running for that. It cannot answer "what will this cost",
  and the tell is that the cost term is *superlinear or absent* at the small size: a RandomForest is
  roughly `n log n` in rows and linear in trees, so a 100x row count is not a 100x anything you can
  read off the small run. The general shape: measure the cost term at the size that will actually be
  paid, or at two sizes so you can see the slope — one point on a curve is not an estimate, it is a
  number. Related: [[instrument-contaminates-measurement]], [[eval-baselines]],
  [[failure-domain-separation]], [[reference-system-independence]].

### silent-refusal-looks-like-a-result — the failure that writes a row
- Priority: load-bearing
- Came up: 2026-09-01, pricing `--subset full` and discovering four datasets cannot be run
- Status: flagged
- Why it matters here: the profiler writes the train/test split as a JSON artifact holding every row
  index, and `read_artifact` caps every read at 1 MiB. Above roughly 36k rows those two facts
  collide, and `feature_eng` correctly refuses to fit on a partial split — it raises with
  `recoverable=False`, which is exactly the right call. The problem is what happens next: *nothing
  in the graph or the router reads `recoverable`*. The run continues through reviewer and reporter,
  spends a full run's worth of tokens, and `publishable()` accepts the result. So four of the
  thirteen benchmark datasets do not fail loudly — they produce a **row**, with no model, no
  verified score, and a shape indistinguishable at a glance from a real measurement. Nobody noticed
  because nobody had run one.
  The general shape, and it is the thing worth carrying: **a component can refuse correctly and
  still produce a wrong system**, whenever the refusal is recorded somewhere nothing downstream
  consults. The guard was written, the guard fired, the guard was ignored. This is the same class of
  defect as a caught exception that is logged and swallowed, and it is *more* dangerous in an eval
  harness than in an application, because an application shows a user a broken screen while a
  harness quietly appends a line to the file you will later compute an average over. The defence is
  not a better guard, it is asking of every refusal: who reads this, and what do they do differently?
  If the answer is "nothing", the refusal is a comment.
  FIXED 2026-09-01, and how it was fixed is half the lesson: the graph now halts to the reporter on
  an unrecoverable error, but the row is still WRITTEN, carrying `halted_at`. Refusing to write it
  would have deleted exactly the hardest datasets from the results file, which is the same failure
  wearing the opposite coat. See docs/DECISIONS.md 2026-09-01 (third entry).
  Related: [[failure-domain-separation]], [[cost-hides-in-the-cheapest-case]],
  [[status-column-beside-the-number]], [[a-guard-that-stops-being-reachable]].

### two-axes-treated-as-one — when "how big is it" has more than one answer
- Priority: useful
- Came up: 2026-09-01, designing `bench-mid` as a 2x2 rather than a line
- Status: flagged
- Why it matters here: this project had been sizing datasets by rows, because rows are what
  `n_rows` says and what a RandomForest fit scales with. But an agent pipeline pays on two
  independent axes: **wall clock tracks rows** (fits, permutation importance) and **token cost
  tracks columns** (the schema, the per-column profile, the drop plan all get rendered into a
  prompt, and the feature prompt is re-rendered on every review pass). `credit_g` is small on both,
  so one measurement of it constrains neither. The manifest contains shapes that separate cleanly —
  `jasmine` is 144 columns on 2984 rows, `nomao` is 118 columns on 34465 — and a table sampled only
  along the row axis had missed that `nomao` has *more cells* than `higgs`.
  The general shape: before extrapolating a cost from one measurement, ask how many things the word
  "bigger" could mean for this workload, and sample a **cross** rather than a line. Two points on
  one axis look like a trend right up until the second axis moves. The cheap tell is a wide-and-short
  case: if it is expensive, the expensive thing is not length.
  Related: [[cost-hides-in-the-cheapest-case]], [[eval-baselines]].

### encoding-as-a-contract — a representation decides what can be said, including what can be lied about
- Priority: load-bearing
- Came up: 2026-09-01, replacing the split manifest's explicit index lists with an assignment string
- Status: flagged
- Why it matters here: the new split manifest stores one character per row and derives every
  partition from it, including each fold's training set. That derivation rests on a rule -- a fold
  trains on the train rows it does not validate on -- which is true of `KFold` and false of
  `TimeSeriesSplit`. The old form could not have this bug, because it wrote the fold's training rows
  down. The new form is smaller precisely because it does not, and the price is that a *correct*
  encoder plus a *correct* decoder can still produce a wrong split the day the splitter changes:
  implement `temporal`, and the same code silently hands later folds training rows from the future.
  Nothing raises, because nothing is broken -- the assumption just stopped being true.
  The general shape: **compression is the removal of redundancy, and redundancy is what catches
  errors.** Every field you derive instead of storing is a rule you are now responsible for keeping
  true, somewhere far from where it is applied. The defence is to put the rule IN the artifact
  (`fold_train: "complement"`), refuse any other value, and check the claim against reality at write
  time — which turns an assumption into a recorded, falsifiable property. The same instinct is why
  the manifest carries `counts` it does not need and a `sha256` nobody has to read.
  Related: [[eval-baselines]], [[a-guard-that-stops-being-reachable]],
  [[status-column-beside-the-number]].

### a-guard-that-stops-being-reachable — the checks a change quietly disarms
- Priority: useful
- Came up: 2026-09-01, shrinking the split manifest below the read cap
- Status: flagged
- Why it matters here: the split manifest was fixed by making it 40x smaller. Two guards stopped
  working as a side effect, and neither was mentioned in the change that broke them. `truncated` was
  a real backstop while the artifact could exceed the 1 MiB read cap; now it can't, so it never
  fires. Each consumer filtered row ids with `0 <= i < len(df)`, which was a real bound while the
  manifest listed indices; under an assignment string every position is in range by construction, so
  the filter is now a no-op — and a manifest written for a *different, larger* frame would decode
  silently, dropping the tail of the data from training and holdout and every fold at once while
  reporting a perfectly plausible row count.
  The general shape: **fixing the thing a guard was watching can disarm the guard**, and the loss is
  invisible because nothing fails. Before shipping a change that makes a failure mode unreachable,
  ask which existing checks were only working *because* of it. Here the answer was to make the
  replacement deliberate: `decode_split` takes the frame length as a required argument, so no caller
  can forget to prove it is decoding against the frame it actually read.
  Related: [[silent-refusal-looks-like-a-result]], [[encoding-as-a-contract]].

### a-fit-is-not-a-model — four points, two parameters, and the residual nobody looked at
- Priority: load-bearing
- Came up: 2026-09-02, deciding whether `adult`'s $0.0158 refuted the `bench-mid` cost model
- Status: flagged
- Why it matters here: `bench-mid` fitted `cost ~= a + b * n_features` on four measurements, and
  this project immediately began quoting new runs against it — `docs/NEXT.md` recorded the one
  `adult` smoke run as "**13% high**", and that phrasing is what set the next session's agenda. The
  fit has 4 points and 2 parameters — OLS of cost on `n_features` over the four `est_cost_usd`
  values `bench-mid` shipped, which is what the pre-registration actually used, and is now
  re-derived every run by `tests/test_cost_model.py` rather than quoted. Its own in-sample residuals
  are −$0.0008 / +$0.0003 / +$0.0028 / −$0.0023, a residual sd of **$0.0026 on 2 degrees of
  freedom**, and `adult`'s residual is
  **+$0.0019** — *smaller than the error the model already makes on the data it was fitted to*. A
  textbook 95% prediction interval at 14 columns is `[$0.0004, $0.0274]`, a span wider than any cost
  this project has ever measured. So the honest report was never "the model missed"; it was "the
  model is under-determined and this point constrains nothing."
  The general shape: **before calling a new observation a miss, compute what the fit already fails
  to explain.** A model quoted without its residual scale converts noise into findings, and it does
  so in the direction people want, because a discrepancy is a story and agreement is not. The cheap
  tell is degrees of freedom: at `n − p ≤ 2` the interval is set by the `t` multiplier (4.30 here)
  rather than by the data, and the right output is a band and a decision rule stated in advance, not
  an interval nobody can act on. This is [[replication-before-attribution]] applied to a continuous
  quantity instead of a count — and note the eventual finding needed *five* points and a sign test
  to become real, not a sharper reading of the first one.
  One practical corollary this project had to learn twice: a residual scale that lives in prose is
  not checkable, and the first thing `tests/test_cost_model.py` did on being written was find that
  `amazon_employee_access` was priced $0.000007 *below* its own measured mean, against a rule the
  comment above it had stated correctly for two sessions.
  Related: [[replication-before-attribution]], [[two-axes-treated-as-one]],
  [[cost-hides-in-the-cheapest-case]].

### confounded-by-what-you-did-not-vary — the fitted set had one dtype, and nobody chose that
- Priority: load-bearing
- Came up: 2026-09-02, pricing four datasets outside the cost model's fitted range
- Status: flagged
- Why it matters here: `bench-mid` was designed as a careful 2x2 crossing rows with columns, and it
  did separate those two axes — that part worked. But all four datasets it selected (`phoneme`,
  `jasmine`, `amazon_employee_access`, `nomao`) have **zero categorical columns**, so `feature_eng`
  one-hot encodes nothing on any of them and `LEVELS` in the emitted transform is empty in all four.
  Nobody controlled for that, because dtype was not one of the two axes anyone was thinking about.
  Both out-of-sample points measured since were categorical and both ran high, and there was a real
  mechanism to explain it: the reviewer reads the transform artifact into its prompt, and that
  artifact grows with one-hot level count rather than with rows.
  The general shape: **a designed experiment controls the axes you named and silently confounds
  every axis you did not**, and "we ran a crossed design" is no protection against a factor that
  never entered the design. The defence is a pre-run inventory — list what else varies across the
  cells, and check whether it is correlated or anti-correlated with what you are testing. Here it
  was anti-correlated by luck, which is what made the arm interpretable at all; had rows and
  categoricals moved together, the same $0.30 would have settled nothing.
  **And the ending is the part worth keeping.** The hypothesis was refuted: adding a
  categorical-count term to the refit made the residual *worse*. A plausible mechanism, a real
  confound in the fitted set, and prior evidence pointing the same way — and it was still not the
  missing term. Finding a confound tells you a result is *unproven*, never that the alternative is
  true. The confound is a reason to run the experiment, not a substitute for running it.
  Related: [[two-axes-treated-as-one]], [[a-fit-is-not-a-model]], [[encoding-as-a-contract]],
  [[replication-before-attribution]].

### is-it-an-error-or-a-decision — two identical-looking defects, one session apart, took opposite fixes
- Priority: load-bearing
- Came up: 2026-09-02, adding `ModelResult.fit_error` before `--subset full`
- Status: flagged
- Why it matters here: both defects arrive wearing the same sentence -- "`errored` is uninformative"
  -- and the repair is opposite in each case, so the sentence is not the diagnosis. On 2026-09-02
  `feature_eng` was recording a routine one-hot cardinality skip as a `PipelineError`, and the fix
  was to move the fact **out** of `errors` entirely: the skip became `n_skipped_high_cardinality`, a
  count of a decision, and `errored`'s definition (`bool(self.errors)`) never changed. Hours later
  the same sentence pointed at `modeler.py`, where a candidate that will not fit raises a
  `PipelineError` whose text never reaches a column -- and there the fix was to leave the error
  exactly where it is and add the column **beside** it. Doing either repair to the other defect is a
  published wrong number in a predictable direction: reclassify a real fit failure as a decision and
  a genuine anomaly silently leaves every failure rate this project reports; add a companion column
  to the cardinality skip and `errored` stays true on four healthy `adult` runs while looking
  better-instrumented than before.
  The general shape: **"the column is uninformative" is a symptom, and it has two causes with two
  cures.** Ask what the underlying EVENT is before touching the column. If the event is a routine
  decision the code makes on purpose, it was never an error and belongs in its own count. If the
  event is a real anomaly, the error is correct and what is missing is a way to ask which anomaly
  fired. The tell is a question: would a reader who saw this event want the run's failure rate to go
  up? The audit that made this decidable is worth as much as the rule -- all twenty-two
  `PipelineError` sites were read, exactly one was reclassified, and that ratio is what makes this a
  rule about a site rather than a licence to sweep errors into counts.
  Related: [[complete-list-or-nothing]], [[silent-refusal-looks-like-a-result]],
  [[instrument-contaminates-measurement]], [[caught-vs-remediated]].
