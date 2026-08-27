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
- Status: flagged
- Why it matters here: findings are the product. A baseline quietly adjusted to make a run look
  good destroys the only thing this repo is for.

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
  and the target, and that number is the entire evidence base the model gets for deciding what to
  flag. On the toy fixture the planted leak reads 0.518, the legitimate strong feature 0.094, and
  the meaningless id column 0.194. If you cannot read those three numbers and say which pattern is
  which, you cannot tell a profiler that reasoned well from one that guessed and got lucky, and
  the leakage_recall column stops being interpretable. The distinction it forces — high
  association is not the same as leakage — is the whole task.

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
