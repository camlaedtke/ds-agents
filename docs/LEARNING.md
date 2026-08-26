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
