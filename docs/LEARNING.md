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
```

## Entries

### langgraph-state-machine — how LangGraph turns nodes and edges into a running loop
- Priority: load-bearing
- Came up: 2026-08-22, session 0 setup
- Status: flagged
- Why it matters here: the reviewer's ability to block and route back is the whole thesis. If the
  conditional-edge and loop-cap semantics are not clear to you, you cannot tell a real "exhausted"
  verdict from a wiring bug, and the eval numbers become uninterpretable.

### target-leakage — a feature that encodes the answer, so the model scores well and is useless
- Priority: load-bearing
- Came up: 2026-08-22, session 0 setup
- Status: flagged
- Why it matters here: this is what the adversarial reviewer is being measured on. You need to be
  able to look at a planted trap and say why it is a trap, or you cannot grade the reviewer.

### mcp-tool-boundary — why tools are a separate server rather than Python functions
- Priority: useful
- Came up: 2026-08-22, docs/DECISIONS.md
- Status: flagged
- Why it matters here: it costs a phase of plumbing (PLAN.md Phase 2). The payoff is that the
  single-agent-vs-team ablation gives both sides the same tool surface. Worth understanding before
  you are tempted to skip it.

### structured-output — forcing a model to return a typed object instead of prose
- Priority: useful
- Came up: 2026-08-22, docs/NEXT.md open question
- Status: flagged
- Why it matters here: the reviewer's `Objection` has to be a parseable object for routing to work
  at all. The open question in NEXT.md (LangChain vs the Anthropic API directly) is a real fork.

### eval-baselines — why published numbers are the ground truth and get no hand edits
- Priority: load-bearing
- Came up: 2026-08-22, CLAUDE.md rules
- Status: flagged
- Why it matters here: findings are the product. A baseline quietly adjusted to make a run look
  good destroys the only thing this repo is for.

### measurement-independence — why the thing being tested cannot report its own score
- Priority: load-bearing
- Came up: 2026-08-22, adversarial review of the state contract
- Status: flagged
- Why it matters here: `holdout_score` is currently written by the modeler, which also chose the
  model and did the split. Every headline number and every ablation rests on a figure the agent
  writes about itself. The fix (harness re-scores on a holdout the agents never see) also creates
  the project's best single finding: the gap between what the agent claimed and what it actually got.

### langgraph-reducers — why appending to a list in LangGraph needs an annotation
- Priority: load-bearing
- Came up: 2026-08-22, adversarial review of the state contract
- Status: flagged
- Why it matters here: a node returning `{"node_trace": [event]}` REPLACES the list rather than
  appending, so the trace and the cost accounting silently collapse to whatever ran last. This is
  the difference between a cost table that is real and one that is quietly wrong.

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
