# Decisions

One paragraph each. What was decided, the alternative, why. Newest at the bottom.

## 2026-08: Reviewer is adversarial, not advisory
The reviewer can block and route back, rather than only annotating the report. Advisory would be
easier to build and never changes outcomes, which makes it impossible to measure. Blocking creates
the loop-cap problem and the "exhausted" outcome, both of which are interesting to evaluate.

## 2026-08: Tools are an MCP server, not in-process functions
Costs a phase of plumbing. Pays for itself by making the single-agent-vs-team ablation honest
(same tool surface for both) and by giving the project a real MCP server to point at.

## 2026-08: Tabular only, small model set
The project measures the system, not the model. Linear baseline, XGBoost, optionally a small
PyTorch net. Anything more is scope creep that does not change the findings.

## 2026-08: Haiku by default
Every node runs on Haiku unless an ablation says otherwise. Keeps per-run cost low enough to run
the benchmark repeatedly, and makes the Sonnet-reviewer ablation a clean comparison.

## 2026-08-22: The system under test does not report its own grade
The modeler writes `claimed_holdout_score`; the harness writes `verified_holdout_score`, scored
outside the graph on a holdout the agents never see. The alternative was to trust the modeler's
number, which is simpler and is what the first draft did. It cannot work: the same node chooses the
model, performs the split, and reports the result, so every headline number and every ablation
would rest on a figure the system writes about itself. The rewrite also adds a pinned
`split_artifact` before feature engineering runs, without which the `contamination` objection
category is unfalsifiable — the reviewer would have nothing to check against and could only object
on vibes, polluting the false-alarm rate. The gap between claimed and verified is kept as a
reported column, because a system that overstates its own score is a finding, not a bug.

## 2026-08-22: Anything the eval counts is a field, never prose
`Objection` carries `columns: list[str]`, required for leakage and contamination. The alternative
was to compute `leakage_caught` by searching the reviewer's free-text evidence for the planted
column name. That fails in both directions: an objection naming the wrong column scores as a catch,
and it makes the headline metric depend on an LLM's phrasing. Leakage is now scored as a set
comparison against harness-written ground truth, reported as precision and recall rather than two
booleans, since a binary pair cannot express "caught the real one and also flagged three clean
columns." A free-text `subcategory` sits alongside the closed category list so that a reviewer
catching something we did not anticipate is recorded rather than mislabelled.

## 2026-08-22: The router owns the loop counter and the verdict
`review_iterations` is incremented by the router, and `review_verdict` is derived there:
`reviewer_claim == "block"` at the cap becomes `exhausted`. The reviewer only states a claim. The
alternative — the reviewer maintaining its own counter and writing its own verdict — means a weak
model that emits "pass" at the cap is indistinguishable from one that passed on merit, which
destroys the outcome DECISIONS.md already called the interesting one. Relatedly, objections are
append-only and never mutated; whether one was accepted, withdrawn, or simply never mentioned again
is recorded per pass in `ReviewPass.dispositions`, so a reviewer going quiet does not look like a
reviewer agreeing.

## 2026-08-22: Run conditions are frozen onto the state object
`RunConfig` carries `run_id`, `arm`, `reviewer_enabled`, `reviewer_model`, `loop_cap`,
`reviewer_sees_code`, and `random_seed`, and cannot be modified after construction. The alternative
was harness-side bookkeeping keyed by run id. It fails because a results row then cannot be read on
its own: with the reviewer disabled the state is byte-identical to a run where the reviewer crashed
or was never wired. Freezing it also stops a node from raising its own loop cap, and gives
`log_metric(run_id, ...)` an argument it previously had no way to obtain.

## 2026-08-22: History fields are append-only via reducers
`objections`, `review_passes`, `errors`, and `node_trace` are annotated `Annotated[list[X],
operator.add]`. LangGraph merges partial node updates by replacing a field's value unless a reducer
says otherwise, so an unannotated `node_trace` would be overwritten by whichever node returned last
and the cost table would report a single node's cost as the run total. The derived numbers are
`@computed_field` rather than plain properties for the same class of reason: a property is absent
from `model_dump()`, so the harness would silently write results rows missing exactly the columns
it exists to produce. The cost is that `extra="forbid"` state cannot round-trip its own dump, since
computed fields read back as unexpected extras; `Contract.from_dump()` handles that. Keeping
`extra="forbid"` is worth the wart, because a node inventing a field should fail loudly in a
project whose subject is contract adherence.

## 2026-08-22: The toy leak is noisy and ships with a ground-truth manifest
`tests/fixtures/toy/` plants a status code that disagrees with the target on roughly 9% of rows
rather than a perfect copy, and `manifest.json` records what was planted, including the realised
agreement rate at n=200. A perfect copy would be a trivial catch that fails to discriminate between
a good reviewer and a bad one, which is the only thing the fixture is for. Tests score against the
measured rate in the manifest, never the nominal one, so sampling variance in the generator cannot
quietly change the difficulty of the task.
