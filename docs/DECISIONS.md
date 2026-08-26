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

## 2026-08-26: Nodes receive their tools and model as arguments, bound in graph.py
A node is `node(state, *, tools, model) -> dict`, and `graph.py` binds the last two with
`functools.partial` so LangGraph still sees the one-argument function it expects. The alternative
was a module-level client constructed on import, which is what most examples do. It fails twice
here: every node test would need to monkeypatch a global, and the Haiku/Sonnet reviewer ablation
would stop being a config change. The related rule is that a node returns a NARROW dict of new
values and never the state object. Nothing in the type system prevents `return state`, and with
`operator.add` on the four history fields it would concatenate the accumulated trace onto itself
and double every event and objection.

## 2026-08-26: Intake sees the dataset through the artifact store, not run_python
ARCHITECTURE.md gave intake only `read_artifact` while requiring it to name `spec.target`, which
it cannot do without seeing the columns. The dataset is now registered as an artifact at run start
under `dataset:<dataset_id>`, carrying columns, dtypes, row count, and per-column cardinality in
its metadata. The alternative was giving intake `run_python`. Rejected because it lets the first
node in the graph execute arbitrary code before anything has been profiled, and because intake
only ever needed the schema, not the data.

## 2026-08-26: The profiler computes statistics in code and asks the model only for judgement
A fixed snippet through `run_python` produces dtypes, missing fractions, cardinality, class
balance, and each column's normalized mutual information with the target. The model sees those
numbers and nominates leakage candidates; it is never asked for a statistic. The alternative was
an agent that writes its own profiling code, which is closer to the thesis but adds cost and a
failure mode to a step where there is no judgement to exercise -- a model adds nothing to
`df.nunique()` except a way to be wrong. What stays with the model is the part the eval measures:
on the toy fixture the planted leak scores 0.518, the legitimate strong feature scores 0.094, and
the meaningless id column scores 0.194, so separating them is a real decision and not a threshold.

## 2026-08-26: The Phase 1 tool shim runs subprocesses and strips the environment
`tools/local.py` has the MCP server's signatures and none of its isolation: no container, no
memory cap, no network block. It does keep the two properties the thesis rests on. Code runs in a
subprocess rather than `exec`, because an in-process `exec` puts `PipelineState` -- including
`planted_leakage_columns` -- inside the agent's reach, and a `leakage_recall` of 1.0 would stop
distinguishing reasoning from reading the answer key. And the child gets a minimal environment
holding the dataset and artifact paths and nothing else, so agent code cannot read an API key or
the repo it is being graded in. Both are asserted in `tests/tools/test_local.py` rather than left
as comments, because Phase 2 will inherit whatever this file actually does.

## 2026-08-26: `StubModel` is a placeholder and every event it touches says so
With no API key configured, the graph runs against `StubModel`, which answers intake from column
conventions and nominates zero leakage candidates. It is deliberately bad at the thing being
measured, and it is not allowed to be quiet about it: `NodeEvent.model` reads `"stub"`, and the
CLI prints a warning that the run's numbers are not results. The alternative -- a stub with
heuristics good enough to flag the planted column -- would make the toy run look like a working
reviewer and would put leakage detection in our code rather than in the system under test. The
harness must refuse to write a results row when any event reads `"stub"`; that check does not
exist yet.

## 2026-08-26: The split the agents see is not the holdout the harness scores on
`split_artifact` is pinned by the profiler and never rewritten, and it partitions the rows the
agents were given. It is not the independent holdout behind `verified_holdout_score`; that one is
withheld before the graph starts and never appears in this manifest. Two splits sounds redundant
and is not: the agents need a holdout to make a claim about, and the claim is worthless as a grade
precisely because they chose the split. Conflating them would delete the gap between claimed and
verified, which is the project's best single finding.

## 2026-08-26: `leakage_flagged` counts what was ever raised; false alarms also get a standing count
An adversarial re-review of the Phase 0 contract found `objected_columns` scanning the append-only
objection list with no reference to `ReviewPass.dispositions`, so a reviewer that withdrew a bad
flag scored the same false alarm forever. Rather than change what `leakage_caught` means, the
results row now carries both: `leakage_flagged` / `false_alarm` are "ever raised", and
`leakage_flagged_standing` / `false_alarm_standing` drop columns whose every objection was later
resolved or withdrawn. Catching a leak at any point is a genuine catch, but taking back a false
alarm is better behaviour than leaving it standing, and one number cannot say both. The same
review found three defects that would have made published numbers wrong -- an empty feature matrix
counting as leakage remediation, `score_ratio` raising on the exactly-reachable zero denominators
(rmse 0.0, r2's predict-the-mean baseline 0.0) and taking the whole results row down with it, and
`holdout_claim_gap` unsigned so that an overclaim on rmse and one on roc_auc cancel when averaged.
All are fixed and now have branch tests, which is what was missing.

## 2026-08-26: An artifact is immutable once issued, and a mislabelled split is worse than none
Two fixes from the review of this session's own diff, both the same shape. `LocalTools` detects
snippet-written artifacts by content change rather than by filename, and copies the file under its
artifact id. Before, a snippet overwriting a name it had already written -- and `split_manifest.json`
is a fixed name -- reported nothing written while silently changing the bytes behind an id already
handed out, which makes "`split_artifact` is pinned and never rewritten" an invariant that nothing
enforces. Separately, the profiler now refuses `temporal` and `grouped` splits rather than falling
through to a random one: both are valid on `TaskSpec` and neither is implemented in the snippet, and
a manifest labelled `grouped` that was produced by a random split puts one entity on both sides of
the partition while the file claims otherwise. The alternative in each case was to keep the
convenient behaviour and write a comment. That is exactly the trade this project exists to argue
against: a number that is quietly wrong costs more than a run that stops.
