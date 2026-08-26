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

## 2026-08-26: The model client calls the Anthropic SDK directly, not through LangChain.

NEXT.md put the burden of proof on going direct, on the grounds that LangChain would buy LangSmith
tracing and the model-swap ablation for free and `langgraph` was already a dependency. Checking
rather than assuming settled it: `langsmith 0.11.1` and `langchain-core 1.6.0` are *already
installed transitively via langgraph*, so LangSmith tracing was never LangChain's to sell us — the
graph is traced already, and `@traceable` on one method adds the LLM spans. The model-swap ablation
is one string either way. That leaves the differences that do matter, both favouring direct:
`client.messages.parse(output_format=Schema)` is constrained decoding, a schema the server enforces
during generation, whereas `with_structured_output` on ChatAnthropic routes through tool-calling and
hopes the result validates; and `response.usage` arrives verbatim rather than normalized into
LangChain's own shape. The second point is the deciding one. Token counts are a *published output*
of this project — the cost-per-caught-leak headline and the Haiku-vs-Sonnet ablation are read
straight off `NodeEvent.cost_usd` — so a translation layer between us and the API's own numbers is
a liability, not a convenience. Cost: one dependency (`anthropic`), and we write the ~40-line
adapter ourselves. The `StructuredModel` Protocol meant this was a one-file change, as designed.

## 2026-08-26: An unpriced model raises rather than costing zero.

`tools/pricing.py` holds published per-million-token rates and refuses to price a model it does not
know. The tempting alternative is to return 0.0 and carry on, which fails silently and in the
flattering direction: nothing downstream can distinguish a genuinely cheap run from an unpriced one,
and a cost table that reads low is exactly the kind of wrong number this project exists to catch in
other systems. `AnthropicModel.__post_init__` calls `price_for` at construction, so an unpriceable
model fails before a benchmark run spends money rather than after it has produced rows nobody can
use. Cache-read and cache-write tiers are carried even though Phase 1 sends no `cache_control`,
because the profiler and reviewer prompts repeat a large fixed block per run and costing a cached
read at the full input rate would overstate Phase 4's cost by roughly the hit rate.

## 2026-08-26: The router blocks on any open objection; `severity` is recorded and unread.

Open since session 0, and forced by the router landing. Blocking only on `high` would fold the
reviewer's severity *calibration* into its *detection* score: a reviewer that correctly spots the
planted leak but rates it `medium` would count as a miss, and no later analysis could separate the
two failures. So the router loops on any standing objection, and `severity` stays on the contract,
written and never read, so a Phase 5 ablation can re-run with a high-only threshold and measure
calibration as its own axis. The cost is a handful of prompt tokens per objection and the risk that
a nitpicky reviewer burns the loop cap on trivia — which is itself a finding worth having, not noise
to suppress.

## 2026-08-26: The loop cap is enforced by the router and the edge together, not by a clamp.

The router increments `review_iterations` before testing it against `loop_cap`, so `loop_cap=3`
permits three reviewer invocations and two returns upstream, and a `block` at the cap becomes
`exhausted` rather than `pass` — a reviewer that gives up at the cap must not look like one that
passed on merit. `route_target()` then reads that written verdict, not the reviewer's claim, and
sends an `exhausted` run to the reporter. Nothing inside `router` prevents a caller invoking it a
fourth time; it would happily report `review_iterations=4`. That is deliberate. A defensive clamp
would let a mis-wired `graph.py` keep looping while reporting a compliant number, converting a
topology bug into a quietly wrong results row. As written, a wiring bug shows up as a wrong
`review_iterations` next to a wrong loop count, which is visible.

## 2026-08-26: `shap_artifact` is now `importance_artifact`.

It holds sklearn permutation importances and always would have; `shap` is not a dependency and is
not worth adding for a skeleton whose importances nothing yet scores. The rename cost two lines and
a doc cell because there are no readers and no committed results files. Keeping the old name would
have left a field that lies to every future reader — including a Phase 3 reviewer prompt that
renders field names, and the Phase 5 writeup — which is the same defect as a mislabelled split
manifest, and this repo has already decided that one. The artifact itself also carries
`"method": "permutation_importance"`, `n_repeats`, `scoring`, and `evaluated_on`, so it stays
self-describing regardless of what the field is called.

## 2026-08-26: One vocabulary for column names, and it is the source names.

`final_features`, `dropped_features`, `top_importances`, and `Objection.columns` all hold original
column names, never one-hot expansions. This is not tidiness. `results_row()` computes
`leakage_remediated` as `not (planted & set(final_features))`, and `planted_leakage_columns` holds
source names — so putting `account_status_code=CLOSED_R2` in `final_features` empties the
intersection and scores **every run as remediated even with the leak fully present**. A defect that
fails only in the flattering direction, invisible in every test that does not check it specifically,
and it now has one. The alternative fix — teaching `results_row()` to prefix-match — was rejected:
substring matching against ground truth is what the 2026-08-22 "anything the eval counts is a field"
entry forbids, and it breaks the first time a dataset has both `region` and `region_code`. Getting
source-level importances correctly falls out of putting the feature transform *inside* the sklearn
`Pipeline`, so `permutation_importance` permutes raw columns natively rather than us approximating
by summing per-dummy scores.

## 2026-08-26: feature_eng emits a fitted transform as code, not a transformed table.

`feature_code_artifact` is a module: constants (train-fitted medians, one-hot levels,
`FEATURE_ORDER`) plus `transform(df) -> DataFrame`, reading no data at import. The modeler execs it
inside the sandbox rather than being handed a matrix. The reason is Phase 4: the harness re-applies
this exact artifact to a holdout the agents never saw, to produce `verified_holdout_score`. If the
modeler scored off a table while the harness scored off code, the two paths could diverge and
`holdout_claim_gap` — the project's best single finding — would silently conflate "the agent
overstated itself" with "our two transform paths disagree". Making the modeler exercise the Phase 4
code path on every toy run means a non-reappliable transform fails now, in a 60-second test, rather
than in Phase 4 against a benchmark set. A table would also be a second source of truth about what
the model was fit on, with nothing to say which one produced the score. The `exec` in the modeler's
snippet is not a breach of the never-`exec` rule: that rule protects the *node* process, whose state
holds `planted_leakage_columns`. The sandbox subprocess is precisely where agent code belongs.

## 2026-08-26: Imputation medians and one-hot levels are fitted on train rows only.

The feature transform reads `split_artifact` and computes its constants over the pinned train rows,
which is why a missing `split_artifact` is an unrecoverable error in `feature_eng` rather than a
fallback to the full frame. Fitting a median over all rows moves the agents' own holdout into the
training statistic — that is the `contamination` objection category, the thing the reviewer exists
to catch. Planting it in our own scaffolding would make every contamination finding unfalsifiable,
and would be indefensible in a writeup about other systems' leakage.

## 2026-08-26: The reporter and router call no model.

Both are deterministic. The report is not scored by the eval, so a model-written narrative is pure
token cost on every benchmark run against CLAUDE.md's cost discipline, and it would put prose where
the reviewer and the harness both read fields. The router's job is arithmetic on a counter and a
lookup on a verdict; asking a model to do it would mean the system under test decides when to stop
reviewing itself, which the 2026-08-22 router entry already ruled out. Both still accept `model` for
signature uniformity and both have a test asserting they never call it — a node quietly acquiring a
model call is a cost regression that would otherwise show up only in a bill.

## 2026-08-26: feature_eng picks columns; it does not write code.

The model returns a `FeaturePlan` — which columns to drop, with a justification attached to each
named column — and a fixed templated snippet does the transform. The model does not author arbitrary
feature code in Phase 1. Two reasons: the sandbox is a bare subprocess until Phase 2, so arbitrary
model-authored code has a much larger blast radius than a drop list; and a plan attached to named
columns is checkable against the profile, whereas a free-text summary or a code blob is not. The
node also assembles `feature_summary` itself from the snippet's output rather than letting the model
write it, because a model-authored summary can claim a drop that did not happen — the "system
reports its own grade" problem in miniature. The cost is a lower ceiling on what the pipeline can
achieve, which bounds the system's capability without changing what the eval measures. Revisit in
Phase 2 once `run_python` is genuinely isolated.

## 2026-08-26: Forced drops make remediation partly mechanical, and that is fine.

`feature_eng` force-drops the target, the id-column class, and any column named by an open
objection, before the model is consulted; model-proposed drops are added to that set and can never
remove from it. This makes `leakage_remediated` a mechanical consequence of `leakage_caught` inside
a reviewer-enabled run that loops. The two numbers stay genuinely distinct where the eval needs them
to: the reviewer-off arm, where the model may keep a profiler-flagged column; an `exhausted` run,
where the reviewer catches the leak at the cap and `feature_eng` never runs again, giving
`leakage_caught=True, leakage_remediated=False`; and objections routed to `modeler` instead. Worth
recording so a future reader seeing the two columns correlate does not mistake it for a bug.

## 2026-08-26: sklearn's `neg_*` scorer convention stops at the snippet boundary.

Found by the reviewer pass on this session's own diff, and it would have corrupted every regression
row. `SCORERS` maps `rmse` to sklearn's `neg_root_mean_squared_error`, because sklearn scorers are
uniformly greater-is-better and encode "lower error is better" by returning a negative number. That
convention was escaping the snippet, and it broke three things at once: `best_by_cv` applied
`greater_is_better` on top of the already-negated score and therefore selected the *worst* candidate
on every lower-is-better metric; the modeler's prompt told the model "false means lower is better"
while handing it negative numbers, steering a compliant model to the same wrong answer; and
`claimed_holdout_score` reached `results_row()` with the opposite sign from the harness's
`verified_holdout_score`, fabricating a `holdout_claim_gap` of roughly twice the true error. The
rule now is that every score leaving a snippet is in the metric's own natural units — an rmse of
0.30 is 0.30 — and `SIGN` inside the snippet is the single place the conversion happens. Direction
is then applied exactly once, by whoever compares two numbers.

The reason this survived seventeen unit tests is worth more than the bug: the toy fixture is binary,
and the one test that touched the direction logic hand-wrote `best_by_cv` into canned snippet output
and asserted the node echoed it. Canned JSON cannot catch a bug in the code that produces the JSON.
`tests/test_toy_pipeline.py` now runs the real `MODEL_SNIPPET` against a small regression fixture
and asserts that no score is negative and that the smallest error wins.

## 2026-08-26: the run config records the model that actually ran.

`--model sonnet` used to construct the client straight from argv while `RunConfig.default_model` and
`reviewer_model` kept their defaults, so a Sonnet-everywhere run would publish a results row
labelled `haiku`. `RunConfig` is built from the flag first and the client is constructed from the
config, so there is one source of truth. This is ARCHITECTURE.md's "every results row is
self-describing from the state object alone" rule; the fix is small but the failure would have
silently mislabelled an entire ablation.

## 2026-08-26: the graph's recursion limit scales with the loop cap.

LangGraph's default `recursion_limit` is 25 and knows nothing about `config.loop_cap`. At
`loop_cap=6` the cycle overruns it, the run dies with `GraphRecursionError`, and — crucially — no
results row is written. That deletes exactly the runs that looped the most, which biases every
published table upward. `run_pipeline` now derives the limit from `loop_cap`. Same principle as the
reporter never being allowed to fail: a dataset that vanishes from the results is worse than one
that reports badly.
