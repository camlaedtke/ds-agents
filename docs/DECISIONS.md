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

## 2026-08-27: the sandbox forks from one warm worker, and Docker is not the Phase 2 backend.

The choice PLAN.md posed was Docker-per-run versus one long-lived container. Both lose to a third
option once the numbers are in front of you. A fresh interpreter costs 0.013s; a fresh interpreter
that imports pandas and scikit-learn costs 0.898s, and a toy run makes four `run_python` calls, so
imports alone were about 3.6s of a 16s run and would be multiplied by every dataset in Phase 4. The
implementation is one long-lived worker process, launched with a built-from-nothing environment,
that pays the imports once and then `fork()`s a fresh child per snippet. Measured per-call cost
including a RandomForest fit: 0.04s, against 0.898s cold — and a live toy run went from ~16s to
14.15s, the rest being Haiku latency. The worker is shared process-wide rather than per run,
because it holds no run state: the environment, the working directory, and the output paths all
ride on the request. Booting one per run cost 0.87s per unit test and would cost that per dataset
in a benchmark.

The alternative that looks equivalent and is not is a long-lived interpreter that `exec`s each
snippet in a fresh namespace. It saves the same imports and loses the property that pays for them:
globals, imported modules, monkeypatched libraries, and a mutated `sys.path` would all survive from
the profiler's snippet into the modeler's, and a run would stop being reproducible from its inputs.
Forking gives a genuinely new process per snippet, so `test_globals_do_not_survive_between_snippets`
is an assertion rather than a hope. This is also why the CLAUDE.md rule against `exec` in-process
is not violated by `_worker.py` calling `exec`: the process doing it is a forked child of a process
that has never imported `ds_agents`, so `PipelineState` is not merely out of scope, it is in another
process's memory.

Docker is deferred, not rejected, and the reason is scoped: the daemon is not even running on the
development machine, so requiring it would make `pytest -m fast` unrunnable on every edit, which is
the one test contract this repo actually enforces. What Docker buys that this does not is a memory
cap and a network block — neither of which the current snippets need, because every snippet through
Phase 2 is templated by us. It becomes load-bearing the moment model-authored feature code runs
(the parking-lot item in NEXT.md), and the `SandboxPool` interface is the seam it would land behind.

The build also closed a hole the Phase 1 shim's docstring claimed was already closed. `local.py`
stripped the environment and asserted a snippet "cannot see the repo it is being graded in" — but
an editable install puts the repo root and `src/` on `sys.path` through a `.pth` file in
site-packages, which no environment stripping touches. Snippets could `import ds_agents`. Nothing
handed them the live state object, so no result to date is wrong, but "the agent could read the
scoring code" and "the agent reasoned" are not distinguishable after the fact from a results table.
`_worker._prune_sys_path` now keeps only the standard library and site-packages, and
`test_the_repo_is_not_importable_from_a_snippet` is the regression test. The general lesson is
worth more than the fix: `sys.path` is not part of the environment, so an environment-based
isolation argument is incomplete by construction.

## 2026-08-27: the MCP server is a skin over the in-process tools, not a second implementation.

`mcp_server/server.py` constructs a `LocalTools` — the same binding of `SandboxPool` and
`ArtifactStore` the graph used through Phase 1 — and exposes its four methods as MCP tools.
Nothing in the protocol layer decides how a snippet runs, what an artifact id looks like, or when
a read is truncated. The alternative, a server that talks to the store and the sandbox directly,
reads cleaner and quietly makes "we swapped the shim for the server" mean "a second implementation
appeared and matched the first by inspection". Since the single-agent-vs-team ablation rests on
both arms getting the *same* tool surface, sameness has to be a fact about the code rather than a
claim about it. The cost is a package-level import from `mcp_server` into `ds_agents.tools` and
back; the rule that actually matters — the sandbox worker never imports `ds_agents` — is
unaffected, because the worker is a separate process launched from `_worker.py`.

`tools/mcp_client.py` closes the loop from the other side and `cli.py` gained `--tools mcp|local`,
defaulting to `mcp`. The default is the point: a default that took the in-process shortcut would
leave the transport exercised only by its own tests, and the numbers would be produced under a
configuration nothing publishes.

## 2026-08-27: one server process per run, because a fifth tool would change what the ablation holds constant.

The store is run state — it owns the artifact index and the run's directories — while the sandbox
worker deliberately holds none and is shared process-wide. So a long-lived server would need to
key stores by `run_id`, which means either a fifth tool to open a run or a `run_id` argument on
the other four. Both change the agent-facing tool surface, and that surface is exactly the thing
held constant between the single-agent and team arms; it does not get an extra tool for our
convenience. One process per run instead, with the run's root and dataset passed on argv, which is
also the shape an external MCP client is configured with. The cost is one 0.87s worker boot per
dataset, which a Phase 4 benchmark pays against minutes of fitting. It stops being the right
trade if a run ever needs to outlive its client or be resumed.

## 2026-08-27: a read with no `max_bytes` is still capped, and the cap lives in the store.

`ArtifactStore.read` now applies `DEFAULT_READ_BYTES` (1 MiB) when the caller names no limit. Two
reasons it is not a transport concern. First, uncapped is not a thing a caller can actually want:
the only two consumers render an artifact into a prompt or into snippet source, and neither has a
size it survives. Second, if the cap lived in the MCP layer, a node would truncate at a different
byte depending on how it was wired, so `--tools local` and `--tools mcp` could disagree about
whether an artifact was whole — and both `feature_eng` and `modeler` refuse to act on a truncated
artifact, meaning the transport could decide whether a run errored.

The cap also found a live gap rather than only preventing a future one. `feature_eng` guarded its
read of the split manifest and `modeler` did not, which was harmless while an unbounded read was
possible and became a silent wrong number the moment it was not: the fitting snippet would have
trained and scored on a subset of the pinned split while reporting `claimed_holdout_score` as if
it were the holdout. The guard is now symmetric, with a test on each side. The general shape is
worth noticing — a default that turns an impossible state into a possible one activates every
unwritten check at once, and the ones that were never written stop being dead code quietly. The escape hatch for a large
artifact already exists and is not a bigger cap: `ArtifactMeta.extra["sandbox_path"]`, opened from
inside a snippet. On the toy dataset nothing comes near the cap; the Phase 4 split manifest will,
which is the parking-lot item this makes urgent rather than a reason to raise the number.

## 2026-08-27: a sandbox failure keeps its type across the wire.

MCP reports failure as an error result carrying a string, so in-process the two failure types —
`ToolError`, a finding about the agent, and `SandboxError`, a finding about us — would arrive
identically and every broken worker would be filed as an agent mistake in the results table. The
server prefixes a `SandboxError` with `SANDBOX_ERROR_PREFIX` and the client raises it back as
itself. The prefix is matched with `partition` rather than `startswith` because the SDK prepends
its own "Error executing tool <name>: " to an expected tool error. Anything that is *not* one of
those two types is left to the SDK, which withholds the text — correct, because an unexpected
exception in the protocol layer is a bug in our code, not a finding about a run.

## 2026-08-27: the generalist baseline is verified by use, not by an approval flag.

The Phase 2 checkbox asked whether Claude Code can connect to the same server, because Phase 5's
single-agent arm is Claude Code driving these four tools directly and the comparison is only
honest if both arms face the identical surface. Last session answered it with a hand-rolled
JSON-RPC client, which proved the protocol was standard but not that a real MCP client would find
the tools usable. It is now answered the direct way: inside Claude Code the tools resolve as
`mcp__ds-agents-tools__*` and all four were driven against the toy dataset by an agent sharing no
code with `tools/mcp_client.py`. What that exercise actually confirmed, beyond "it connects", is
that the two invariants the sandbox is for survive a client we did not write — `run_python`
reported the repo absent from `sys.path` and no `ANTHROPIC_API_KEY` in its environment — and that
the truncation contract is legible from the outside: a 202-byte artifact read with `max_bytes=40`
came back `truncated: true`, and the same artifact opened whole from inside a snippet at the
`sandbox_path` its own metadata carried. That last pair is the one thing a hand-rolled client
could not have shown, since the escape hatch only means anything if the client that hits the cap
can find it.

The residual is that `claude mcp list` still prints "pending approval" for the project-scoped
entry, so a fresh terminal session prompts once before the tools appear. That is a per-machine
trust prompt about running `uv run mcp-server`, not a property of the server, and it is worth
being precise about which of the two the checkbox was ever asking after. Phase 5 needs the tools
usable by a generalist agent, which is now demonstrated; it does not need the prompt to have been
pre-clicked on any particular machine, and a CI arm would supply trust its own way rather than
through this flag.

## 2026-08-27: the router mints the whole ReviewPass; the reviewer hands dispositions across on one narrow field.

`ReviewPass` needs two things only one node knows: `dispositions`, which only the reviewer produced,
and `routed_to`, which only the router can compute (it is the sole reader of `reviewer_dispositions`).
The natural-looking alternative is each node appending its own partial pass. It fails immediately:
`review_passes` is `operator.add`, so two appends in one iteration mint two records instead of one,
and every downstream count built on it doubles. The router mints the whole `ReviewPass` instead, and
the reviewer hands its half across on `reviewer_dispositions`, a field with exactly one writer and
one reader, deliberately not `operator.add` -- the reviewer overwrites it wholesale every pass
because it is a handoff for the invocation in progress, not a record of history the way `objections`
and `review_passes` are. `routed_to` gets the same single-authority treatment for a reason that is
not cosmetic: the router computes the destination against the state as of its own invocation, before
this pass's update is merged, while a graph edge re-deriving it would run after the merge. A pass
that resolves a `feature_eng` objection while raising a new `modeler` one gives the two computations
different, both locally defensible, answers -- so the router decides once, writes `routed_to` onto
the pass, and `route_target` reads it back rather than recomputing. The last piece closes a
staleness hole rather than an ownership one: the reviewer rewrites both `reviewer_claim` and
`reviewer_dispositions` on every return path, including a failed model call, so a crashed pass reads
as `pending` rather than leaving the previous pass's claim sitting on the field for the router to
count a second time.

## 2026-08-27: a malformed objection is filtered one at a time, never by the response schema.

The reviewer node filters each proposed objection's `columns` against the profile's columns minus
the target, and a column-scoped category (`leakage`, `contamination`, `implausible_importance`) left
with no surviving column is rejected outright, not downgraded to `other`. Downgrading looks safer
and is not: it produces an objection `feature_eng` will not act on -- `other` carries no obligation
-- while the router still counts it as open and keeps blocking on it, which guarantees `exhausted`
instead of giving the model a real path to `pass`. The first implementation put that same rule on
`ProposedObjection` as a pydantic validator, one item enforcing itself. On 2026-08-27 Haiku raised an
`implausible_importance` objection with an empty `columns` list; the validator raised, the whole
`ReviewFinding` failed to parse, and a pass that carried a real claim and real dispositions alongside
the one bad objection was recorded as a reviewer crash -- `pending`, not the block it should have
been. The rule moved to a loop inside the node that filters proposals one at a time, so one bad item
is dropped and logged while the claim, the dispositions, and every well-formed objection in the same
pass survive. The transferable point: a per-item rule enforced on the response schema is not actually
a per-item rule, it is a whole-response rule, and at an LLM boundary that converts one bad item into
total data loss for the pass.

## 2026-08-27: trap difficulty is set by the column name, not by the association number.

The two new leakage-trap fixtures were tuned on the wrong axis, and finding that out is the useful
part. The premise was that the toy leak is caught every time because it screams statistically --
0.518 normalized mutual information against 0.094 for the legitimate strong feature -- so the new
traps were built to sit inside the range a good feature occupies. `claims_timing` puts `days_to_close`
and `adjuster_touches` at 0.125 and 0.184 against a legit 0.080; `reissued_ids` puts `member_number`
at 0.0945 against `credit_score` at 0.0952, which is as close to statistically indistinguishable as a
fixture can be built. It made no difference. Across six live Haiku runs the profiler nominated the
planted column in 6 of 6, `feature_eng` dropped it, and the reviewer was handed a clean matrix again
-- the same dead end the toy fixture produces, reached from the opposite direction.

A name ablation says why. The same `claims_timing` CSV with `days_to_close` and `adjuster_touches`
renamed to `metric_a7` and `metric_b3` -- identical values, identical mutual information, identical
everything the profiler computes -- was nominated in 1 of 3 runs against 3 of 3 for the descriptive
names. n=3 a side, so this is directional and not a rate, but the direction is not subtle and the
mechanism is not in doubt: the profiler's prompt contains the column names, and the model is reading
them. What is being measured when we "plant a leak" is therefore mostly whether the column is
*named* like a leak, and the association number is closer to a tiebreak than to the evidence base.
The consequence for the benchmark is that name transparency belongs as an explicit condition -- an
ablation axis, descriptive versus opaque, held against the same rows -- rather than as an unrecorded
property of whatever the fixture author happened to call a column. That is a manifest change and a
harness change, so it is written up for the next session rather than taken in passing here. Both
fixtures stay as built and honestly labelled; retuning them until Haiku misses would fit the
benchmark to one model, which is the failure mode the fixtures exist to avoid.

## 2026-08-27: the reviewer misses a top-ranked leak that reaches it, in 3 of 3 runs.

Incidental to the fixture work and more important than it. The point of Phase 3 is unmeasurable while
upstream cleans the matrix first, so the runs where a trap *did* survive are the only real data the
project has on its central claim. There are three of them: one live `claims_timing` run where
`adjuster_touches` survived to a claimed roc_auc of 0.954, and two name-ablation runs where both
traps survived as the number one and number two permutation importances at a claimed 0.986 against a
legitimate ceiling near 0.82. In all three the reviewer returned `claim: "pass"` and raised no
`leakage` or `contamination` objection at all. It is not that the reviewer argued the columns were
fine -- it never mentioned them. Set against what it does raise unprompted (four `metric_mismatch`
objections in one `reissued_ids` run), the shape of the failure is that the reviewer reliably audits
the numbers it is shown and does not interrogate what the columns mean, which is the one thing the
matrix cannot tell it. This is the finding Phase 3 exists to produce; it should be reproduced with a
larger n and against a Sonnet reviewer before it goes in a README.

## 2026-08-27: name transparency becomes a run condition, applied at load time.

The previous session established that trap difficulty in this project is set mostly by what the
fixture author called a column: two traps tuned until their association with the target was
indistinguishable from a legitimate strong feature were nominated anyway, and renaming them defeated
the profiler where no amount of association tuning had. That makes the column name an uncontrolled
variable under every leakage number, so it had to become a condition that gets recorded. Three
shapes were available. A paired fixture -- a second committed CSV with opaque headers -- is
reproducible but differs from its partner in the names *and* in whatever its generator's noise drew,
so a difference between the arms would have two candidate explanations. A `naming` field on the
manifest has the same problem plus a maintenance one: two files that are supposed to be identical
below the header, kept in step by hand. What landed instead is `src/ds_agents/naming.py`, which
rewrites the header line and copies every remaining byte of the CSV through unchanged. The arms
differ in one line of one file, which is the only shape in which the comparison isolates the
variable. Two tests hold that claim up rather than asserting it in a comment: the file body is
compared character for character, and both arms are required to produce identical permutation
importances under `StubModel` -- if they ever diverge, something other than the names moved.

The rename is applied above the tools boundary, in `cli.py`, and `nodes/`, `tools/` and
`mcp_server/` are untouched by it: the only thing that reaches them is a different CSV path. That is
also what keeps the fixture rule intact, since nothing in `nodes/` may import `naming.py` any more
than it may import `fixtures.py`. It lives outside `fixtures.py` because that module commits to
never reading the CSV and the map needs the header -- `claims_timing` has a column no manifest field
declares, so the manifest could not have been the source. Two consequences worth stating. First,
*all* non-target columns are renamed, not just the traps: renaming only the traps, which is what the
previous session's ad hoc ablation did, makes an opaque name the thing only traps have and hands the
profiler a replacement cue for the one it lost. The cost is that the new rate is not directly
comparable to the earlier 3-of-3 versus 1-of-3, and it should not be quoted against it. Second, the
target keeps its name, because intake is given prose and inferring the target from it is the node's
actual job; renaming it would change the task rather than the condition.

Ground truth follows the rename. `_fixture_state` maps `planted_leakage_columns` through the same
dictionary the CSV got, because otherwise `results_row()` would compare objections against columns
that do not exist in the opaque arm's data and every opaque run would score a silent zero -- a name
effect that was entirely an artefact of the bookkeeping. NEXT.md's reproducibility objection to a
load-time rename resolves itself: `ArtifactStore.register_dataset` already copies the dataset the
agents saw into `<artifacts-dir>/data/`, so the exact renamed CSV is on disk for any run started
with `--artifacts-dir`.

## 2026-08-27: results rows start being written, and learn to see the profiler.

Two things were fixed because without them the naming session would have produced prose over ad hoc
runs for the third time. `ds-agents run` grew `--repeat N` and `--results PATH`, which appends one
`results_row()` line per run behind the same `publishable()` gate the Phase 4 harness will apply.
This does not preempt that harness, which is about the OpenML benchmark set and its baselines; it is
the smallest thing that makes a number in a doc traceable to a committed row.

The second is more substantive. `results_row()` scored leakage from *reviewer objections only*, and
the dependent variable in a naming ablation is a **profiler** decision -- so twenty runs would have
produced a results file with the effect being measured entirely invisible in it. Four fields now
come off `profile.leakage_candidates`: `profiler_nominated`, `profiler_caught`, `profiler_recall`
and `profiler_false_alarm`. They are deliberately separate from the reviewer's numbers rather than
folded into `leakage_caught`, because on the trap fixtures the two nodes give different answers to
the same run -- the profiler nominates the planted column and the reviewer, shown the result, says
nothing -- and one combined number would report a team that catches leaks while hiding which member
caught it. All four are null rather than zero when `profile` is None, because a profiler that
crashed nominated nothing in a different sense than one that looked and declined, and averaging
those together over a benchmark would be a lie. This closes the narrow half of the open question
about `results_row()` being unable to see a profiler false alarm; the first live opaque run scored
`profiler_false_alarm: 1` on the fixture's honest id column, which is the first time that
long-standing blind spot has appeared in a row rather than in a note.

## 2026-08-27: the name effect is real, it is large, and it decides whether the reviewer is tested.

Twenty live Haiku runs on `claims_timing`, ten a side, over byte-identical rows differing only in
the header. The 3-of-3 versus 1-of-3 from the previous session was directional; this is a rate, and
it is bigger than the directional version suggested. The profiler nominated at least one planted
trap in **10 of 10 descriptive runs and 2 of 10 opaque** ones; mean `profiler_recall` 0.95 against
0.10. The traps are the same columns with the same values and the same mutual information with the
target in both arms. Renaming them was worth roughly the entire signal.

The consequence is not confined to the profiler, which is what makes this worth acting on rather
than noting. Because `feature_eng` drops what the profiler nominates, the trap **survived into the
model matrix in 2 of 10 descriptive runs and 10 of 10 opaque** ones, and mean claimed roc_auc rose
from 0.840 to 0.974 against a legitimate ceiling near 0.82. A benchmark built from descriptively
named fixtures therefore mostly does not test the reviewer at all: upstream cleans the matrix first
and the reviewer is handed a clean problem. That is the mechanism behind three sessions of "the
reviewer never gets a leaky matrix", stated as a number for the first time.

One control makes the mechanism specific rather than vague. The honest identifier `claim_ref` was
nominated in 10 of 10 descriptive runs and, as `var_01`, in 9 of 10 opaque ones -- essentially
name-invariant, because its distinctness is visible in the profile statistics the model is given.
The traps collapse from 10 to 2. So the profiler is not simply flagging less when it loses the
names, and it is not hedging more either (mean false alarms 1.0 against 1.2). It detects
*statistically* visible leakage regardless of naming and *semantically* visible leakage only when
the name says so. Both traps were deliberately tuned to be statistically ordinary, which is exactly
the class it cannot see.

## 2026-08-27: the reviewer misses the leak in 12 of 12 runs where it had one, and looks in the
wrong place while doing it.

The previous session's 3-of-3 is now 12-of-12. Across all twenty runs the planted trap survived
into the matrix twelve times, and in **zero** of those twelve did the reviewer raise a `leakage` or
`contamination` objection naming any column. It is not that the reviewer is passive: the opaque arm
raised nine objections to the descriptive arm's two, because a claimed 0.974 is conspicuous. It
noticed. One run at a claimed 0.9886 is the whole failure in one artifact: the reviewer raised
`metric_mismatch` ("a 0.074-point improvement ... not credible without evidence of how it was
computed") and `implausible_importance` naming `var_05, var_06` -- the two *noise* columns, flagged
because their importance was near zero -- while `var_07` and `var_08`, the planted traps sitting at
the top of the importance table and solely responsible for the inflated score, are not mentioned in
either objection. The reviewer has the number that says the score is too good, reasons about it
correctly, and then searches the bottom of the importance ranking instead of the top. That is a
sharper and more actionable finding than "it misses leaks", and it points at the prompt: nothing
asks it to explain *which column* produced an implausible score.

Two things follow for Phase 5 and neither is taken now. The Haiku-versus-Sonnet reviewer arm is now
clearly worth running -- `--reviewer-model` exists, is unexercised, and this is the first result
that makes the comparison interesting. And the opaque arm is the configuration in which the
reviewer ablation is actually measurable, because it is the only one that reliably puts a leaky
matrix in front of the reviewer; the descriptive arm answers a question about the profiler.

## 2026-08-28: the reviewer arm is a 2x2, because the prompt was a confound under any model
number, and the prompt turns out to be the whole effect.

The plan was the Haiku-versus-Sonnet reviewer arm alone. It was run as a 2x2 against a second
condition, `reviewer_prompt`, for one reason: yesterday's finding was that the reviewer reasons
correctly about an implausible score and then names the wrong columns, and nothing in its system
prompt asks which column produced the score. That leaves "Haiku cannot see this" and "we never
asked" indistinguishable, and a model arm run under one arbitrary prompt would have produced a
number -- "Sonnet catches 4 of 10" -- silently conditional on whichever prompt happened to be in the
file that week. The prompt was not even recorded, so the dependency would have been invisible to
anyone reading the results file later. The alternative that looks equivalent is running the model
arm first and the prompt arm after, and it is not: the cells cost the same either way, and only the
factorial version can say whether a prompt fix helps the weaker model, the stronger one, or both.

The answer is that the prompt is the binding constraint and the model is not. One appended rule --
if the claimed score is implausible, read `top_importances` from the top and name the column that
explains it -- moves Haiku from 1 of 10 to 9 of 10 runs naming a planted trap. Sonnet under the base
prompt is 0 of 4, no better than Haiku. This reframes yesterday's entry: the miss was not a
capability limit and not an evidence-surface gap either, since `top_importances` was already in the
prompt with the traps at the top of it. It was an unasked question. The rule names no fixture, no
column and no trap type and points only at a field the reviewer already receives, which is the line
this project draws between fixing a prompt and injecting the answer; the hint-injection ablation
still parked in NEXT.md is the version that deliberately crosses it, and should be labelled as such
when it runs.

Two implementation choices are worth recording because both were the cheap-looking option's
opposite. `reviewer_prompt` is a `Literal` on the frozen `RunConfig` rather than an edit to
`REVIEWER_SYSTEM`, and `base` is byte-identical to the prompt every earlier run used: an unrecorded
prompt change would have made every prior row incomparable to every later one with nothing in either
row to say so, which is the same failure `naming` was created to prevent. And the new
`reviewer_nominated`/`reviewer_caught`/`reviewer_recall`/`reviewer_false_alarm` fields sit *beside*
`leakage_flagged` rather than widening it. Widening was tempting -- `leakage_flagged` covers only
`leakage` and `contamination`, so a reviewer naming the trap under `implausible_importance`, which
is exactly what it does, scored as a miss -- but the committed naming-ablation rows were written
under the narrow definition, and redefining it would have silently changed the meaning of the only
published results file in the project. The divergence is deliberate and is pinned by a test that
asserts both numbers in the same row.

The finding underneath the finding is that naming the column is necessary and not sufficient. Eight
of the ten Haiku `which_column` runs end `exhausted`: the reviewer names the trap, the loop cap
expires before `feature_eng` removes it, and a 9-of-10 catch rate becomes a 1-of-10 remediation
rate. The reviewer half of the pipeline now works and the remediation half is the next bottleneck,
which is a better problem than the one this session started with.

## 2026-08-28 (second entry): the loop cap was never the bottleneck; an objection needs a
route it can act through and a condition that closes it.

Yesterday's session ended with 9 of 10 opaque Haiku runs naming a planted trap and 1 of 10 removing
it, 8 of 10 ending `exhausted`, and the obvious reading that the loop cap expires before
`feature_eng` gets there. That reading was wrong, and it was cheap to falsify: three diagnostic runs
at $0.10 total, then a pre-registered `loop_cap` sweep at 1/3/5 which moved remediation 0, 1, 1 out
of 10 while quadrupling cost per run. The cap is a null lever.

What the diagnostic runs found instead is two independent breaks in the objection lifecycle. The
first is **routing**. `feature_eng` force-drops an objected column before its own model is
consulted, but only for objections whose `target_node` is `feature_eng`; `open_objections` filters
by target. The reviewer, under `which_column`, overwhelmingly raises `implausible_importance` --
which is an objection about a *score* -- and `REVIEWER_SYSTEM` tells it to send anything about "how
the run was evaluated" to `modeler`. That is a correct reading of the prompt and a dead end in the
graph: the modeler has no column lever at all, since every candidate is fit on the one transform
`feature_eng` already froze. In the sweep's `loop_cap=5` cell, 5 of 10 runs addressed every
objection to `modeler` and 6 of 10 never routed to `feature_eng` once. The second is **closure**.
Across 3 diagnostic runs the reviewer never marked a single objection `resolved`. The clearest run
dropped both traps, watched the claimed roc_auc fall from 0.986 to 0.823, wrote that the fall "is
consistent with removing leakage", and held the objection open anyway on the grounds that the
columns "were never validated as non-leaking, only removed" -- an unfalsifiable standard. A reviewer
holding one can never let a run pass, which means `exhausted` stopped being evidence that the fix
did not land. `leakage_remediated` is the field that answers that question and it is the one to
quote; the eval was never wrong, but the sentence we wrote around it was.

The fix is deliberately not in this session. Three candidates were open in NEXT.md and the
diagnosis narrows them without choosing: routing is now the leading one and closure is a second,
separate change, and each deserves a controlled before/after against these numbers rather than being
bundled with them. What did land is the instrumentation that makes the next attempt measurable.
`results_row()` gained `objections_by_target_node`, `objected_columns_unremediated`,
`route_sequence` and `new_objections_per_pass`. All four are *derived* from `objections`,
`review_passes` and `final_features` rather than recorded by any node, which is the same rule as
every other outcome on that row: a node that wrote down its own remediation would be a node
reporting its own score. No node, tool or `PipelineState` field changed. `objected_columns_
unremediated` is `None` rather than `[]` both when no reviewer pass completed and when the matrix is
empty, matching `reviewer_nominated` and `leakage_remediated` -- an empty list there would say
"objected to and all of it dropped", which is the opposite of what the reviewer-off arm did.

`--loop-cap` also became a CLI flag. It had existed on the frozen `RunConfig` with a default of 3
and been reported on every results row since the first one, but there was no way to set it, so the
condition every published row named was one no run could vary. It is validated at the CLI with the
exit-code-2 pattern `--repeat` uses rather than left to Pydantic's `ge=0`, and `0` is deliberately
legal: the router already special-cases it, and a cap of zero is the reviewer-off condition
expressed as a cap.

Postscript, same day, from topping the Sonnet cells to n=7. The routing diagnosis above was found on
Haiku and it generalises, and it also explains the model arm. Across all 27 rows that carry
`route_sequence`, 0 of the 21 runs that never routed to `feature_eng` remediated, against 3 of the 6
that did. Sonnet under `which_column` remediates 3 of 7 where Haiku manages 1 of 10 -- and it does
so while *detecting less* (5 of 7 name a trap against Haiku's 9 of 10). The stronger model's
advantage here is not that it sees more; it is that it addresses what it sees to the node that can
act on it. That is a finding about the pipeline's contract rather than about either model, and it
raises the value of the routing fix: it is worth more than the reviewer-model upgrade that would
otherwise be the obvious thing to buy.

## 2026-08-28 (third entry): who acts on an objection becomes a recorded condition, and the
## reviewer's own choice stays on the record.

NEXT.md left the routing fix as an open question rather than a task, and the question was whether
it is a fix at all: `implausible_importance` naming a column is a `feature_eng` problem whatever
the reviewer calls it, so the graph could ignore `target_node` for column-scoped categories -- but
the reviewer is a model under test, and deciding its dispatch on its behalf makes the pipeline work
by making one fewer thing measurable. That is the same shape of question `naming` and
`reviewer_prompt` raised, and it gets the same answer: record the condition rather than argue about
it. `objection_routing` is a `Literal["as_addressed", "by_category"]` on the frozen `RunConfig`,
defaulting to `as_addressed`, which is byte-identical to every run committed before today. Under
`by_category` an objection whose category is in `COLUMN_SCOPED_CATEGORIES` has an *effective* target
of `feature_eng` regardless of what the reviewer wrote.

The load-bearing detail is what does *not* change. `Objection.target_node` is never rewritten. The
reviewer's dispatch judgement stays on the record and `objections_by_target_node` keeps counting the
raw field, so the thing the override is accused of hiding is still measured in the arm that
overrides it -- and `objections_rerouted` beside it says how often the graph disagreed. Applying the
condition in one place is what makes that guarantee checkable: `PipelineState.effective_target` is
the only reader of `config.objection_routing`, `open_objections(target)` is its only caller, and
`_route_for_block` was rewritten to ask `open_objections(destination)` rather than comparing
`target_node` itself. The router had been a second, independent answer to "who acts on this", which
is the dual-authority split `routed_to` was introduced to close; it is now one answer. For the same
reason `FORCING_OBJECTION_CATEGORIES` in `feature_eng.py` was deleted in favour of importing
`COLUMN_SCOPED_CATEGORIES`. The two frozensets were byte-identical, and had they ever diverged the
router would have sent an objection to `feature_eng` that `_forced_drops` then skipped -- the same
dead end reintroduced one layer down.

Three things about this arm are not flattering and belong here rather than in a footnote. First, it
is not a pure edge change. `feature_eng` and `modeler` both read `open_objections(target)`, so
`by_category` also changes what those two nodes are *shown*: the column objection leaves the
modeler's prompt and enters `feature_eng`'s. That is the right call -- the modeler's only lever is
which candidate to promote, so a column complaint in its prompt can only produce a spurious response
that looks like remediation in a trace and is nothing of the kind -- but the arm does not hold two
prompts constant and must not be described as if it did. Second, `objections_by_target_node` demotes
from an outcome to an annotation. In the `as_addressed` arm it predicts the run's fate (0 of 21 runs
that never routed to `feature_eng` remediated); in `by_category` it predicts nothing, and two rows
carrying the same value now mean different things depending on another column. `objection_routing`
on every row is what keeps that legible. Third, the arm converts reviewer false positives into real
dropped features. Under `as_addressed` a wrong objection addressed to `modeler` was harmless because
nothing could act on it; the one run in the cap=5 cell that remediated also objected to `var_04`,
which is `prior_claims_12m` and which the `claims_timing` manifest lists as a legitimate strong
feature. `n_final_features` was added to `results_row()` for this: `leakage_remediated` is `None` on
an empty matrix but `True` on a one-column one, so without a width beside it a run that "remediated"
by force-dropping most of the fixture reads identically to one that dropped the trap.

Hence a rule about language, which is the part most likely to be lost. The `by_category` remediation
number is *reviewer detection plus scripted dispatch*. It is never "the agent team remediated N of
10". The claim the arm can support is that the pipeline's contract, not the reviewer's competence,
was the binding constraint on remediation -- which is a finding about our design and is worth having
precisely because it is not a finding about the model.

## 2026-08-28 (fourth entry): a resolved objection keeps its column out, and the closure arm's
## premise gets checked before it gets funded.

Two changes here, one a bug fix and one a condition, plus a finding that arrived between them and
changed what the condition is for.

**The bug: resolution un-fixed itself.** `feature_eng._forced_drops` recomputes from scratch on
every entry to the node, and it read `open_objections("feature_eng")`. `open_objections` closes on
`resolved` OR `withdrawn`, so the moment the reviewer marked a column objection `resolved`, the
column stopped being force-dropped -- and the next return to `feature_eng`, for any unrelated
reason, put it back in the matrix. Verified directly against the node: with the objection open the
snippet reads `DROP = ['account_status_code', 'churned', 'customer_id']`; with it resolved,
`DROP = ['churned', 'customer_id']`. `test_a_resolved_objection_does_not_force_a_drop` had pinned
that as intended behaviour since the day this node learned about objections, and across a single
pass it is indistinguishable from correct, which is why it survived. `objection_routing=
"by_category"` produces the two-return shape routinely and closure would produce it more, so the
fix had to land before the arm or the arm would have been measured on top of it.

The fix is a second named question rather than a flag on the first. `PipelineState.
binding_objections` answers "what must stay OUT of the matrix" and releases only on `withdrawn`;
`open_objections` still answers "what is still being complained about" and closes on both. The
distinction is not pedantry: on this pipeline the fix for a column objection IS the drop, so
`resolved` cannot also mean "put it back", while `withdrawn` -- the reviewer saying it was never a
problem -- is the opposite claim and is the only route back for a false positive. That matters
because `by_category` dropped `prior_claims_12m`, a legitimate strong feature, in 2 of 10 runs.
A boolean parameter on `open_objections` was rejected: it presents two questions as one question
with a switch, on a method with six call sites, and the names are the documentation. The blast
radius is the invariant to check -- `binding_objections` has exactly one caller in the graph. The
router must keep asking `open_objections`, or a resolved objection would route the run upstream
forever and every run would end `exhausted`; so must the reviewer's own facts, or it would
re-adjudicate what it already closed and closure would be a treadmill rather than a termination
condition. Both are pinned by tests that passed before the fix and must keep passing after it.
The fix is unconditional rather than a fifth ablation axis, because the un-sticky arm is a
known-buggy pipeline and measuring it would measure nothing. It is not free: 7 of 84 committed rows
meet the necessary condition for the old behaviour to have fired, 4 of them in the cell that would
otherwise have been the closure control, which is why that control is re-run rather than reused.

**The condition: `objection_closure`, a `Literal["off", "on"]` on the frozen `RunConfig`,
defaulting to the byte-identical old prompt.** `on` appends one rule saying an objection about a
column is answered when that column is absent from `final_features`. Two choices inside that are
worth recording. It is a prompt rule and not node-side auto-resolution, because the node could
trivially mark the objection resolved itself and that would take the reviewer out of the decision:
the honesty metric below would be structurally zero and the arm would measure the pipeline rather
than the agent. And it is its own axis rather than a third `reviewer_prompt` value, because the
rules are independent -- bundling would mean closure could never be measured without `which_column`
attached and the two effects could never be attributed separately. The criterion is mechanical
absence rather than "the claimed score is back in a plausible band" for a reason worth more than
simplicity: the reviewer has no plausible band, and any band we supplied would be derived from the
fixture's known legitimate ceiling. That is the answer key, injected into the one judgement the
whole project exists to measure. `final_features` is a field the reviewer is already shown.

**The finding, which arrived before the money did.** The arm was designed against a diagnosis in
NEXT.md and PLAN.md: the reviewer never dispositions an objection `resolved`, so `exhausted` is a
default rather than evidence. Re-reading the control cell's own committed rows before running
anything, that diagnosis does not hold at n=10. Eight of the ten routing rows closed at least one
objection, including all four that ended `exhausted`; and all four of those raised a *new* objection
on their *final* pass, which the cap routes straight to the reporter. They did not exhaust by
refusing to close. They exhausted because the reviewer names one trap per pass and the cap cuts the
sequence off. The original diagnosis rests on n=3 diagnostic runs that predate the routing arm.

What is still unknown is whether those closures are `resolved` or `withdrawn`, because no row
written before today splits them -- `objections_open_at_end` conflates two opposite claims about the
reviewer, and that same gap is why the sticky-drop screen above cannot resolve its 7 at-risk rows
past "at risk". `objections_resolved` and `objections_withdrawn` are on the results row from this
commit for that reason, alongside `objections_falsely_resolved`: an objection marked `resolved`
while one of its columns is still in `final_features`. That last one is the arm's falsifier and the
reason the arm is worth running carefully rather than eagerly -- a prompt that buys termination by
teaching the reviewer to say "fixed" is worse than no prompt, and nothing on any earlier row would
have caught it. All three are pure derivations from state the pipeline already had; no node changed
to produce them.

So the control cell is pre-registered as a decision gate rather than as a baseline, with the
stopping rule fixed in advance in `evals/results/LOG.md`: if the reviewer already resolves honestly
without being told how, the arm is not run and that is the finding. Pre-registering a metric whose
direction you cannot predict is pre-registering a coin flip, and the same applies to funding an arm
whose premise you have not checked. `leakage_remediated` is recorded as a non-inferiority guardrail
here and explicitly not as the endpoint, because closure ends the loop *earlier* and `claims_timing`
plants two traps: a reviewer that resolves after the first drop never reaches the pass in which it
would have named the second. It is allowed to fall, and that is written down before the run.

**Outcome, appended after the control cell ran.** The gate closed at exactly its pre-registered line
-- `objections_resolved > 0` in 6 of 10 -- and the arm was not run. Two numbers fixed in advance
make that verdict stronger than the 6/10 alone: `objections_falsely_resolved` is 0 in every row, so
the dishonest-closure failure the rule exists to risk is absent from the population it would be
applied to; and `leakage_remediated` is 9/10, with the single non-remediating run being the
unrelated "claimed `block` with zero objections raised" bug, so there is nothing left for a prompt
rule to buy. The condition, the rule and the three columns stay in the tree, because a recorded null
is worth more than an untested hunch and the columns are what made the gate decidable at all. The
unpredicted result is that the sticky-drop fix, the bug fixed on the way to the arm, moved
remediation 5/10 to 9/10 -- larger than the routing arm's own 1/10 to 5/10 -- with 6 of 6 resolving
runs remediating. That has no pre-registration of its own and is written up in LOG.md as a finding
needing its own cell, not as a headline.

## 2026-08-28 (fifth entry): the release rule becomes a recorded condition, and the fourth entry
## of the same day is overturned.

The fourth entry decided the sticky-drop fix would be unconditional: "The fix is unconditional
rather than a fifth ablation axis, because the un-sticky arm is a known-buggy pipeline and measuring
it would measure nothing." That is two claims, and only one of them survives.

**The first claim is still true and this change does not contest it.** The un-sticky rule is a bug,
not a design alternative. That is exactly why `forced_drop_release` is the first condition on
`RunConfig` whose **default is not the pre-existing behaviour**. `naming`, `reviewer_prompt`,
`objection_routing` and `objection_closure` all default to the arm that reproduces every committed
row byte for byte, because each of those is a real fork with two defensible answers. This one
defaults to the *fixed* rule, `withdrawn_only`, and its off value is documented in the field
description as a defect reproduction. The inversion is the entire content of the change, and it is
pinned by `test_the_default_release_rule_is_the_fixed_behaviour_and_deliberately_not_the_old_one`
with the reasoning in the docstring -- because it reads as an inconsistency, and a future session
tidying it toward the surrounding pattern would silently make the buggy pipeline the shipped one.

**The second claim was right about a hypothetical arm and wrong about the one that now exists.**
When it was written the un-sticky pipeline was something nobody would ship and nobody had a claim
about. Since the closure cell ran it is the **denominator of the project's largest published
number**. "The sticky rule moved `leakage_remediated` 5/10 -> 9/10" is a causal claim about the
un-sticky pipeline, and you cannot make that claim while refusing to measure the thing it is about.
Refusing does not avoid measuring a buggy pipeline; it means the buggy pipeline gets measured once,
badly -- across a code boundary, by a cell run for another purpose, in a file that lacks
`objections_resolved`, `objections_withdrawn` and `objections_falsely_resolved`, which are the
columns the mechanism check needs. The choice was never "measure the bug or don't". It was "measure
it deliberately for $0.30, or keep quoting a number derived from having measured it accidentally".
What changed between the two entries is evidence, not taste: the effect turned out to be +4/10,
larger than the routing arm's own pre-registered effect, and the closure columns that make the
mechanism decidable did not exist when the decision was taken.

The condition is applied in exactly one place, the release set in `PipelineState.binding_objections`,
and under `resolved_or_withdrawn` that method is provably identical to `open_objections` -- the
pre-fix predicate itself rather than a reimplementation of it, pinned by
`test_the_unsticky_arm_is_exactly_open_objections_again`. That matters more than it looks: a control
arm that approximated the bug slightly differently would produce a difference nobody could attribute.
The same commit finally enforces the single-caller invariant that two docstrings have asserted since
the fix landed. `test_binding_objections_has_exactly_one_caller_in_the_graph` walks the package's
AST and fails if anything but `feature_eng._forced_drops` calls it. This is a new kind of test for
this repo -- nothing here read its own source before -- and it is the cheapest thing in the session:
the router and the reviewer must keep asking `open_objections` or a resolved objection routes the run
upstream forever and every run ends `exhausted`, and neither failure would break any existing test.
Both are silently-wrong-number failures, which is the class this project exists to catch.

**The finding that came out of reading the fix's own diff, and that would have confounded the arm.**
`d5a9a28` changed two things in `feature_eng`, not one: the predicate, and the justification string
attached to each forced drop, which reaches the feature_eng model's prompt through `already_dropped`
and `dropped_features` on the results row (`"open reviewer objection X: ..."` became `"reviewer
objection X, not withdrawn: ..."`). So "make the control arm byte-identical to the pre-fix tree" is
not achievable through one application site. Reverting the wording under the control arm would give
the condition a second site; leaving it arm-dependent would mean the arms differ in prompt text as
well as in the release rule. The new wording is kept in **both** arms, which is true in both -- under
`resolved_or_withdrawn` a forced drop still comes only from an objection that is neither resolved nor
withdrawn -- and pinned by `test_the_drop_justification_is_identical_under_both_release_rules`. The
consequence has to be stated wherever the cell is quoted: the control arm reproduces the pre-fix
**release rule**, not the pre-fix **tree**, so comparisons to `2026-08-28_objection-routing.jsonl`
remain cross-commit and remain secondary. A one-line prompt change riding along with a predicate
change is exactly the kind of thing that turns a clean arm into an uninterpretable one, and it was
visible only by reading the diff rather than the code.

**What it costs, unhedged.** A permanent public axis whose off value is a known defect, so a reader
of `RunConfig` now sees five recorded conditions of which only four are design questions. Permanent
schema: every results row from here on carries `forced_drop_release`, forever, for one cell. A
misuse risk, that someone crosses it with another axis and publishes a 2x2 with a bug in one cell.
And a session, with Phase 4 untouched. The mitigations are the inverted default, the test pinning it,
the field description, a stderr warning on every run of the off arm, and this entry -- which closes
the axis: **`forced_drop_release` has exactly one legitimate use, it is the cell pre-registered in
`evals/results/LOG.md` for 2026-08-28, and no other arm may use it as a baseline.**

**Outcome, appended after both arms ran.** The axis paid for itself immediately and not in the
direction it was built to confirm. **The 5/10 -> 9/10 headline did not replicate: with a same-commit
control the effect is -1/10** (control 7/10, sticky 6/10), and the sticky arm did not reproduce its
own 9/10 either, coming in at 6/10. The pre-registered primary and its replication floor both fail,
and the line is retired from the results tables rather than defended. What replaced it is worth more
than it was: four cells now exist at the same nominal configuration, and **two running identical
behaviour returned 9/10 and 6/10**, so model nondeterminism alone moves a 10-run count on
`claims_timing` by about 3 -- most of the original effect. That is a fact about this benchmark's
resolution, it was invisible until something was run twice, and it puts error bars on every other
10-run comparison in the project including the routing arm's own 1/10 -> 5/10.

The mechanism, separately, is confirmed. `objections_falsely_resolved` is non-zero in 3 of 10
control rows and 0 of 10 sticky rows, matching 0/10 in the committed post-fix cell: resurrection
really happens under the old rule and the sticky rule really eliminates it. What is falsified is the
link from that event to the outcome -- the difference is not concentrated in the two-return rows
where resurrection must occur, and the reading that fits every number is that the pipeline recovers
on its own, the reviewer re-objecting to the re-admitted column on a later pass. Resurrection costs
a loop, not an outcome, which is why the control arm ran longer and dearer exactly as predicted and
still reached 7/10.

The fix stays in, on evidence that is not the primary: it removes a real defect, the guardrail shows
the sticky arm's matrix is *wider* rather than narrower (5.30 against 4.90), and `false_alarm_
standing` is lower. A pipeline that un-fixes itself is wrong whether or not the wrongness shows up
in a 10-run count -- the same argument this file already made for refusing a mislabelled split
manifest. What changed is the claim attached to the code, not the code.

So the reversal recorded above was correct on its own terms and for a reason better than the one
given: the axis was justified as the way to confirm a headline, and its actual value was to
**refute** one. Had it not been built, the project would have carried a causal claim into Phase 5
that a single replicate dissolves.

## 2026-08-29: a `block` with nothing to act on gets asked again, once, and the router's question
## moves onto the state where both callers can share it.
The zero-objection `block` bug -- the reviewer claims `block` while nothing survives for
`feature_eng` or `modeler` to act on, the router correctly refuses it, and the run reaches the
reporter having dropped nothing -- was the leading cause of failure at 1-2 runs in 10, and after it
drew 3-and-0 across the two arms of the forced-drop cell it was a measurement hazard as well: it
eats a cell's numerator without touching what the cell measures. The cause was never established.
Two teed logs existed and were not committed, so the standing hypothesis (objections raised, then
dropped one at a time by the malformed-objection filter) could not be checked this session. The fix
is therefore deliberately blind to the cause: `PipelineState.would_be_open` answers "what will still
be open once this pass lands", `router._route_for_block` now calls it instead of deriving the answer
itself, and the reviewer asks the same question one node earlier through `_nothing_to_act_on`. All
three ways the open set can end up empty -- every objection filtered away, none raised at all, this
pass's own dispositions closing the last one -- reach the reporter identically and are treated
identically. Sharing the predicate is the other half of the decision: a reviewer and a router
holding two independently maintained answers to "is this actionable" is the shape of the bug, and
the same dual-authority split `effective_target` and router invariant 3 already exist to close.

When the answer is "nothing", the node re-asks the model **once**, with a `retry_reason` and the
profile's `known_columns` added inside the JSON facts block (inside, because `tools/llm.py:_payload`
reads `find("{")..rfind("}")` and prose appended after the block reaches nobody). Three alternatives
were rejected. Repairing the claim to `pass` in the node would hide the failure in the one column
the eval reads, and would break the reviewer's invariant 2. Adding a rule to `REVIEWER_SYSTEM` would
break byte-identity of the `base` prompt against all 84 committed rows -- and the rule is already
there as bullet 5, so the model is violating an instruction it has rather than missing one. Looping
the retry would spend money to hear the same answer a third time. A second empty answer falls
through to the router's documented terminal-block path unchanged, which is why this is a retry and
not a repair.

Two consequences are recorded rather than mitigated. A run the retry rescues now reports
`errored: true`, because every retry appends a `PipelineError` -- the stable `block-retry` prefix is
how a cell counts firings and rescues off the new `errors` column, and there is deliberately no
`block_retries` field, since a counter derived by string-matching our own messages is exactly the
kind of metric `state.py`'s rule 2 rejects. And a malformed block on the final pass can now end
`pass` where it would have ended `exhausted`, so verdict distributions shift at this commit. This is
applied unconditionally rather than behind a `RunConfig` axis, unlike `forced_drop_release`: no
published claim rests on the buggy behaviour, nobody will want to measure "the retry moved
remediation by X", and the general instrument for a boundary like this is the `commit` field added
in the same session, not a condition per bug fix.

## 2026-08-29 (second entry): the row says what went wrong and which tree produced it, and the two
## new labels sit on opposite sides of the state boundary.
Three columns were added to `results_row()`. `errors` carries every `PipelineError` as node, message
and `recoverable`, because `errored` is a single bit covering every way a run can fail: a filtered
column name and a dead sandbox are the same value, which is why the zero-objection `block` bug had
to be counted by hand off `route_sequence == ["reporter"]` and why the only two logs that could have
explained it were never kept. It is always a list -- zero errors is a real 0, the same argument
`objections_resolved` makes, and `None` would be indistinguishable from a row written before the
column existed. The message is truncated at 500 characters on the row and left whole on the state:
the row is a line someone greps, and a model client's exception repr can carry an entire HTTP body.
`default_model` was simply missing while `reviewer_model` was present, so the first `--model sonnet`
arm would have written rows indistinguishable from every Haiku row.

`commit` is the interesting one, and it went on the frozen `RunConfig` rather than being annotated
by the harness at write time. The rule it follows is `RunConfig`'s own: a row must be
self-describing from the state alone, which is the property that stops a row being labelled by
something outside the run that could disagree with what actually ran. Which code produced a run is a
condition exactly like `naming` or `loop_cap` -- and it is the general instrument for a class of
difference no flag can express, which is every unconditional change this project has made. Every
code boundary reasoned about so far (the sticky-drop fix, the naming ablation's schema change, the 7
rows that cross the forced-drop boundary) was reconstructed from commit messages after the fact.
Putting it here also means `--results` rows and harness rows share one schema, so `results_row()`
stays the single definition of a row. `-dirty` is suffixed onto the same string rather than kept as
a second boolean, which keeps the cell key one value wide and makes a dirty run group as its own
cell in `eval-diff` -- correct, because a dirty run is not reproducible. `provenance.git_commit`
never raises: a provenance helper that can take a benchmark run down is worse than a null field.

The contrast is deliberate and it is the reason the two labels live in different places.
`cell`, `replicate` and `run_index` are facts about the *sampling design of an invocation*, not
about what the run did, and no node may ever be able to read which replicate it is in -- so those
stay write-time annotations passed through `_append_results_row(extra=...)`, above the state
boundary, while everything describing the run comes off the frozen config below it.

None of the three back-fill onto the six committed results files: only rows were committed, the
states they came from are gone, and results files are never edited by hand. For `commit`
specifically, `LOG.md`'s prose is the only provenance some earlier cells have -- the reviewer
ablation records `2d5c1fc`, the naming ablation records nothing.

## 2026-08-29 (third entry): the harness is replicate-aware before it is anything else, and the
## refusal to call a difference an effect lives in the tool rather than in a reader's discipline.
Phase 4 started with the harness rather than the manifest, because the manifest is blocked on an
open question carried since session 0 (which OpenML suite has citable published baselines) and
because the constraint the harness had to be built around was discovered the session before: a
single 10-run cell cannot resolve a 4-in-10 difference. That is a design constraint, not a caveat to
add later, and retrofitting it would have meant rewriting whatever got built first.

So `plan()` is replicate-major: every cell's replicate 1 before any cell's replicate 2. The
alternative, cell-major, is what put the Sonnet reviewer cells at n=4 and n=3 against a budgeted
n=8 -- a cap that binds partway through a cell-major plan starves the last cell entirely, and
unequal n is what made that 2x2 unquotable. Replicate-major truncates every cell by roughly the same
fraction instead. The cost cap is invocation-level and not per-cell, because replicate-major
ordering already solves the starvation problem a per-cell cap would have been for, and a second
budget is a second number to reason about; two budgets means two invocations. A run that raises is
counted and skipped rather than ending the cell (the parking-lot fix for `cmd_run --repeat`), and it
is charged its cell's ESTIMATE, reported separately as `charged_estimate_usd` -- the tokens it spent
are unrecoverable because the state that counted them never came back, and a cap that ignored failed
runs would let a cell failing late, after paying for most of a pipeline, spend without limit.
`SystemExit` still propagates: `cli` raises it to mean the run never started, and whatever stopped
run 7 from starting will stop run 8.

The refusal rules live in `evaldiff.py`, not in the harness and not in a reader's discipline.
Underpowered is checked before anything else: fewer than 2 replicates on either side prints
"underpowered" and no delta, which covers every row this project has committed, since none carry a
`replicate`. Then overlapping Wilson intervals print "not separating", and disjoint ones print
"separates at this n" -- never the word "effect" and never a p-value. A `None` metric is excluded
from the denominator and reported, never counted as a failure: `leakage_remediated` is `None` on an
empty matrix, and counting that as False would make a `feature_eng` crash look like a reviewer miss.
A cell present on one side only is reported rather than dropped, because silently dropping it is how
an arm loses its control. Worth recording that the cheap half of this would have been enough on its
own: pooled Wilson alone refuses the retired sticky-drop headline, since 5/10 is [0.237, 0.763] and
9/10 is [0.596, 0.982]. The replicate requirement is therefore not about narrowing the interval --
it is about testing the iid-Bernoulli assumption the interval rests on, which is why `Count` carries
per-replicate counts beside the pooled number.

`random_seed` never becomes a `Cell` field or a `run_eval` argument. It is a DATA seed -- the split
and the estimators -- so moving it would fold split variance into the number this project reports as
model variance and break comparability with all 84 committed rows. The honest limitation, recorded
rather than fixed: every rate here is conditional on one split, so "the pipeline remediates 70% of
the time" is a claim about this split, not about the fixture.

Two placements. The code lives in `src/ds_agents/harness.py`, not `evals/harness.py` as CLAUDE.md's
layout block said -- `evals/` is not a package and is not in the wheel, and `cmd_eval` has to import
it; CLAUDE.md is corrected in the same commit. And `harness` imports `cli` inside its functions
while `cli` imports `harness` inside `cmd_eval`, because `harness` needs `_run_once` and
`_append_results_row` while `cli` needs `run_eval`. The clean fix is a `runner.py` holding the
pieces both need. It is deferred deliberately: taking it now would churn three test modules in a
session already landing three things, and the deferred import costs nothing at a function call.

## 2026-08-31: the `ci` subset is pre-registered as a baseline rather than a hypothesis, and what a
## zero-count on the block-retry is allowed to mean is fixed before the run rather than after it.

The `ci` subset has been runnable since 2026-08-29 and has never been run. This entry pre-registers
the first live invocation, written and committed before the runner is called:

```
ds-agents eval --subset ci --name ci-baseline --replicates 2 --n 5 --max-cost-usd 1.00
```

30 runs, three cells (`toy-default`, `claims-opaque-which`, `reissued-opaque-which`), 2 replicates
of n=5 each, live Haiku over MCP, `random_seed=20260822`, on a clean tree at the commit this entry
itself creates -- `2b6a22e` plus this pre-registration, which touches `docs/` only, so the rows'
`commit` field is not `2b6a22e` and the tree that produced them is behaviourally identical to it.
Estimated $0.65
against a $1.00 cap; the estimate is three guesses that have never been checked, which is one of the
things the run is for.

**This is a baseline, not an ablation, and that changes what a pre-registration is for.** There is
no hypothesis and no contrast: the three cells differ in *fixture*, so any difference between them
is a fixture difference and is not evidence about anything else. Nothing here may be quoted as an
arm. What the pre-registration fixes instead is what the run is allowed to claim afterwards, which
matters more here than usual, because the most interesting endpoint is one where the tempting
reading of a null is wrong.

**Descriptive endpoints, per cell**, each reported as per-replicate counts beside a pooled Wilson
interval, never as a pooled count alone: `leakage_remediated`, `leakage_caught`, `reviewer_caught`,
`errored`, plus mean `cost_usd` and mean `review_loops`.

**The block-retry endpoint, and its null.** The zero-objection `block` retry landed at `1e5f30a` and
has never been observed firing live. Its base rate was 1-2 runs in 10 on `claims_timing`, so the
expected count here is roughly 1-3, and only the 10 `claims-opaque-which` runs are on the fixture
that produced that rate. Fixed in advance: **at least one `block-retry` in the `errors` column is
the first live evidence the path fires; zero is not evidence the bug is gone.** At a ~15% per-run
rate over the 10 runs that carry it, seeing nothing has probability around 0.2, which is an ordinary
outcome and not a result. A zero gets written up as "still unobserved" and the question stays open.
The two messages -- "retry produced N actionable objection(s)" versus "still produced nothing
actionable" -- are separate findings and are not to be merged into one count: the first says the
repair works, the second says the model meant it.

**Stopping rule.** If the cap binds, every cell is reported at its real truncated n and no cell is
topped up. The Sonnet cells on 2026-08-28 were topped up and that was defensible only because the
top-up was to a pre-registered n; topping up after seeing the numbers is what turns an n into a
choice. Three consecutive failures abort the invocation, which is `MAX_CONSECUTIVE_FAILURES` and not
a decision made here.

**Replicate check.** The replicate exists to test the iid-Bernoulli assumption the Wilson interval
rests on, not to narrow it. If between-replicate spread on any cell exceeds what binomial noise
allows, that cell's pooled interval is thrown out rather than reported narrower.

**What the run is expected to buy, stated so a later reader can see whether it did.** The first
replicate-aware cells in the repo, so that every remaining ablation has something `eval-diff` will
legally compare against; the first live rows for `reissued_ids`; measured per-run costs to replace
the three guesses in `SUBSETS`; and the block-retry observation above. It does not buy a CI
threshold. A threshold also needs a policy for who pays for a gate that spends API money per push,
and this repo has no `.github/` and no key available to Actions.

**Outcome, appended after the run.** 30 rows, $0.7291, 0 refused, 0 failed, cap not binding.
`evals/results/2026-08-31_ci-baseline.jsonl` and LOG.md 2026-08-31 carry the table; three things
here are decisions rather than numbers.

**The block-retry endpoint resolved in the direction the pre-registration could not assume.** 7 of
30 runs fired it, against an expected 1-3, and 4 of those 7 retries produced an actionable
objection. The pre-registration's work was done on the branch that did not happen: had the count
been zero, the write-up was already committed to "still unobserved" rather than "fixed". Worth
recording that the count landed high for a reason the pre-registration did not anticipate -- 5 of
the 7 were on `reissued_ids`, which never contributed to the 1-2-in-10 base rate the estimate came
from, so the estimate was extrapolated from the wrong fixture. A base rate is a property of a
configuration, not of a bug.

**Provenance moves from per-run to per-invocation.** `_run_once` called `git_commit()` for each run,
and the results file it writes is untracked, so run 0 recorded `8a629bf` and runs 1-29 recorded
`8a629bf-dirty`: the harness dirtied its own tree with its own output and then recorded the fact as
a run condition. Since `commit` is one of `evaldiff.CONDITION_FIELDS`, the effect was to split
`toy-default` into cells of n=1 and n=9. `commit` is now a required keyword on `_run_once` with no
default -- `None` is a legitimate value (git missing), so any sentinel default would be
indistinguishable from the answer it stands in for, and the bug was a default rather than a call
site. Both callers snapshot once before their first run. This is the same once-per-invocation rule
`_live_run` already applied to `materialize`, and the argument is identical: anything that must be
identical across an invocation's runs has to be read before the runs start changing it. The
committed rows are NOT back-fixed. They record what the run recorded, and a results file edited to
say something other than what happened is the one thing this project will not do.

**The baseline is not a comparand, and NEXT.md was wrong to say it would be.** `commit` being a
condition field means a future arm can never be `eval-diff`ed against this file, because a new arm
is almost always a new `Literal` on `RunConfig` and therefore a new commit. That is not a defect in
`eval-diff` -- it is the controlled-ablation rule enforced by the tool: **both arms of a comparison
run in one invocation at one commit**, as the forced-drop-release cell already did. What this file
is for instead is descriptive: the first replicated live characterization of the three `ci` cells,
the corrected `est_cost_usd` values, and the block-retry evidence. Future ablations must budget for
running their own control, which roughly doubles every remaining Phase 5 estimate a second time.

**One recorded failure, unfixed.** `claims-opaque-which` rep=1 idx=2 dropped every column, failed to
fit any candidate, and ended `review_verdict: "pass"` with `n_final_features: 0` and a null score.
`publishable()` admitted it. A run that produced no model is not a `pass` under any reading, and the
verdict derivation is what needs to say so -- it is left open rather than patched here, because
changing verdict derivation changes a column every committed row carries and that is its own cell,
not a footnote to this one.

## 2026-08-31: the benchmark manifest is generated, and a published number is not a baseline

`evals/datasets/manifest.yaml` exists. It carries 13 external binary-classification datasets from
the AutoML Benchmark -- Gijsbers et al., JMLR 25(101), 2024, whose classification tasks are OpenML
suite 271 -- and it closes the question carried open since session 0. AMLB was chosen over
OpenML-CC18 and Grinsztajn45 for one reason: it is the only one of the three that publishes
*baseline framework definitions* (a constant class-prior predictor at 0, a tuned RandomForest at 1)
rather than only a curated list of datasets. CC18 has no per-dataset baseline table; Grinsztajn45
rebalances classes and drops rows with missing values, so its CSVs are not the canonical ones and a
number computed on them is not comparable to anything published elsewhere.

**The manifest is generated, and that is enforced rather than promised.** `.claude/settings.json`
denies `Edit` and `Write` under `evals/datasets/`, so the file cannot be typed into -- it can only
be produced by `uv run ds-agents datasets refresh`, which runs as a program. That constraint was
discovered while planning and it turned out to be the right architecture rather than an obstacle to
route around. A manifest is exactly the shape of file an agent will happily fill with
plausible-looking baseline numbers; it is ground truth, so a wrong number in it is unfalsifiable
and would silently poison every table in the Phase 5 writeup. Under the generated-only rule, every
number in the file is either a field of an API response or a measurement taken on the fetched
frame, and every sentence a person chose lives in `src/ds_agents/benchmark.py` where review and the
fast-test hook can see it. Acquisition needed no new dependency: `sklearn.datasets.fetch_openml`
plus stdlib `urllib`. Only `pyyaml` was promoted from transitive to direct.

**A published score is recorded, and it is deliberately not `baseline_score`.** Each entry carries
a `published_reference`: the best AUC uploaded to that OpenML task, with the run id that produced
it, fetched from OpenML's evaluation API. That API is clean HTTPS and re-checkable, which AMLB's
own raw-results store is not -- `openml1.win.tue.nl` presents a self-signed certificate, and a
number this repo cannot re-fetch and re-verify is a number it will not publish. But a
`published_reference` was produced on OpenML's 10-fold cross-validation by a third-party flow,
while `verified_holdout_score` will be produced by this pipeline on a holdout this repo withholds.
Dividing one by the other and calling it `score_ratio` would attribute the difference between two
*protocols* to the difference between two *systems*. So the manifest carries two separate things
under two separate names: a `published_reference` that is informational and labelled with its
protocol on the value itself, and a `baselines` block that is a *definition* with no number in it,
naming what `baseline_score` must be computed from -- both AMLB points, fit on this repo's train
split and scored on the same withheld holdout as the run they are compared against. The separation
is pinned by an AST test asserting no code path in either module reads or writes `baseline_score`.

**The selection rule is data, and it was applied to measurements rather than to expectations.**
`SELECTION_RULE` is a typed constant written into the manifest and checked against it: binary
only, <=100k rows, <=200 features, <=5M cells, and >=5 features surviving `feature_eng`'s
mechanical filters. That last criterion is the one with teeth and it earned its place immediately.
The session's planning notes predicted a specific 12 datasets from recalled and API-read shapes;
the rule, run against what was actually fetched, returned 13 and disagreed about membership. Two
disagreements are worth recording. `blood-transfusion` has only 4 usable features and is out.
`amazon_employee_access` was excluded on the first build with **zero** usable features and admitted
on the second with nine -- because the first build measured the frame `fetch_openml` returns, where
its nine ID columns are high-cardinality `category`, and the second measured the CSV as re-read,
where they are `int64`. The CSV is what gets mounted at `$DS_DATASET`, so the second measurement is
the correct one and all measured fields now come from the re-read file. The same round trip moved
`kc1`'s `positive_class` from `'true'` to `'True'`; a manifest recording the former would have
named a positive class no run could ever match. Its nine integer-encoded ID columns still make
`amazon_employee_access` a plausible candidate for a degenerate run, which is an open question
rather than a reason to hand-adjust the rule.

**One documented leak landed, and the check that validates it caught a real error.**
`bank_marketing` carries `V12` under `known_leakage`, sourced to UCI's own statement that call
duration must be discarded for a realistic model. It is the project's first non-synthetic,
independently documented leak, against the standing weakness that every leakage finding so far is
on a fixture this repo wrote itself -- a reviewer that catches our traps may only be catching our
habits. It was first recorded as `duration`, the name UCI uses; OpenML data 1461 ships anonymised
headers and has no such column, so the entry named a phantom and could never have fired. The
identification of `V12` is measured, not recalled: it is the twelfth feature, matching UCI's
documented column order, its observed range is 0 to 4918 against UCI's documented duration maximum
of 4918, and the neighbours corroborate the offset (V6 goes negative as `balance`, V14 bottoms out
at the `-1` sentinel as `pdays`). The build now fails loudly when a `known_leakage` column is
absent from the frame. `known_leakage` is emphatically **not** promoted into
`planted_leakage_columns`, and `in_planted_leakage_columns` is a `Literal[False]` so it cannot be
without a schema change: that field is read as a *complete* list, and claiming completeness for a
real dataset would score every other genuinely-suspicious column the reviewer names as a false
alarm.

**`planted_leakage_columns` is now a complete list or nothing, and nine columns respect that.**
Before this session, an empty planted list made `leakage_caught` read `False`, `leakage_precision`
read `0.0`, and the four `*_false_alarm` columns count every flagged column as a mistake -- which
on an unlabelled benchmark dataset asserts both that the dataset contains no leak and that the
reviewer failed to find it, with no evidence for either. A single named gate,
`graded_for_leakage = bool(planted)`, returns `None` from all nine, generalising the rule
`leakage_remediated` and the three `*_recall` columns already followed. Observations are still
recorded: what was flagged and nominated stays on the row, so a benchmark run is readable by a
human even though no rate can be computed from it. This was safe to do now rather than later
because the precondition was proved rather than assumed -- all 145 committed rows in
`evals/results/*.jsonl` carry a non-empty `leakage_planted`, checked by a test that will fail if
that ever stops being true. No published number moves.

**What is deliberately not here.** No dataset in this manifest can be run. There is no `Runnable`
protocol over `Fixture | DatasetEntry`, no withheld holdout, and nothing computes
`verified_holdout_score` or `baseline_score` -- so `--subset full` still refuses, and its error
message was rewritten because the old one was wrong on both counts it named. Running the set today
would emit 13 rows whose headline column is null, which is money spent for no finding. Kaggle was
declined rather than deferred: it needs credentials and per-competition licence acceptance, and a
leaderboard score has no reproducible published protocol, which is the exact property AMLB was
chosen for.

## 2026-08-31 (third entry): the re-scorer is pre-registered, and the withheld holdout is carved
## before the graph starts because no split a node decided can grade the node that decided it

Written and committed before any of the code below exists. `verified_holdout_score` is the column
the whole benchmark half of this project reports, and a grading protocol chosen after seeing the
first number is not a protocol.

**What is fixed in advance.** The withheld fraction is **0.2**, stratified by the target, seeded
from `RunConfig.random_seed` (20260822). The carve happens **before the graph starts** and its rows
never enter `split_artifact`, the run's artifact store, or the mounted `$DS_DATASET`. Stratification
is numpy-only -- group positions by `str(label)`, permute each group with
`np.random.default_rng(seed)`, take `max(1, floor(0.2 * n))` where the group has at least two rows.
`train_test_split` is deliberately not used: a holdout whose membership depends on sklearn's
internal shuffling can change identity on a version bump, and this is the one partition every
headline number is measured on. The exact `credit_g` indices at that seed are pinned by test, so a
change is loud rather than silent.

**Fixtures do not get a withheld holdout, and that is a decision rather than an omission.**
`Runnable.withheld_fraction` is 0.0 for every fixture and 0.2 for every manifest dataset. Carving
20% out of a 200-row toy would change what the agents are shown, which changes what they do, which
makes all 145 committed rows incomparable to everything written after it -- for no gain, because
planted leakage is a *column* and a randomly withheld holdout still contains it. A leaky model's
claimed score does not collapse on those rows, so there is nothing there to catch. Widening the
carve to fixtures is a cell of its own if anyone ever wants it.

**The refit is the claim, so the refit gets a self-check.** No fitted model is persisted anywhere
-- `ModelResult.model_artifact` has been declared and never written since Phase 1 -- so scoring the
withheld rows means refitting. The recipe comes from `modeler.CANDIDATE_SPECS` with the same
`SEED_SENTINEL` substitution, **not** from `ModelResult.params`, which is a flat merge across every
pipeline step and is lossy by construction: `logistic_l2`'s `StandardScaler` contributes an empty
dict and is invisible in it. Before the withheld rows are touched at all, the same pipeline is fit
on `split["train"]` and scored on `split["holdout"]` -- the *agents'* own holdout -- and compared to
`claimed_holdout_score`. That single comparison validates the whole chain at once: transform
re-application, seed, positive-class resolution, scorer sign, split parsing. A disagreement beyond
`REFIT_TOLERANCE` (1e-6 absolute; same sandbox, same rows, pinned `random_state`, so a discrepancy
is a bug and not float noise) sets `rescore_status="refit_mismatch"` and the score is **kept and
flagged rather than deleted** -- deleting it would hide the finding, and the status column is what
says "do not pool this". Without this check the re-scorer would be a number computed on some rows;
with it, it is the modeler's model measured on rows it never saw.

**The re-scorer runs in a sandbox, through its own `LocalTools`, never in-process.** CLAUDE.md
forbids in-process `exec` and the 2026-08-26 entry gives the reason: it puts `PipelineState`,
`planted_leakage_columns` included, inside the agent's reach. The grading risk is gone by re-score
time, but the orchestrator still holds the API key, and `modeler.py` already execs the feature
artifact inside a snippet. Independence points the same way rather than the opposite: the sandbox
keeps no state between snippets and a second `LocalTools` gives the grader a store the run's store
cannot see. Transport is `local` even when the run used `mcp` -- MCP exists so *agents* get the
standard surface, and the grader is not an agent. One rule falls out of this: a `SandboxError`
during re-scoring is **caught, not propagated**. In a node it propagates so a broken machine is not
filed as an agent mistake; here the run has already produced everything it will, and losing the row
would drop exactly the rows a table needs.

**What "outside the run root" actually buys, stated honestly.** The sandbox has no filesystem
namespace, so a snippet could in principle open the withheld CSV by relative path. What the layout
gives is an *assertable* property -- no file the run's store holds contains a withheld row, checked
by test -- and not isolation the code does not have. Real isolation waits on the Docker backend
already parked behind `SandboxPool`.

**A rescore failure is not a run failure.** `rescore_status` is a thirteen-value enum
(`not_attempted`, `no_withheld_holdout`, `no_spec`, `no_split`, `no_feature_code`, `empty_matrix`,
`no_model`, `unknown_model_spec`, `single_class_holdout`, `snippet_failed`, `sandbox_error`,
`refit_mismatch`, `ok`) and nothing in it ever appends a `PipelineError`. `errored` means the *run*
went wrong; the block-retry session already recorded what overloading a column costs. This is the
`leakage_graded` fix applied to a score: an explicit "not graded, and here is which kind of not
graded" beside the null, rather than a bare null a reader has to interpret. `publishable()` is
unchanged, so a benchmark run with `rescore_status="no_model"` still writes its row -- ARCHITECTURE
requires a row even on hard failure, because losing the rows for the worst outcomes biases every
table upward.

**`baseline_score` is deferred to the next session, and the reason is not time.** The zero point is
not a design decision at all -- a constant class-prior predictor scores exactly 0.5 roc_auc by
construction, which makes it a correctness assertion on the re-scorer rather than a column. The
unit point is. AMLB publishes the *convention* (normalise from a constant predictor to a tuned
RandomForest); it does not publish a grid this repo can re-fetch, because `openml1.win.tue.nl`
presents a self-signed certificate. Any recipe written here is **ours**, untraceable in a way
nothing else in the manifest is, and it deserves its own pre-registration rather than a paragraph
at the end of a session that is already landing three modules. There is also a defect in the
column it feeds, found while planning and recorded now so it is not discovered after the number
exists: the baseline sees every column, **including a leak**. On a labelled dataset a pipeline that
correctly drops the leak scores *below* the baseline, so `score_ratio < 1` is evidence of good
behaviour there and of bad behaviour everywhere else. `score_ratio` cannot be pooled across
labelled and unlabelled datasets without saying which is which, and that is a bigger reason to take
another session over it than the cost is.

**What the first live numbers are allowed to claim.** One `credit_g` run plus a 4-row
`bench-smoke` invocation, capped at $0.25. That is a **write-path proof and a first
characterisation, not a cell and not a comparand**: `commit` is an `eval-diff` condition field, so
both arms of any comparison have to run in one invocation at one commit. A further limit is fixed
here rather than discovered later -- 200 withheld rows at a 30% positive rate put the standard
error of roc_auc near 0.04, so a `holdout_claim_gap` below roughly 0.08 is not distinguishable from
noise at n=1. That is the "no 10-run count without a replicate" lesson arriving on a continuous
column, and it applies to the first gap this project measures.

**Outcome, appended after the runs.** Everything above was written before the code. What the runs
returned: `rescore_status` is `ok` on 4 of 4 `credit-g-smoke` rows and on the standalone live run,
and `refit_claim_gap` is **exactly 0.0** on every one of them — the harness's refit reproduced the
modeler's own claimed score on the agents' own holdout to within 1e-6, so `REFIT_TOLERANCE` was not
merely unviolated, it was never approached. The instrument's sensitivity is established by
construction rather than by these rows: `tests/test_rescore.py` builds a dataset whose one
informative column is informative in exactly the rows the agents get and pure noise in the rows they
are graded on, and measures claimed 1.0 against verified 0.45. Without that test the whole module
would be a number generator with no evidence it would ever disagree with the agents.

**One thing the pre-registration did not anticipate, and it is a limit rather than a result: all
four rows are numerically identical in every column.** Same claimed score, same verified score, same
20 features, same `pass`, same single loop. The two replicates therefore bought no variance estimate
at all — they are four samples of one deterministic outcome, not four draws. The cause is visible in
the row: `credit_g` has no planted leak, the reviewer raised zero objections and `feature_eng`
dropped nothing, so there was no decision available to be made differently. That is a genuine
contrast with `claims_timing`, where nondeterminism alone moves a 10-run count by about 3, but it is
a fact about this dataset and not about the pipeline, and the measured +0.0044 gap sits at a fifth
of the pre-registered noise floor. It is not evidence the agents did not overstate themselves; it is
evidence this dataset gave them no opportunity to.

**Two things changed from the plan while building.** `NEXT.md` asked for a `Runnable` *protocol*
and what landed is a concrete adapter with two named constructors. A structural `typing.Protocol`
would be unchecked — the dev dependencies are pytest and ruff, there is no type checker — and the
thing it would document is exactly the guardrail `DatasetEntry`'s docstring says must be a property
rather than a promise. Under the adapter you cannot reach `_run_state` from a benchmark dataset
without passing through the one reviewed line that writes `planted_columns=[]` on purpose. And
`cmd_run --repeat 1` stopped collapsing the run root into the invocation root. That was tolerable
while the invocation root held only `input/`; it stopped being tolerable when the carve put
`withheld/` beside it, because "the withheld rows are outside every run root" is a property a test
has to check and it cannot check a layout that is sometimes one shape and sometimes another. No
path reaches a results row, so nothing published moves.

**`bench-smoke`'s cost estimate was a guess and is now a measurement**: 0.040 became 0.017, wrong by
more than a factor of two and wrong in the cheap direction. That is the argument for measuring the
other twelve before `full` runs rather than extrapolating from this one — a factor-of-two error on
`credit_g` at 1000 rows says nothing useful about `higgs` at 98k.

## 2026-08-31 (fourth entry): the baseline is two points and not one number, and `score_ratio` is
## retired because a single column cannot say which end of a scale it is

Written before the eval invocation below, and before any cell was funded. The re-scorer's own
pre-registration deferred `baseline_score` for two named reasons and this entry settles both.

**`score_ratio` is retired, and so is `baseline_score`.** `PipelineState.score_ratio` computed
`verified / baseline`. AMLB's convention is an affine normalisation, `(x - zero) / (unit - zero)`,
and those are different functions that disagree about what `1.0` means. What ships instead is
`baseline_zero_score`, `baseline_unit_score`, `baseline_status`, `baseline_detail`,
`baseline_recipe`, and one computed `baseline_normalised_score` built from the two points. The
alternative -- keep `baseline_score` as the unit point and gate `score_ratio` on `leakage_graded` --
was the smaller change and was rejected: it preserves a singular `baseline_score` whose meaning a
reader cannot recover from the row, which is the exact defect the whole exercise is about. Retiring
a published column is only free if nothing was published in it, and that is checked rather than
claimed: **all 149 committed rows across nine results files carry both as `null`**, now pinned by
`tests/test_evaldiff.py::test_no_committed_row_ever_carried_a_retired_score`. No committed file was
edited. Both retired names stay in the AST ban list in `tests/test_benchmark_manifest.py`, because a
resurrection would be a regression rather than a feature.

**The pooling hazard is closed by a gate, and the gate protects a case that is not yet reachable.**
`baseline_normalised_score` returns `None` whenever `planted_leakage_columns` is non-empty. The
baseline is fit on every raw column including the trap, so on a labelled dataset a pipeline that
correctly drops it scores *below* a baseline that kept it -- the same number meaning opposite things
on labelled and unlabelled rows. Worth stating plainly: **that contradiction cannot fire today.**
Fixtures have `withheld_fraction = 0.0`, so no fixture row is graded at all and every row that can
carry a baseline is a benchmark row with an empty planted list. The gate is written now because it
costs one `if` now, and because DECISIONS already parks "widening the carve to fixtures" as a future
cell -- at which point the contradiction would fire silently, on rows nobody re-read. The two raw
points are NOT gated: they are honest measurements, and a reader who knows about the trap can use
them. This is `graded_for_leakage` applied for the mirror-image reason.

**The unit point's recipe is ours, and it is versioned on the row.** AMLB publishes the convention
and not a grid this repo can re-fetch -- `openml1.win.tue.nl` presents a self-signed certificate --
so `rescore.BASELINE_SPECS` is a RandomForest at `n_estimators=200, max_features="sqrt",
min_samples_leaf=1, n_jobs=1`, chosen here and labelled as ours everywhere it is described. Fixed
rather than tuned: a tuned baseline needs a search space and a validation protocol, and each is
another invented number nobody can cite. `n_estimators=200` rather than 500 is a cost decision, not
a statistical one -- thirteen datasets up to 98k rows, and the unit point is a yardstick rather than
a competitor. `baseline_recipe` ("rf-v1") goes on every row that attempted the fit, because two runs
graded against different yardsticks must not be pooled and a reader holding a results file cannot
see a commit.

**The baseline sees the raw columns, through the grader's own encoder.** Not the agents'
`feature_code_artifact`. A yardstick that inherits the decisions it is measuring cannot say whether
those decisions helped -- it would answer "was the promoted model the right one", a modeler
question, rather than "did the team beat a reference system". The encoder is mechanical and
deliberately dumb: median-impute numerics, ordinal-encode categoricals with `unknown_value=-1` so a
level present in the withheld rows and absent from train has somewhere to go. It imposes a false
ordering on nominal codes and keeps high-cardinality columns `feature_eng` skips, which makes the
unit point a **floor rather than a ceiling** -- stated here so nobody later reads a run beating it
as beating a strong model. Both points are fit on the agents' own `split["train"]`, not on every
non-withheld row: giving the baseline more data than the model got would bias the comparison against
the agents.

**The baseline runs in its own process, with its own timeout and its own enum, and that is the
design.** A RandomForest that dies on a wide frame must not take `verified_holdout_score` with it.
In one process those two failures are the same exit code and telling them apart means reconstructing
from tagged stdout a distinction the OS just erased. `BaselineStatus` is a ten-value enum whose
load-bearing member is `unit_point_failed`, which **keeps `baseline_zero_score`** -- discarding it
would hide that the scale has a floor and no ceiling. `rescore_unavailable` is the coupling that
does exist, named rather than hidden: the baseline shares the split and the withheld rows, so nearly
every reason the re-scorer could not run binds it too, and the detail says which. Two things are
deliberately not statuses. A degenerate scale -- the unit point level with or below the zero point --
stays `ok`, because both points really were measured and that is a finding about the dataset rather
than a failure to measure; only the quotient is withheld. And a planted leak leaves both raw scores
untouched.

**One docstring was wrong and the difference form is why it stopped mattering.** The retired
`score_ratio` claimed "the predict-the-mean baseline for r2 is exactly 0.0" and needed a guard for
it. Measured: it is slightly **negative**, because a `DummyRegressor(strategy="mean")` predicts the
*train* mean while r2's denominator is the holdout's variance about its own mean. `(v-z)/(u-z)`
handles that with no special case. The guard that replaced it is a different one and is not
cosmetic: `separation <= eps`, not `abs(separation) < eps`, because a **negative** separation means
the RandomForest did worse than the class prior and inverts the axis -- a run that beat the prior
would read negative and a reader would take that for "worse than the prior".

**Sensitivity is proved by construction, as it was for the re-scorer.** `_write_split_leak` cannot
separate the baseline from the pipeline: under `StubModel` nothing is nominated and `feature_eng`
drops nothing, so both see the same columns and a grader that reported the pipeline's number twice
would pass. The separating mechanism is `MAX_ONE_HOT_LEVELS = 20`.
`tests/test_rescore.py::_write_high_cardinality_signal` builds a 60-level *string* column whose
level index is the target: `feature_eng` skips it as un-one-hot-encodable, so the pipeline is fit on
noise alone, while the grader's `OrdinalEncoder` keeps it and the forest finds the threshold. The
mirror image is tested too, on `_write_split_leak`, where the baseline is fooled by the leak exactly
as the pipeline is -- so the instrument is not "the RandomForest always wins".

**What was checked before anything was spent, and what that costs this pre-registration.** One
`ds-agents run --dataset credit_g` was made first, deliberately, to protect the cap against a
baseline that could not run on a real dataset at all. It returned `baseline_status: ok`,
`baseline_zero_score` **exactly 0.5**, `baseline_unit_score` 0.7660, `verified_holdout_score` 0.7476
(identical to the previous session's), `baseline_normalised_score` 0.9309, at $0.0168. So the zero
point identity is stated here as an assertion already confirmed offline and once live, **not** as a
prediction this entry gets credit for, and the cell below is not allowed to claim it as one.

**What the cell IS allowed to claim, and the endpoints fixed in advance.** One `--subset
bench-smoke` invocation, `--max-cost-usd 0.10`, 2 replicates x n=2. It is a **write-path proof and a
first characterisation, not a comparand** -- `commit` is an `eval-diff` condition field, so both
arms of any comparison must run in one invocation. Endpoints: (1) `baseline_status` is `ok` on 4 of
4 rows; (2) `baseline_zero_score` is exactly 0.5 on every row, which is a correctness assertion on
positive-class resolution, scorer sign and row selection all at once, and any other value is a bug
rather than a measurement; (3) `baseline_recipe` is `rf-v1` on 4 of 4; (4) spend lands near the
measured $0.0168/run, because **the baseline costs no tokens** -- it is pure sandbox compute -- so a
material overrun would mean the forest is being charged somewhere nobody expected. The wall time of
the two extra fits is recorded as the first data point for the per-dataset cost estimate that is now
the ONLY remaining `--subset full` blocker. **A fifth number is explicitly not an endpoint**: whether
`baseline_normalised_score` lands above or below 1.0 is a fact about `credit_g` and this recipe, at
n=1 effective, on a dataset already known to produce numerically identical rows. It is a
characterisation and will not be quoted as a result.

**What is deliberately not here.** `--subset full` still refuses, and its message was rewritten for
the third time -- the blocker is now only a measured cost per dataset. `benchmark.py`'s
`BASELINE_DEFINITION.note` and `benchmark_build.py`'s `HEADER` still name `baseline_score` in prose,
and are left alone on purpose: both are rendered into `evals/datasets/manifest.yaml`, which only
`ds-agents datasets refresh` may write, and a refresh re-fetches every OpenML response and can move
`published_reference`, which takes the max over uploaded runs. Changing prose is not worth a
silently-moved citation. The rename rides the next planned refresh; no offline test compares the two,
so nothing is red in the meantime.

**Outcome, appended after the runs.** Everything above was written and committed before the runner
was called, on a clean tree, at `2ad4a37`. All four pre-registered endpoints passed:
`baseline_status` is `ok` on 4 of 4, `baseline_zero_score` is **exactly 0.5** on every row,
`baseline_recipe` is `rf-v1` on 4 of 4, and spend was $0.0161/run against a measured $0.0168 --
marginally *cheaper*, confirming the two extra fits cost no tokens.
`evals/results/2026-08-31_baseline-smoke.jsonl`, 4 rows, $0.0645 against a $0.10 cap. The
characterisation, which is explicitly not an endpoint: unit point 0.7660, verified 0.7476,
normalised 0.9309. `credit_g`'s four rows are identical in every substantive column again, at a
second commit -- the only field that varies is `cost_usd` -- so the previous session's "four samples
of one deterministic outcome" reproduces, and the open question of whether that is the dataset or
the pipeline still needs a second manifest dataset.

**One thing the pre-registration did not anticipate, and it changes what `--subset full` costs.**
The baseline's wall cost is invisible at `credit_g` and dominant at `higgs`. Run-to-run LLM latency
spread is about 4 seconds, which swamps a 0.20s fit, so the unit point was timed directly at three
shapes: 0.20s at `credit_g` (1000x20), 9.85s at `adult` (48842x14), **72.09s at `higgs`
(98050x28)** -- against an LLM portion of roughly 21 seconds per run. So on the largest dataset in
the manifest the yardstick would roughly quadruple wall time, and none of that is visible from the
cheapest one. This is the same lesson `bench-smoke`'s factor-of-two cost error taught, arriving on
wall clock rather than dollars, and it is now a concrete input to the per-dataset estimate that is
`full`'s only remaining blocker. `BASELINE_TIMEOUT_S = 900` is comfortable against 72s. If wall time
becomes the binding constraint, `n_estimators` is the lever and `baseline_recipe` is what makes
pulling it visible in the data rather than only in git.

**`eval-diff` refused to compare the two `credit_g` files, which is the rule working.** Both cells
are `dataset_id=credit_g` at byte-identical conditions and they still separated, on `commit` alone.
Each side reported "only in before / only in after -- not compared". That is the pre-registered
reason no smoke in this project is a comparand, demonstrated rather than asserted.

## 2026-09-01: four of the thirteen datasets cannot be run, and the column that would have priced
## the rest could not see the thing it was pricing

Written and committed BEFORE the runner was called, on a clean tree. The outcome is appended at the
bottom in a second commit.

This session set out to do what `docs/NEXT.md` asked -- measure a mid-sized dataset before funding
thirteen -- and found two things first that change what the measurement can be.

**1. `wall_seconds` never included the grader.** `ended_at` is stamped inside `run_pipeline`
(`graph.py`), and `rescore.rescore` and `rescore.baseline` both run after it returns, in
`cli._run_once`. Neither `RescoreOutcome` nor `BaselineOutcome` carried a duration and there was no
`perf_counter` anywhere in `rescore.py`. So the term `--subset full` is blocked on pricing -- a
unit-point fit quoted at 72s on `higgs` -- was structurally invisible in every row this project has
ever committed. It was worse than invisible: the four `baseline-smoke` rows read 19.65-21.00s while
the four `credit-g-smoke` rows that did two FEWER fits read 22.79-27.34s, and the difference is
run-to-run LLM latency spread of about 4s. Anyone reading those two files together would conclude
the baseline made runs faster.

The fix is three derived columns and no node changes: `rescore_seconds` and `baseline_seconds`,
timed at the one call site that can see both halves and threaded through `rescore.apply` (already
the single writer of grader results), plus `node_seconds`, a `{node: seconds}` map summed over
repeats from the `NodeEvent.wall_seconds` that `NodeEvent` has computed since Phase 1 and that only
`ds-agents run` ever printed. Two grader columns and not one because they are two processes with two
timeouts that fail independently -- the entire point of `BaselineStatus` is that the yardstick can
die without taking the measurement with it, and a single `grader_seconds` could not say which half a
stall was in. `None` rather than `0.0` on the precondition path, because "did not run" and "ran
instantly" are different claims.

**2. Four of the thirteen manifest datasets cannot complete a run, and this is a SECOND blocker on
`--subset full` that is code rather than money.** The profiler writes the split as a JSON artifact
holding every row index -- `train`, `holdout`, and five folds of (train, valid), so roughly six
times the agent row count in integers -- and `read_artifact` caps every read at
`DEFAULT_READ_BYTES = 1 MiB`. Measured by rebuilding the real manifest from the cached CSVs:

| dataset | rows | split manifest | vs cap |
|---|---|---|---|
| amazon_employee_access | 32769 | 862,321 B | 0.82x |
| nomao | 34465 | 910,064 B | 0.87x |
| bank_marketing | 45211 | 1,210,808 B | **1.15x** |
| adult | 48842 | 1,312,688 B | **1.25x** |
| numerai28_6 | 96320 | 2,641,714 B | **2.52x** |
| higgs | 98050 | 2,690,410 B | **2.57x** |

The failure is quiet, which is why nobody caught it. On a truncated read `feature_eng` refuses with
`recoverable=False` and `modeler` refuses identically -- but nothing in `graph.py` or the router
branches on `recoverable`, so the run continues through reviewer and reporter, spends a full run's
tokens, and `publishable()` accepts the row. **A dataset that cannot be run does not look like a
failure. It looks like a measurement.** `errored` is `true`, which is the one column that catches it,
and `errored` being uninformative is already a standing NEXT.md item.

`adult` was the dataset `docs/NEXT.md` named first as the second point on the wall-clock curve. It is
the wrong choice and could not have produced a number.

This is pinned by `tests/test_split_manifest_size.py`, which projects the manifest size from
`n_rows` arithmetically -- no CSV is read, and the projection was checked against the real thing on
nine datasets and is within 0.15%. Three tests: no subset may name a dataset that would truncate
(proved to fire by temporarily adding `adult`), the unrunnable set is exactly those four, and the
cliff sits between `nomao` and `bank_marketing`. The second fails the day someone fixes the
representation, which is the only mechanism that has ever got `_resolve_subset`'s message corrected.

**The `bench-mid` subset is therefore a 2x2 and not a line.** Two axes were being treated as one:
wall clock tracks rows, token cost tracks columns, and `credit_g` is small on both.

|  | few columns | many columns |
|---|---|---|
| **few rows** | phoneme 5404x5 | jasmine 2984x144 |
| **many rows** | amazon_employee_access 32769x9 | nomao 34465x118 |

Crossed rather than sampled, because with only a wide-and-tall cell a timeout there is
uninterpretable -- `jasmine` vs `nomao` separates width from length, `phoneme` vs `amazon` separates
length from width. `amazon` and `nomao` are the two largest datasets in the manifest that can still
complete a run. All four `est_cost_usd` are GUESSES, unlike every other entry in `SUBSETS`, and
replacing them is the point of the run.

**The shape-timing table is now reproducible, and it was measured at the wrong shape.** The old
table exists only as prose in three documents; `ef15609` touched no code. It also timed the full
frame, when the baseline fits on `SPLIT["train"]` -- the 80% train split of the 80% agent frame, so
**0.64 x n_rows**. `tests/test_baseline_cost.py` (opt in with `DS_AGENTS_TIMING_TESTS=1`, gated by
env var and not a marker, following the `network` precedent) fits `rescore._RF` at every manifest
shape and asserts each clears `BASELINE_TIMEOUT_S` with 5x headroom. Measured:

| dataset | fit rows | seconds | % of budget |
|---|---|---|---|
| credit_g | 640 | 0.20 | 0.0% |
| jasmine | 1910 | 1.72 | 0.2% |
| amazon_employee_access | 20972 | 7.72 | 0.9% |
| adult | 31259 | 14.94 | 1.7% |
| nomao | 22058 | 34.98 | 3.9% |
| numerai28_6 | 61645 | 45.30 | 5.0% |
| higgs | 62752 | 57.18 | 6.4% |

These are an UPPER BOUND, not a prediction: the frames are Gaussian noise with one weak signal, which
is near worst case for tree depth. The table disagrees with the prose one in both directions --
`higgs` 57.18s against a quoted 72.09s (the old number used 1.56x too many rows), but `adult` 14.94s
against a quoted 9.85s despite fitting on 36% fewer rows, which is what synthetic noise costs.

**This answers one of NEXT.md's open questions and it is not the answer that was expected.** "Is
`higgs` affordable, and at what recipe?" -- yes, at `rf-v1`. The slowest shape in the manifest uses
about 6% of the timeout. `n_estimators` does not need to drop for large frames and `baseline_recipe`
does not need to fork, so the "two yardsticks" hazard that question worried about does not arise.
What blocks `higgs` is the split manifest, not the forest.

### Pre-registered endpoints

Invocation: `--subset bench-mid --name bench-mid --replicates 2 --n 1 --max-cost-usd 0.50`, 8 runs,
$0.35 estimated.

1. **Feasibility, primary.** All 8 runs produce a publishable row with `errored: false`. A
   `rescore_status` of `no_split` or `no_feature_code` on any row means the size arithmetic above is
   wrong and the unrunnable set is larger than four; response is to STOP the invocation rather than
   fund the second replicate.
2. **`rescore_status` and `baseline_status` are `ok` on 8 of 8**, and `baseline_zero_score` is
   **exactly 0.5** on every row. The zero point is a correctness assertion, not a measurement with a
   tolerance, and it has only ever been checked on one dataset.
3. **Four measured per-dataset costs**, replacing four guesses. A cell whose measured mean exceeds
   its estimate by more than 2x is reported as an estimate failure, because that is precisely the
   error `bench-smoke` made and the reason this run exists.
4. **The two axes separate.** Predicted so it can fail: `cost_usd` is driven by columns, so
   `jasmine` and `nomao` are the dearest and land within ~25% of each other despite an 11.5x row
   difference; `node_seconds` and `baseline_seconds` are driven by rows, so `amazon` and `nomao`
   are the slowest. If `jasmine` is also slow, or `amazon` is also dear, the axes are not
   independent and `full` needs one cost term rather than two.
5. **Determinism, the standing question.** `credit_g` returned four identical rows at two separate
   commits. Two replicates cannot settle it, so this is reported as directional, not as a finding.
   What would be informative is variation on the wide cells, where `feature_eng` has real decisions
   available.
6. **The floor-vs-pipeline asymmetry becomes reachable.** `MAX_ONE_HOT_LEVELS = 20`, and the grader's
   encoder ordinal-encodes what `feature_eng` skips. `credit_g`'s widest nominal has 10 levels, so
   nothing was ever skipped and the parking-lot note "the unit point is a floor, not a ceiling" has
   been latent in every committed row. `amazon_employee_access` is nine integer-encoded
   high-cardinality ID columns, which land in the baseline's NUMERIC branch and pass straight
   through. Pre-registered: `baseline_normalised_score` on `amazon` is expected to be depressed
   relative to `credit_g`'s 0.9309 for that structural reason rather than because the pipeline did
   worse.
7. **Not endpoints.** The absolute value of `baseline_normalised_score`, `verified_holdout_score` or
   `holdout_claim_gap` on any of the four. n=2 per cell, one commit, the grid is ours and the unit
   point is a floor. Characterisation only. No comparison against the committed `credit_g` rows:
   `commit` is an `eval-diff` condition field and this is a different commit, so `eval-diff` will
   refuse, deliberately.

### Failure endpoints

- **A modeler timeout on `nomao` but not `jasmine`** means the binding term is rows x width in
  `permutation_importance` -- 10 repeats x ~118 columns x 2 candidates, all inside ONE `run_python`
  call with ONE 240s budget. Projected work is 6.5M row-predicts on `nomao` against 0.69M on
  `jasmine` and 0.47M on `amazon`. Response is NOT to raise `MODEL_TIMEOUT_S` reflexively: lowering
  `N_PERMUTATION_REPEATS` changes the numbers `top_importances`, `reviewer_nominated` and
  `leakage_caught` are all computed from, so it needs its own axis and its own version string the
  way `baseline_recipe` has one, or rows at two repeat counts get pooled. That is a session, and the
  finding would be that `full` is blocked on a third thing.
- **A timeout on both wide cells** means width alone, and `SELECTION_RULE.max_features = 200` -- set
  without a measurement -- is back on the table.
- **A profiler timeout on `nomao`** (`PROFILE_TIMEOUT_S = 60`) would implicate the per-column mutual
  information loop, a term nobody has costed. Raising it is defensible because no recipe moves.
- **One risk measured and closed before the run.** `ArtifactStore.register_dataset` copies the CSV
  and does an unbounded `read_csv` plus a per-column `nunique` on every run, and it runs inside
  `_select_tools` BEFORE the MCP initialize handshake completes, against `_CONNECT_TIMEOUT_S = 60`.
  A `SystemExit` there is re-raised by `run_eval` and aborts the WHOLE invocation rather than one
  run, so it is the one path here where a single slow dataset kills everything. Measured
  2026-09-01: 0.01s at `phoneme`, 0.02s at `jasmine` and `amazon`, **0.17s at `nomao`** (22 MB, 118
  columns), and it runs three times per benchmark run -- the run's own tools plus a fresh
  `LocalTools` inside each of `rescore` and `baseline` -- so about half a second at the worst cell.
  Negligible, and the NEXT.md parking-lot item that asked for this to be measured before a large
  frame is answered for every dataset that can actually be run.
- **Note what will NOT fire.** `MAX_CONSECUTIVE_FAILURES = 3` counts only runs that RAISE. A node
  timeout does not raise, so none of the harness's brakes stop the failure this probe is looking
  for; `errored` on the row is the only signal, which is the third worked example of the
  `errored`-needs-a-companion item.

### Outcome, appended after the runs

Everything above was written and committed before the runner was called, on a clean tree, at
`b39a4c0`. `evals/results/2026-09-01_bench-mid.jsonl`, 8 rows, **$0.2120 against a $0.35 estimate
and a $0.50 cap**, 0 refused, 0 failed, nothing stopped early.

| cell | shape | cost | est | wall | rescore | baseline | modeler | verified | normalised |
|---|---|---|---|---|---|---|---|---|---|
| phoneme-narrow-short | 5404x5 | $0.0102 | 0.020 | 18.6s | 1.8s | 0.6s | 9.6s | 0.9466 | 0.9925 |
| jasmine-wide-short | 2984x144 | $0.0419 | 0.060 | 119.6s | 1.6s | 0.4s | **108.2s** | 0.8610 | 0.9328 |
| amazon-narrow-tall | 32769x9 | $0.0130 | 0.030 | 19.3s | 0.6s | 3.5s | 8.0s | 0.8126 | 0.8751 |
| nomao-wide-tall | 34465x118 | $0.0409 | 0.065 | 96.9s | 2.5s | 4.2s | **74.5s** | 0.9954 | 1.0040 |

**Endpoints 1, 2 and 3 passed.** All 8 rows publishable with `errored: false`; `rescore_status` and
`baseline_status` `ok` on 8 of 8; `baseline_zero_score` **exactly 0.5** on 8 of 8 and
`baseline_recipe` `rf-v1` on 8 of 8, so the correctness assertion that had only ever fired on
`credit_g` now holds on four more datasets. `refit_claim_gap` is **exactly 0.0** on every row, which
is the self-check that earns the withheld number, now proved on four datasets nobody here wrote. No
cell exceeded its estimate; every one came in UNDER, between 0.43x and 0.70x. That is the safe
direction and still a 40% miss, recorded as such.

**Endpoint 4 FAILED, and it is the finding of the session.** The pre-registration predicted two
independent axes: token cost driven by columns, wall clock driven by rows. The first half is
confirmed almost exactly -- `jasmine` $0.0419 and `nomao` $0.0409 land **within 2.4%** of each other
despite an 11.5x row difference, and the two narrow cells sit together at $0.0102 and $0.0130. Cost
is a function of width and essentially not of length.

The second half is **wrong**. Wall clock is not driven by rows either. `jasmine` -- 2984 rows, the
second-smallest dataset in the manifest -- is the **slowest cell in the run at 119.6s**, beating
`nomao` at 11.5x its size, while `amazon` at 32769 rows finished in 19.3s. `node_seconds` says why
in one line: the modeler takes **108.2s on jasmine and 8.0s on amazon**, because
`permutation_importance` costs `N_PERMUTATION_REPEATS x n_columns` scoring passes per candidate.
**Both axes are width.**

So the term this whole phase has been worrying about was the wrong one. The baseline -- the 72s
`higgs` number that motivated the deferral, the recipe fork, and the `baseline_recipe` column -- is
the one thing here that IS row-driven, and it is small: 0.4s on `jasmine`, 4.2s on `nomao`, never
more than 4% of a run. `--subset full` should be priced as **`a + b x n_features`**, one term, not
two, and the risk at scale is `MODEL_TIMEOUT_S`, not `BASELINE_TIMEOUT_S`. At 144 columns the
modeler already uses 45% of its 240s budget; `SELECTION_RULE.max_features = 200` -- set without a
measurement -- is what stands between the manifest and a timeout, and it now has a number behind it
for the first time.

**Endpoint 5, determinism: three cells identical, one not, and the exception is informative.**
`phoneme`, `jasmine` and `amazon` are byte-identical across both replicates on every substantive
column, reproducing `credit_g`'s result on three more datasets. `nomao` varies on
`profiler_nominated` -- one replicate nominated `V1, V7, V97, V100` as leakage candidates and the
other nominated nothing -- while still landing on the same 118 final features. **And the pre-run
smoke of `nomao` at this same commit dropped five columns and finished with 113.** So there are
three distinct outcomes on one dataset at one commit, and the "identical rows" result is narrower
than it looks: it holds where the pipeline has no decision available to make differently, and
`nomao` is the first dataset here where it does. This is still not a determinism finding at n=2; it
is a reason to stop treating the identical-rows observation as a property of the pipeline.

**Endpoint 6 confirmed in direction.** `amazon_employee_access` returns the **lowest**
`baseline_normalised_score` of the four at 0.8751, against `phoneme` 0.9925, `jasmine` 0.9328 and
`nomao` 1.0040. It is also the only cell where `feature_eng` dropped columns the grader kept -- 7
final features from 9 -- which is precisely the floor-vs-pipeline asymmetry the parking lot
describes, reaching a results file for the first time. `nomao` above 1.0 means the pipeline beat a
raw-column RandomForest there; at n=2 on our own grid that is characterisation, not a result.

**What did not happen.** No node timed out, so none of the failure endpoints fired and
`MAX_CONSECUTIVE_FAILURES` was never tested. `register_dataset` cost what it was measured at.
`holdout_claim_gap` is small and positive on all four (+0.0112, +0.0028, +0.0013, +0.0001) on 596 to
6892 withheld rows -- far more rows than `credit_g`'s 200, so the standing "a gap under ~0.08 is
noise at n=1" caveat binds much less tightly here, though these are still not gaps worth
interpreting.

**Two columns earned their place immediately.** `wall_seconds` alone would have reported `jasmine`
at 119.6s and said nothing about why; `node_seconds` attributed 90% of it to one node. And the
grader's own cost -- invisible before today -- turns out to be 3 to 7 seconds a run, which is 16% of
`phoneme` and would have gone on being absent from every cost model built from these files.

## 2026-09-01 (second entry): the split manifest is one character per row, and the complement rule
## stops being an assumption

The blocker the entry above names is closed. The split manifest no longer lists row indices; it
carries an `assignment` string of one character per agent row -- `h` for a holdout row, `0`..`4` for
the fold that row *validates* in -- and every other partition is derived. Train is every non-`h`
position. A fold trains on every train row it does not validate on.

Measured on the real thing, through the real store: `higgs` goes from 2,690,410 B to 78,831 B, which
is 7.5% of `DEFAULT_READ_BYTES`. All four datasets that could not previously complete a run now do,
offline and for nothing, at 36,553 B (`bank_marketing`) to 78,831 B (`higgs`). The headroom runs to
roughly a million agent rows.

**This is not a cap bump, and the reason is that the representation was never private to the artifact
store.** The split JSON text is substituted verbatim into FOUR snippet sources -- `feature_eng`,
`modeler`, and the grader's two bodies in `rescore.py`. Raising `max_bytes` at the three read sites
would have put a 2.7 MB string into snippet source four times per run on `higgs` and left the cap
that exists to stop exactly that intact everywhere else.

**Alternatives, costed rather than argued.** On a 5,000-row stratified 5-fold split: base64 of a
byte array is 6,668 B, 1.33x WORSE than the 5,000 B digit string, because the digit string is already
one printable byte per row. Run-length as JSON pairs is 41,340 B, 8.3x worse -- mean run length 1.21,
since a shuffled fold assignment is incompressible by construction. `docs/NEXT.md` floated run-length
as an option worth costing; that is the cost. Re-deriving the split from the seed inside each snippet
is smaller than all of these and is refused twice over: the split stops being a recorded object, so a
later contamination objection has nothing to be falsifiable against, and it would make the partition
depend on the installed sklearn version, which `holdout._withhold_rows` already refuses to do for the
withheld carve. Passing `ArtifactMeta.extra["sandbox_path"]` instead -- which `mcp_server/store.py`'s
own `path_of` docstring has advertised since Phase 2, naming this exact problem and this exact number
-- is the closest call. It is rejected as the primary fix because the artifact would stay 2.7 MB and
still truncate: it has to be READABLE through `read_artifact`, not merely usable, since the reviewer
and any future generalist arm reach it no other way. `path_of` remains right for a genuinely large
artifact and its docstring has been corrected, because the 7 MB figure in it is now 79 KB.

**`fold_train: "complement"` is in the file, and it is the part of this design that is not about
size.** "A fold trains on the train rows it does not validate on" is a property of `KFold` and
`StratifiedKFold`, not of a split. It is false of `TimeSeriesSplit`, where fold-train is a prefix.
`TaskSpec` already declares `temporal` and `grouped`, and `profiler.SUPPORTED_SPLIT_STRATEGIES`
refuses them today precisely so a manifest never claims a partition that did not run. Without the
rule recorded IN the file, the day someone implements `temporal` this encoding stays perfectly
writable while the decoder hands back fold-training sets containing future rows, with nothing
raising -- the same contamination the manifest exists to make falsifiable, produced by the artifact
format itself. So the rule is written down, the decoder refuses any other value, and `SPLIT_SNIPPET`
checks the claim against what the splitter actually returned on every single run rather than
assuming it.

**Shrinking the artifact removed a guard, and the replacement had to be deliberate.** Under explicit
index lists, a truncated read was caught by `ArtifactPayload.truncated` and an out-of-range id by
each consumer's `0 <= i < len(df)`. Under an assignment string every position is in range by
construction, so an assignment SHORTER than the frame -- a partial write, or the wrong frame mounted
-- would silently drop the tail from train AND holdout AND every fold at once, and report a
plausible `n_train_rows` while doing it. `decode_split` therefore takes the frame length as a
REQUIRED argument, so no caller can forget to prove it is decoding against the frame it read, and it
checks the manifest's own `counts` against what it actually decoded. `counts` written and never read
would have been decoration.

**One constraint the old form did not impose, found by the fixtures rather than by the datasets.**
There is no character for "a train row that validates in no fold", so every train row must belong to
exactly one fold's valid set. That is what `KFold` and `StratifiedKFold` produce, so nothing real
changes -- but two unit-test fixtures had been written as a single fold that validated on nothing,
and they had to be reshaped to say what they meant. It is a genuine narrowing of what the artifact
can express, and it is the good kind: a split that discards rows is now unrepresentable rather than
silently misrepresented, and the encoder raises rather than writing a manifest with a hole in it.

**One implementation, not four.** `src/ds_agents/split_manifest.py` holds the encoder and decoder as
Python source-string constants, spliced into the snippets as a `{decoder}` / `{encoder}` format
ARGUMENT rather than concatenated into the template. That distinction is load-bearing: the templates
are `str.format` strings with `{{`-escaped braces and this source is not escaped, so concatenating it
the way `rescore._SNIPPET_PRELUDE + _RESCORE_BODY` does would raise at format time, while a
substituted value is not re-scanned. Nothing is `exec`'d in the node process. `tests/
test_split_manifest.py` compiles the constants in the test process, which is what makes them one
implementation under test rather than a mirror -- ruff never looks inside a string literal, so that
module is the only lint this code will ever have -- and one non-`fast` test runs the same source
through the real sandbox so the in-process shortcut is never the only evidence.

**There is no dual-form fallback, deliberately.** No committed artifact needs reading: results files
never contain a manifest. A decoder that also accepted the old explicit lists would have left the
path production takes exercised only by a bespoke unit test, while `tests/nodes/test_feature_eng.py`
and `tests/nodes/test_modeler.py` -- the only tests that put a manifest in front of the real
snippet-formatting code -- kept using the legacy branch. Fixture readability is bought back with
`split_manifest.manifest_from`, a node-process helper for tests only, pinned by test to agree with
the encoder on a real split.

**What is lost, irrecoverably: fold ORDER.** The splitter returns membership in permutation order and
an assignment array can only carry membership. That was paid in its own commit first (`745614a`),
sorting the fold lists while the explicit form was still in place, so this change could be measured
as a numeric no-op rather than confounded with an ordering shift -- the same reason
`forced_drop_release` exists. It was a no-op twice over: the toy run's `cv_scores`, `cv_mean` and
`claimed_holdout_score` are byte-identical across both commits, and `CANDIDATE_SPECS` holds only
`LogisticRegression`, `HistGradientBoosting` and `Ridge`, none of them order-sensitive, while the one
bootstrap estimator in the repo -- the baseline's RandomForest -- fits on `train`, which was already
sorted.

Live proof, `evals/results/2026-09-01_adult-smoke.jsonl`, one run at $0.0158: `rescore_status` ok,
`baseline_status` ok, `refit_claim_gap` exactly 0.0, `baseline_zero_score` exactly 0.5, verified
0.9244 against a claimed 0.9240. `adult` is the first formerly-unrunnable dataset to produce a graded
row, and the first row anywhere in this project with `baseline_normalised_score` above 1.0 (1.054) --
the pipeline beat the raw-column floor, which is the direction the parking lot's "the unit point is a
floor, not a ceiling" note predicts but had not seen.

## 2026-09-01 (third entry): a refusal nothing reads is a comment

`recoverable` was written by thirteen call sites and read by four, none of which was in the graph.
`feature_eng` refused the truncated split correctly, `modeler` would have refused it identically, and
the run carried on through reviewer, router and reporter and wrote a publishable row anyway. The
guard was written, the guard fired, the guard was ignored.

**The spending stops in `graph.py`.** Every straight-line edge is now
`add_conditional_edges(source, halt_or(destination), ...)`, where `halt_or` returns `reporter` when
`PipelineState.halted()`. `graph.py` stays wiring-only -- the predicate lives beside `route_target`
in `nodes/router.py`, for the reason that file's docstring gives.

**To `reporter` and not to `END`**, and `nodes/reporter.py`'s own docstring settles it: it is written
for the hard-failure path where `spec`, `profile`, `chosen_model` and `final_features` are all
`None`, because the harness needs a row for every dataset including the ones that blew up or the
hardest datasets vanish and every published table biases upward. The reporter calls no model, so
halting is nearly free. One consequence taken deliberately: the router never runs, so `review_verdict`
stays `pending` rather than reading `pass`. That incidentally closes half of the standing NEXT.md
item about a run that produced no model being recorded as a pass -- on the halt path only.

**The router's own conditional edges are not wrapped.** The router is unreachable once a fatal error
is on the state, since every edge that could reach it halts first, and a `block` verdict minted with
no `ReviewPass` is a state `route_target` calls unreachable.

**`publishable()` is NOT changed, and that is the decision rather than an omission.** Refusing a
halted row was the obvious move and it is wrong for the same reason the reporter exists: it would
delete exactly the hardest datasets from `evals/results/` and leave the reason in stdout scrollback,
which is the same shape as the defect being fixed. A halted run's cost, node trace, timings and
errors are all real; only its score columns are `None`. `publishable()` means one thing -- no number
from this run is real -- and overloading it would blur that.

What ships instead is the answer the three previous instances got: a companion column beside the bit.
`halted_at` is the node whose unrecoverable refusal ended the run, or `None`. `errored` is one bit
and cannot separate "a column could not be one-hot encoded" from "this dataset cannot be run", and
the live `adult` row above is the worked example -- `errored: true`, `halted_at: null`, because a
high-cardinality column was skipped. Fourth instance of this fix after `leakage_graded`,
`rescore_status` and `baseline_status`, and the standing NEXT.md item asking for it can now be
closed. All 161 committed rows carry zero errors with `recoverable: false`, so the column ships as
`null` on every one of them and no history moves; that is asserted by test rather than claimed.

One property recorded because it is a property of the reducer rather than of the method: `errors` is
`Annotated[list[PipelineError], operator.add]`, so the first fatal error stays on the state for the
rest of the run and a halted run can never un-halt. Correct by the definition of
`recoverable=False`, but nobody should have to infer it.

## 2026-09-02: the last four datasets are priced, and the cost model is tested rather than applied

`bench-mid` fitted `cost ~= a + b * n_features` on four measurements, and the repo immediately began
quoting new runs against it -- `docs/NEXT.md` recorded the one `adult` smoke run at "$0.0158, 13%
high". This session prices the four datasets that became runnable on 2026-09-01 (`adult`,
`bank_marketing`, `numerai28_6`, `higgs`), which is the last blocker on `--subset full`. It also
tests the model it would otherwise have applied, because two things about that model turned out to
be wrong.

**The `adult` residual was never evidence.** The fit is 4 points and 2 parameters: a=$0.010621,
b=$0.00023375 per column, in-sample residuals -$0.00079 / +$0.00028 / +$0.00280 / -$0.00228, and a
residual sd of **$0.0026 on 2 degrees of freedom**. `adult`'s residual is **+$0.0019** -- smaller
than the error the model already makes on the data it was fitted to. A textbook 95% prediction
interval at 14 columns is `[$0.0004, $0.0274]` (t=4.303 on 2 dof, se_pred=$0.00315), a span of
$0.027 -- wider than any cost this project has ever measured.
The model is under-determined, not refuted, and "13% high" was reading the fit's own noise as a
finding. The pre-registered band below is therefore **+/-$0.0026, the fit's own residual sd**,
stated as a rule rather than as an interval nobody could act on.

**All four points it was fitted on have zero categorical columns, and nobody chose that.**
`phoneme`, `jasmine`, `amazon_employee_access` and `nomao` are entirely numeric, so `feature_eng`
one-hot encodes nothing on any of them and `LEVELS` in the emitted transform is empty in all four.
That fell out of picking a 2x2 on rows x columns, because dtype was not one of the two axes anyone
was thinking about. Both out-of-sample points measured since are categorical and both ran ABOVE the
model: `credit_g` (13 categorical, 54 one-hot levels) +$0.0015 and `adult` (7 categorical plus 1
skipped at `MAX_ONE_HOT_LEVELS`, 58 levels) +$0.0019. `credit_g` has 1000 rows, so that residual
cannot be a row term. The mechanism is not speculative: the reviewer reads `feature_code_artifact`
into its prompt, and that artifact's `LEVELS`, `FEATURE_ORDER` and `COLUMN_SOURCE` grow with one-hot
level count and not with rows.

So the arm has a primary deliverable and a secondary one, and they are independent. The primary is
**four measured per-cell means**, so `full` needs no extrapolation on these four whatever the model
does. The secondary is a test of the model, and it is worth running because rows and categoricals
are **anti-correlated** across exactly these four datasets -- `adult` (48,842 rows, 7 categorical)
and `bank_marketing` (45,211, 9) against `numerai28_6` (96,320, 0) and `higgs` (98,050, 0). The two
hypotheses predict opposite orderings of the residual. That is luck rather than design, and it is
worth recording as luck: had the four been correlated, the same $0.24 would have settled nothing.

**Two risks NEXT.md carried are closed for $0 before any money moved, and one of them was stale.**

- `MODEL_TIMEOUT_S = 240` was flagged as the `higgs` risk on the grounds that it was already 45%
  consumed at 144 columns on `jasmine`. It does not extrapolate, because `permutation_importance`
  issues `n_repeats x n_columns x n_candidates` scoring CALLS and each pays a fixed cost rebuilding
  a frame through the in-pipeline transform. `jasmine` is 2,880 such calls; `higgs`, at 28 columns,
  is 560. Measured two ways: the offline `higgs` run's modeler node took **24.8s** and a direct fit
  of the binary `CANDIDATE_SPECS` at the real shapes (62,752 train / 15,688 holdout / 28 columns)
  took **12.6s**, 5.3% of the budget. The binding term is call count, not rows -- which is
  `bench-mid`'s "both track columns" finding applied to wall clock. **No timeout is raised.**
- `register_dataset` was carried as "Never measured. This is the session to measure it." That was
  stale: DECISIONS recorded it on 2026-09-01 for the nine datasets that could be run. What was
  missing was these four, and the fact that the record was prose in a commit that touched no code.
  Both are fixed by `tests/test_register_dataset_cost.py`, which prints a re-derivable table across
  all thirteen shapes. `higgs` is the worst at **0.25s per call, 0.75s per run, 1.2% of budget**;
  `nomao` reproduces the recorded 0.17s exactly.

### Why the register_dataset measurement is a test and not a results column

It measures harness *setup*, not run behaviour, and it is constant across the replicates of a cell,
so sixteen rows would carry four distinct numbers four times each. Adding a column also makes all
~150 committed rows carry it as `null` forever, a price this repo has paid four times deliberately.

The counter-argument is real and is recorded rather than waved away: `wall_seconds` still excludes
it, so a reader of the results file cannot reconstruct a run's true wall cost -- which is the exact
defect `rescore_seconds` and `baseline_seconds` were added to fix. The distinction being drawn is
that those measure the GRADER, which is the term `--subset full` was blocked on pricing, and this is
about one second of setup. If it ever stops being about one second, the column argument comes back.

**The finding the test records, which is not the timing.** There is no timeout on `register_dataset`
itself. Under the default `--tools mcp` it runs inside the server subprocess before the MCP
`initialize` handshake returns, so the only thing bounding it is `MCPTools._CONNECT_TIMEOUT_S = 60`, and
under `--tools local` not even that. **But that is only true of ONE of the three calls, and the
other two are worse.** `rescore.rescore` and `rescore.baseline` each construct `LocalTools`
directly and never pass through `cli._select_tools`, so their registrations are unbounded under
BOTH transports, always. A breach on the graph's call raises `SystemExit`, which `harness.run_eval`
re-raises rather than counting, so it aborts a whole invocation rather than one run and
`MAX_CONSECUTIVE_FAILURES` never sees it; a hang on either grader call does not raise at all and
simply blocks forever, invisible to every column. So `--tools` was never the axis -- the guard that
exists governs the one call that has one, and the grader's two have none. Recorded, not fixed: at
1.2% of budget at the largest shape in the manifest, the exposure is small and the guard is worth
designing deliberately rather than in a pricing session. Its absence is now an open question in
NEXT.md rather than an unstated assumption. The
test's headroom is 10x rather than `test_baseline_cost.py`'s 5x for the related reason that the 60s
budget is not registration's alone -- it also covers interpreter boot, importing `mcp_server`, and
the handshake itself.

`CALLS_PER_RUN = 3` stops being a sentence at the same time. `test_a_benchmark_run_registers_the_
dataset_exactly_three_times` spies on the store across a full graph-plus-rescore-plus-baseline run,
so the multiplier on every number in that table is now a property. Three is correct rather than
wasteful: `rescore` and `baseline` each build their own `LocalTools` precisely so a grader failure
cannot take the run's store with it.

### Pre-registered endpoints

Invocation: `--subset bench-tall --name bench-tall --replicates 2 --n 2 --max-cost-usd 0.40`,
16 runs, $0.2440 estimated. n=4 per cell rather than `bench-mid`'s 2 because the endpoint is a cell
MEAN compared against a prediction at the $0.002 scale, and `nomao`'s two committed runs were
$0.0378 and $0.0439 -- a within-cell spread larger than the discrepancy being tested. The cap at
1.6x the estimate is deliberate: here the estimate's job is to be the prediction, and the CAP is
what protects the budget.

The four `est_cost_usd` values shipped in `SUBSETS["bench-tall"]` ARE the point predictions --
$0.014, $0.014, $0.016, $0.017, from `a + b * n_features` at 14, 16, 21 and 28 columns. The commit
that replaces them with measured means is the permanent record of the miss.

| cell | cols | categorical | prediction | consistent band (+/-$0.0026) |
|---|---|---|---|---|
| adult-categorical-tall | 14 | 7 (58 levels) | $0.0139 | $0.0113 - $0.0165 |
| bank-categorical-tall | 16 | 9 (44 levels) | $0.0144 | $0.0117 - $0.0170 |
| numerai-numeric-tall | 21 | 0 | $0.0155 | $0.0129 - $0.0181 |
| higgs-numeric-tall | 28 | 0 | $0.0172 | $0.0145 - $0.0198 |

1. **Feasibility, primary.** All 16 runs produce a publishable row with `errored: false` and
   `halted_at: null`. Any non-null `halted_at`, or a `rescore_status` of `no_split` or
   `no_feature_code`, means the offline pass missed something; response is to STOP the invocation
   rather than fund the second replicate. Offline runs at this commit put all four at
   `halted_at: null`, `rescore_status: ok`, `baseline_status: ok`.
2. **Grader correctness, asserted with no tolerance.** `rescore_status` and `baseline_status` `ok`
   16/16; `baseline_zero_score` **exactly 0.5** 16/16; `baseline_recipe` `rf-v1` 16/16;
   `refit_claim_gap` **exactly 0.0** 16/16. These are correctness assertions on positive class,
   scorer sign and row selection at once, and they have held on five datasets. Extending them to
   nine is most of what this run buys beyond the price.
3. **Four measured per-cell means**, replacing four predictions. A cell whose measured mean exceeds
   its estimate by more than 2x is reported as an estimate failure, the error `bench-smoke` made.
4. **The model check, stated so it can fail.** `residual = measured mean - point prediction`:
   - *Column-only model survives:* all four residuals inside +/-$0.0026 with no monotone trend in
     rows. `full` may then be priced from the model, though every cell still ships its own mean.
   - *H_rows:* `numerai28_6` and `higgs` above band, `adult` and `bank_marketing` inside. This
     REFUTES the column-only model and `full` must be re-priced with a row term.
   - *H_categorical:* `adult` and `bank_marketing` above band, the numeric two inside or below. The
     2.9x row extrapolation survives, but the fitted set was confounded -- and `kr_vs_kp` (36
     categorical columns, 73 one-hot levels) becomes the cell in `full` the model can least price.
   - *Neither:* residuals scattered with no pattern. The fit was never determined enough to test,
     and that is the report.
   H_categorical is pre-registered as the hypothesis with prior support (`credit_g` +$0.0015 at
   1000 rows, `adult` +$0.0019), so a result confirming it is confirmatory at n=4 cells and nothing
   more.
5. **Timing, each against an offline number already on record.** `node_seconds["modeler"] < 240`
   16/16 and specifically `higgs` **under 40s**; `node_seconds["profiler"] < 60` 16/16 (offline max
   was 7.5s on `higgs`); `baseline_seconds` on `higgs` under the 57.18s synthetic bound in
   `tests/test_baseline_cost.py`, reported beside it -- the offline run came in at 37.8s, so this is
   the first evidence of how loose that deliberately-worst-case bound is on real data.
6. **Determinism, which closes a standing question for free.** NEXT.md wants one cheap run on a
   second manifest dataset before anyone writes "the pipeline is deterministic on real data". Four
   runs in each of four cells answers it on four more. Pre-register the asymmetry: `adult` and
   `bank_marketing` have real `feature_eng` decisions available (`adult` already skips
   `native-country` at `MAX_ONE_HOT_LEVELS = 20`), so variation is EXPECTED there;
   `numerai28_6` and `higgs` are fully numeric with nothing to decide, so identical rows there
   merely reproduce `credit_g` and mean nothing stronger. Directional at n=4.
7. **`bank_marketing`'s `V12` stops being hypothetical.** It is the manifest's only non-empty
   `known_leakage`, documented and deliberately unscored, and this is the first run that executes
   the dataset. Recorded as an observation rather than an endpoint: what `rescore_status`,
   `n_withheld_rows` and `baseline_normalised_score` do, and whether anything downstream noticed.
8. **Not endpoints.** The absolute value of `verified_holdout_score`, `holdout_claim_gap` or
   `baseline_normalised_score` on any cell. n=4, one commit, the grid is ours, the unit point is a
   floor. Characterisation only. **No `eval-diff` comparison** -- four new `dataset_id` values at a
   new commit, so there is nothing to compare and running it would be theatre.

### Failure endpoints

- **A modeler timeout anywhere.** It would mean both offline measurements were wrong by an order of
  magnitude. Note it CANNOT surface as `halted_at`: `modeler.py` reports it through
  `run.failure(...)`, and `NodeRun.failure` defaults `recoverable=True`, so `halt_or` never fires.
  What appears instead is `errored: true`, `halted_at: null` and no `chosen_model` -- the shape the
  `recoverable` fix exists to make visible, arriving through the one door that fix does not cover.
  This is the first known gap in `halted_at`'s coverage and it is recorded whether or not it fires.
  Response if it does fire: raise `MODEL_TIMEOUT_S` with the measurement as the justification, on
  the same grounds this file already allows for `PROFILE_TIMEOUT_S` -- no recipe moves, it feeds no
  number on a results row, and a run that finishes at 240s finishes identically at 480s. Do NOT drop
  `N_PERMUTATION_REPEATS`, for the reason recorded on 2026-09-01: it feeds `top_importances`,
  `reviewer_nominated` and `leakage_caught`, so it needs its own axis and its own version string.
- **`baseline_status: unit_point_failed` on `higgs` or `numerai28_6`** would mean the 5x headroom in
  `test_baseline_cost.py` was measured against the wrong thing, and `BASELINE_TIMEOUT_S` and
  `n_estimators` come back on the table with `baseline_recipe` as the lever.
- **A `SystemExit` out of `_select_tools`** aborts the whole invocation rather than one run. Measured
  to 1.2% of its only bound in advance; see above.
- **What will NOT fire.** `MAX_CONSECUTIVE_FAILURES = 3` counts only runs that RAISE. A node timeout
  does not raise, and per the first bullet a modeler timeout does not even halt.

### Outcome, appended after the runs

`evals/results/2026-09-02_bench-tall.jsonl`, 16 rows, one commit (`558548e`), **$0.3021** against a
$0.2440 estimate and a $0.40 cap. 0 refused, 0 failed. Full numbers in `evals/results/LOG.md`.

**The four measured means are the deliverable and they are in.** $0.0159 / $0.0178 / $0.0172 /
$0.0246 against predictions of $0.0139 / $0.0144 / $0.0155 / $0.0172. `SUBSETS["bench-tall"]` now
carries them rounded up, and `SUBSETS["full"]` exists.

**The model check returned the fourth pre-registered option: neither.** H_categorical predicted
`adult` and `bank_marketing` above the band; one was. H_rows predicted `numerai28_6` and `higgs`
above it; one was. And `bank_marketing` (45k rows) ran dearer than `numerai28_6` (96k), which no row
term orders. What the four have in common is only that **all four residuals are positive**, as is
`credit_g`'s -- 5 of 5 out-of-sample datasets under-predicted, sign test p=0.031.

Refitting on all nine measured datasets settles the secondary question against the hypothesis this
entry pre-registered as the one with prior support. A row term cuts the residual sd from $0.00296 to
$0.00224; a categorical-count term makes it **worse** ($0.00320) and adds nothing once rows are in
($0.00206 for both). **H_categorical is refuted as a cost term.** The mechanism was real -- the
reviewer does read a transform artifact whose `LEVELS` grow with one-hot count -- and it is not
worth a parameter at 7 to 13 categorical columns. That is the useful shape of the result: a
mechanism can be true and still not be the term you are missing. `kr_vs_kp` at 36 fully categorical
columns is not covered by that refutation and is flagged in `full`'s comment as the softest price.

The retained model, and what `full` prices unrun datasets from:
`cost ~= $0.010163 + $0.000230 * n_features + $0.005609 * (n_rows / 1e5)`, residual sd $0.0022 on 9
datasets and 3 parameters. It is offered as a planning number, not a finding: the row term's
improvement rests heavily on `higgs`, and `est_cost_usd` never reaches a results row.

### Two defects the arm found that it was not looking for

**`errored` is still not a companion column, and the 2026-09-01 entry claiming otherwise was too
strong.** All four `adult` rows carry `errored: true` with `halted_at: null`, no objection, verdict
`pass`, and a `verified_holdout_score` of 0.9244 that nothing objected to. The cause is that `feature_eng` records
"columns skipped as too high-cardinality to one-hot encode at 20 levels: ['native-country']" as a
`PipelineError` with `recoverable=True`. That is an informational note about a routine decision, not
an error, and `errored` is `bool(self.errors)`. `halted_at` fixed the fatal-versus-non-fatal
distinction, which was the fix it was designed for; it did nothing about
recoverable-and-not-a-problem. So a table using `errored` as a rate reports `adult` as a 100%
failure cell today. The fix is not another column -- it is that a routine decision should not be a
`PipelineError` at all -- and that is a node change, so it is recorded here and reopened in NEXT.md
rather than done in a pricing session. Fifth instance of the same family, and the first one where
the previous fix was announced as closing it.

**`baseline_normalised_score` divides by a quantity that can approach zero.** `numerai28_6` returns
**2.089**, the largest value in the project. Zero is 0.5, the unit point reaches 0.5101, the
denominator is 0.0101. The arithmetic is exactly AMLB's `(x - zero) / (unit - zero)` and it is
working correctly; the dataset is near-chance (published reference 0.530) and a normalisation whose
denominator is the yardstick's own span becomes unbounded as that span vanishes. Three of four cells
are now above 1.0. The standing parking-lot note framed this as "the unit point is a floor, not a
ceiling", which is about values slightly above 1; that framing does not cover 2.089. Nothing warns,
and `baseline_status` is `ok` because nothing went wrong. This wants a guard -- a status value, or a
`None` below some minimum span, the way `baseline_normalised_score` already returns `None` on a
planted leak -- and choosing which is a design decision, so it is reopened rather than patched.

### What `--subset full` waits on now, which is nothing this repo can fix

`SUBSETS["full"]` ships with 13 cells: 9 measured means, 4 modelled from the refit plus one residual
sd and rounded up (`australian`, `kc1`, `sylvine`, `kr_vs_kp` -- the four smallest shapes). The
bespoke `ValueError` in `_resolve_subset` is deleted, and with it the pattern of rewriting that
message each time a blocker moved; it had been corrected four times.

The blocker does not move a fifth time. Running `full` properly is 13 cells at `--replicates 2 --n
2` -- 52 runs, about **$1.10 and 45 to 70 minutes**. Nothing technical stands in the way. Whether to
spend it is a decision for a person, it has never been put to anyone, and that is the honest content
of what remains. Recorded in the entry's own comment, where someone about to spend the money will
be reading, rather than in an error message they will never see.

One caveat inherited whole: all 16 runs raised zero objections and took the review loop exactly once.
No manifest dataset has ever been observed looping, and a run that loops three times costs about
2.5x, so `full`'s $1.10 is the price of 52 runs that all pass first time.
