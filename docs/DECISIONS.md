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
