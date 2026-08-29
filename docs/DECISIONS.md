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
