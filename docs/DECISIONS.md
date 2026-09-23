# Decisions

One-line summaries, newest at the bottom. Full text of every decision, including the alternative
considered and the full reasoning and any pre-registrations, is in docs/decisions-archive.md. Run
results and per-run narratives are in evals/results/LOG.md.

## 2026-08: Reviewer is adversarial, not advisory
The reviewer can block and route back instead of only annotating the report. Advisory review never
changes outcomes and can't be measured; blocking creates the loop-cap and "exhausted" outcomes that
are worth evaluating.

## 2026-08: Tools are an MCP server, not in-process functions
Costs a phase of plumbing, but keeps the single-agent-vs-team ablation honest by giving both arms
the same tool surface, and gives the project a real MCP server to point at.

## 2026-08: Tabular only, small model set
The project measures the system, not the model: linear baseline, XGBoost, optionally a small
PyTorch net. Anything more is scope creep that would not change the findings.

## 2026-08: Haiku by default
Every node runs on Haiku unless an ablation says otherwise, to keep per-run cost low enough to
repeat the benchmark and to make the Sonnet-reviewer ablation a clean comparison.

## 2026-08-22: The system under test does not report its own grade
The modeler writes `claimed_holdout_score`; the harness writes `verified_holdout_score`, scored
independently on a holdout the agents never see. Trusting the modeler's own number would mean every
headline figure rests on a system grading itself.

## 2026-08-22: Anything the eval counts is a field, never prose
`Objection` carries a required `columns: list[str]` instead of scoring leakage by searching the
reviewer's free text for a column name, which would tie the metric to an LLM's phrasing and could
not express a partial catch.

## 2026-08-22: The router owns the loop counter and the verdict
`review_iterations` and `review_verdict` are computed by the router, not the reviewer, so a model
that claims "pass" at the cap cannot be confused with one that passed on merit.

## 2026-08-22: Run conditions are frozen onto the state object
`RunConfig` (run id, arm, reviewer settings, loop cap, seed) is immutable after construction, so a
results row is self-describing and no node can quietly raise its own loop cap.

## 2026-08-22: History fields are append-only via reducers
`objections`, `review_passes`, `errors`, and `node_trace` use `operator.add` reducers so LangGraph
does not overwrite history on a partial node update, and derived numbers are computed fields so
they actually appear in `model_dump()`.

## 2026-08-22: The toy leak is noisy and ships with a ground-truth manifest
The toy fixture's planted leak disagrees with the target on about 9% of rows rather than being a
perfect copy, so it can discriminate a good reviewer from a bad one; the manifest records the
measured rate so tests never grade against a nominal one.

## 2026-08-26: Nodes receive their tools and model as arguments, bound in graph.py
A node is `node(state, *, tools, model) -> dict`, bound by `graph.py` via `functools.partial`,
instead of a module-level client, so tests can inject fakes and the Haiku/Sonnet reviewer swap is a
config change rather than a code change.

## 2026-08-26: Intake sees the dataset through the artifact store, not run_python
The dataset is registered as an artifact with schema metadata at run start, so intake does not need
`run_python`, which would let the first node in the graph execute arbitrary code before anything is
profiled.

## 2026-08-26: The profiler computes statistics in code and asks the model only for judgement
A fixed snippet computes dtypes, missing fractions, cardinality, and mutual information; the model
only judges which columns look like leaks. A model adds nothing to a statistic it could just
compute.

## 2026-08-26: The Phase 1 tool shim runs subprocesses and strips the environment
`tools/local.py` runs snippets in a subprocess (never `exec`) with a stripped-down environment, so
agent code cannot reach `PipelineState` (including the leakage answer key) or read secrets or the
repo it is being graded in.

## 2026-08-26: `StubModel` is a placeholder and every event it touches says so
With no API key, the graph runs against a deliberately weak stub that nominates zero leaks, and
every event and the CLI output flag it as "stub" so a stub run is never mistaken for a real result.

## 2026-08-26: The split the agents see is not the holdout the harness scores on
`split_artifact`, the agents' own split, is a different object from the independently withheld
holdout behind `verified_holdout_score`, which preserves the claimed-vs-verified gap as the
project's central finding.

## 2026-08-26: `leakage_flagged` counts what was ever raised; false alarms also get a standing count
Added `*_standing` variants that only count objections still open at the end, alongside the
"ever raised" versions, so withdrawing a false alarm is credited. The same review also fixed three
related scoring bugs: an empty feature matrix counting as remediation, a `score_ratio` divide-by-
zero, and an unsigned `holdout_claim_gap` that let overclaims cancel out.

## 2026-08-26: An artifact is immutable once issued, and a mislabelled split is worse than none
`LocalTools` now detects a snippet overwriting an already-issued artifact by content change and
reissues a new id, rather than silently changing the bytes behind an existing one. The profiler
also now refuses unimplemented `temporal`/`grouped` splits instead of silently falling back to a
random one.

## 2026-08-26: The model client calls the Anthropic SDK directly, not through LangChain.
LangSmith tracing comes for free anyway through langgraph's own transitive dependencies. What
actually decided it was constrained decoding via `messages.parse` and getting raw, unnormalized
token usage back, since cost numbers are a published output of this project.

## 2026-08-26: An unpriced model raises rather than costing zero.
`tools/pricing.py` refuses to price a model it does not know, instead of returning 0.0, so an
unpriceable model fails before a benchmark run spends money rather than producing rows with a
silently, flatteringly low cost.

## 2026-08-26: The router blocks on any open objection; `severity` is recorded and unread.
Blocking only on "high" severity would fold the reviewer's calibration into its detection score.
`severity` is written but never read today, kept for a future ablation that separates the two.

## 2026-08-26: The loop cap is enforced by the router and the edge together, not by a clamp.
The router increments and checks `review_iterations` before the edge reads the derived verdict.
Deliberately no defensive clamp, so a graph-wiring bug shows up as a visibly wrong iteration count
rather than being silently absorbed.

## 2026-08-26: `shap_artifact` is now `importance_artifact`.
It always held sklearn permutation importances, not SHAP values, and `shap` is not even a
dependency. A field name that lies to future readers is the same defect as a mislabelled split.

## 2026-08-26: One vocabulary for column names, and it is the source names.
`final_features` and related fields hold original column names, never one-hot expansions, because
expanded names would silently break the intersection check that computes `leakage_remediated`.

## 2026-08-26: feature_eng emits a fitted transform as code, not a transformed table.
`feature_code_artifact` is executable code (constants plus `transform()`), not a matrix, so the
harness can re-apply the exact same transform to the withheld holdout in Phase 4 without a second
transform path that could quietly diverge from the first.

## 2026-08-26: Imputation medians and one-hot levels are fitted on train rows only.
Fitting constants only on the pinned train split, never the full frame, avoids leaking the agents'
own holdout into the training statistics, which is exactly the contamination the reviewer exists to
catch.

## 2026-08-26: The reporter and router call no model.
Both are deterministic. The report is not scored by the eval, so a model-written narrative is pure
token cost, and letting a model run the router would mean the system under test decides when to
stop reviewing itself.

## 2026-08-26: feature_eng picks columns; it does not write code.
The model returns a drop list with justifications, and a fixed template does the actual transform,
because the sandbox is not yet isolated enough for arbitrary model-authored code and a drop list is
checkable against the profile.

## 2026-08-26: Forced drops make remediation partly mechanical, and that is fine.
`feature_eng` force-drops the target, id columns, and any objected column before consulting the
model, which makes `leakage_remediated` partly mechanical inside a looping run. Recorded so a
future reader does not mistake the correlation for a bug.

## 2026-08-26: sklearn's `neg_*` scorer convention stops at the snippet boundary.
Fixed a bug where sklearn's negated lower-is-better scores leaked out of the snippet, corrupting
candidate selection, the modeler's prompt, and the sign of `claimed_holdout_score`. Scores now
always leave the snippet in the metric's natural units.

## 2026-08-26: the run config records the model that actually ran.
Fixed a bug where `--model sonnet` built the client straight from argv while `RunConfig` kept its
defaults, so a Sonnet run would have been published labelled "haiku".

## 2026-08-26: the graph's recursion limit scales with the loop cap.
LangGraph's default recursion limit could be exceeded by a high `loop_cap`, killing the run with no
results row at all, which biases every published table upward by silently dropping the runs that
looped the most.

## 2026-08-27: the sandbox forks from one warm worker, and Docker is not the Phase 2 backend.
Chose a long-lived worker process that forks per snippet over Docker-per-run or exec-in-a-shared-
namespace: imports dominate cost, and forking gives a genuinely fresh process per snippet where
exec would let state leak between them. Docker is deferred, not rejected, since nothing yet needs a
memory cap or network block. Also closed a hole where the repo itself was importable from inside a
snippet via `sys.path`.

## 2026-08-27: the MCP server is a skin over the in-process tools, not a second implementation.
`mcp_server/server.py` wraps the same `LocalTools` the graph already used rather than
reimplementing against the store and sandbox directly, so "both arms get the same tool surface" is
a fact about the code and not just a claim about it.

## 2026-08-27: one server process per run, because a fifth tool would change what the ablation holds constant.
Rejected a long-lived server keyed by run id, which would need a fifth tool or a `run_id` argument
on every call, in favor of one process per run, keeping the four-tool surface identical for both
arms.

## 2026-08-27: a read with no `max_bytes` is still capped, and the cap lives in the store.
`ArtifactStore.read` applies a default 1 MiB cap even with no explicit limit, so `--tools local` and
`--tools mcp` cannot disagree about truncation. Surfaced a real gap: `modeler` was not guarding a
truncated split-manifest read the way `feature_eng` already did.

## 2026-08-27: a sandbox failure keeps its type across the wire.
The MCP server distinguishes an agent's tool error from our own sandbox error across the transport,
so a broken worker process is never filed as an agent mistake in the results.

## 2026-08-27: the generalist baseline is verified by use, not by an approval flag.
Confirmed Claude Code itself, not just a hand-rolled client, can drive the four MCP tools directly
against the toy dataset, proving the sandbox invariants hold for a client this repo did not write.

## 2026-08-27: the router mints the whole ReviewPass; the reviewer hands dispositions across on one narrow field.
Avoids two nodes each appending a partial `ReviewPass`, which would double-count under the
append-only reducer. The router computes `routed_to` itself rather than letting a downstream edge
re-derive it, since state changes between invocation and merge.

## 2026-08-27: a malformed objection is filtered one at a time, never by the response schema.
Switched from a pydantic validator, which failed the entire response on one bad objection, to
per-item filtering in the node, after a live Haiku response with one empty-columns objection lost
an otherwise well-formed review pass.

## 2026-08-27: trap difficulty is set by the column name, not by the association number.
Tuning a trap column's statistical association with the target made no difference to detection.
Renaming it from a descriptive name to an opaque one did. The profiler is mostly reading column
names, not statistics.

## 2026-08-27: the reviewer misses a top-ranked leak that reaches it, in 3 of 3 runs.
In the few runs where a trap survived into the matrix, the reviewer never raised a leakage or
contamination objection about it at all. It audits the numbers it is shown but does not interrogate
what the columns mean.

## 2026-08-27: name transparency becomes a run condition, applied at load time.
Since column naming turned out to drive detection, it is now a recorded run condition: a module
rewrites headers on all non-target columns above the tools boundary, rather than being an unrecorded
property of whichever fixture a session happened to write.

## 2026-08-27: results rows start being written, and learn to see the profiler.
Added `--repeat`/`--results` to `ds-agents run` so ablation runs leave a traceable row. Also fixed
`results_row()` to score leakage from reviewer objections only, missing that the profiler's own
nomination and recall is the actual dependent variable in a naming ablation.

## 2026-08-27: the name effect is real, it is large, and it decides whether the reviewer is tested.
Twenty live runs, ten a side: the profiler nominates the planted trap in 10/10 descriptive runs
against 2/10 opaque; the trap survives into the model matrix in 2/10 descriptive against 10/10
opaque. Descriptively named fixtures mostly do not test the reviewer, since upstream cleans the
matrix first.

## 2026-08-27: the reviewer misses the leak in 12 of 12 runs where it had one, and looks in the wrong place while doing it.
Across all 20 runs, a trap survived into the matrix 12 times and the reviewer never once raised
leakage or contamination naming it. Even when it correctly flagged a score as implausible, it named
near-zero-importance columns instead of the top-ranked ones actually responsible.

## 2026-08-28: the reviewer arm is a 2x2, because the prompt was a confound under any model number, and the prompt turns out to be the whole effect.
Ran Haiku-vs-Sonnet crossed with a prompt-rule condition (name the column explaining an implausible
score), because the missing rule was itself a confound. The prompt turned out to be the binding
constraint: with the added rule, Haiku goes from 1/10 to 9/10; Sonnet on the base prompt is still
0/4.

## 2026-08-28 (second entry): the loop cap was never the bottleneck; an objection needs a route it can act through and a condition that closes it.
A loop-cap sweep barely moved remediation, disproving the standing hypothesis. The real breaks:
objections about an implausible score route to the modeler, which has no column-removal lever, and
the reviewer almost never marks an objection resolved. Both are instrumented rather than fixed yet.

## 2026-08-28 (third entry): who acts on an objection becomes a recorded condition, and the reviewer's own choice stays on the record.
Added `objection_routing` as a run condition: under `by_category`, column-scoped objections
effectively route to `feature_eng` regardless of what the reviewer wrote, but the reviewer's own
`target_node` is never rewritten, so the override stays measurable rather than hidden.

## 2026-08-28 (fourth entry): a resolved objection keeps its column out, and the closure arm's premise gets checked before it gets funded.
Fixed a bug where a resolved (not withdrawn) column objection let `feature_eng` re-admit the column
on a later pass, by splitting "still complained about" from "must stay out, released only on
withdrawn." Also found the "reviewer never resolves" diagnosis did not hold at n=10, so the planned
closure-prompt arm was never funded; its own control cell already resolved honestly.

## 2026-08-28 (fifth entry): the release rule becomes a recorded condition, and the fourth entry of the same day is overturned.
Reversed the earlier "ship the sticky-drop fix unconditionally" call: added a release-rule axis
defaulting to the fixed behavior, because the buggy behavior had become the denominator of the
project's largest published number. With a same-commit control, the previously reported 5/10 to
9/10 effect did not replicate; most of it was model nondeterminism.

## 2026-08-29: a `block` with nothing to act on gets asked again, once, and the router's question moves onto the state where both callers can share it.
Fixed the "zero-objection block" bug, where the reviewer blocks on nothing actionable and the run
reaches the reporter having done nothing, by sharing one predicate between reviewer and router and
retrying the reviewer once with an added hint when the answer is empty.

## 2026-08-29 (second entry): the row says what went wrong and which tree produced it, and the two new labels sit on opposite sides of the state boundary.
Added a structured `errors` list and a `commit` field (on the frozen `RunConfig`) to results rows,
since prior failures were indistinguishable from each other and no row could be traced to the code
that produced it. Per-invocation sampling labels (cell, replicate) stay separate, applied at write
time, since a node must never know its own position in the sampling design.

## 2026-08-29 (third entry): the harness is replicate-aware before it is anything else, and the refusal to call a difference an effect lives in the tool rather than in a reader's discipline.
Built the Phase 4 harness replicate-major, not cell-major, so a binding cost cap truncates every
cell proportionally instead of starving the last one. `evaldiff.py` refuses to call an underpowered
or overlapping-interval difference an "effect," enforced in the tool rather than left to a reader.

## 2026-08-31: the `ci` subset is pre-registered as a baseline rather than a hypothesis, and what a zero-count on the block-retry is allowed to mean is fixed before the run rather than after it.
Pre-registered the first live `ci`-subset invocation, and what a zero versus nonzero block-retry
count would mean, before running it, since this is a baseline with no contrast rather than an
ablation. The retry fired 7 of 30 times, more than expected, and provenance was found to be
recorded per-run instead of per-invocation, which had silently split a cell.

## 2026-08-31: the benchmark manifest is generated, and a published number is not a baseline
Built the manifest from the AMLB benchmark suite (13 datasets), generated only by a refresh command
that `.claude/settings.json` protects from hand edits, so no person can type a plausible-looking
ground-truth number into it. Keeps a third-party published reference score separate from this
repo's own baseline definition, since conflating the two protocols would misattribute the
difference.

## 2026-08-31 (third entry): the re-scorer is pre-registered, and the withheld holdout is carved before the graph starts because no split a node decided can grade the node that decided it
Pre-registered the re-scoring protocol before writing it: a 20% stratified holdout carved before
the graph starts, refit using the recorded model spec, scored in its own sandbox, with a self-check
against the agents' own claimed score. The refit gap came back exactly 0.0 on every row, and the
smoke-test rows reproduced deterministically since that dataset had no decision to make differently.

## 2026-08-31 (fourth entry): the baseline is two points and not one number, and `score_ratio` is retired because a single column cannot say which end of a scale it is
Retired `score_ratio`/`baseline_score` in favor of two raw reference points plus a computed
normalized score, gated to null whenever a planted leak is present, since the baseline (fit on raw
columns including any leak) and a leak-free pipeline would otherwise mean opposite things on the
same scale.

## 2026-09-01: four of the thirteen datasets cannot be run, and the column that would have priced the rest could not see the thing it was pricing
Found `wall_seconds` never included grader time, making cross-file cost comparisons misleading. Also
found four manifest datasets cannot complete a run at all because their split manifest exceeds the
artifact read cap, a silent failure that looked like a valid result rather than an error. Redesigned
the next pricing subset as a 2x2 on rows x columns instead of a single line, since the two had been
conflated.

## 2026-09-01 (second entry): the split manifest is one character per row, and the complement rule stops being an assumption
Replaced the split artifact's row-index-list encoding with a one-character-per-row assignment
string, shrinking the largest manifest by 97% and unblocking all four previously unrunnable
datasets for free. Also explicitly encoded the "fold trains on what it doesn't validate on" rule so
a future non-KFold split strategy cannot silently misrepresent itself.

## 2026-09-01 (third entry): a refusal nothing reads is a comment
Fixed a bug where a node correctly refused a truncated artifact but nothing in the graph actually
stopped the run, so an unrunnable dataset still produced a full publishable row. Added routing to
the reporter on any unrecoverable error, plus a `halted_at` column so the failure is visible instead
of looking like a bad but real result.

## 2026-09-02: the last four datasets are priced, and the cost model is tested rather than applied
Priced the remaining four datasets and tested, rather than trusted, the earlier cost model, since
its apparent 13% miss on one dataset was within the model's own noise. Found the originally fitted
set was accidentally all-numeric; the new categorical datasets ran above the model, a real mechanism
later refuted as a cost term. Also closed two open cost risks for $0 by measuring offline first.

## 2026-09-02 (second entry): a routine decision stops being an error, the baseline scale publishes its own length, and a metric may ask whether a column is null
Fixed three column-semantics defects: a routine high-cardinality column skip was wrongly recorded
as an error, making every affected row read as failed; added a `baseline_separation` column so an
extreme normalized score can be explained by a narrow reference scale instead of looking anomalous;
and gave metrics a null/not-null predicate so always-null columns can be tallied at all.

## 2026-09-02 (third entry): a fit failure gets a companion column, not a reclassification, and the two repairs to "`errored` is uninformative" are opposite
Unlike the cardinality-skip fix, a candidate model that fails to fit stays a real error, since it is
a genuine anomaly, but gains companion count columns so the rate is calculable instead of buried in
free text. Also fixed a report-truncation bug that could break markdown tables on multi-line
exception messages.

## 2026-09-02 (fourth entry): `--subset full` is a coverage run, and the thing being pre-registered is what would make it uninterpretable rather than what would make it interesting
Unlike prior ablation arms, the full 13-dataset run has no hypothesis or comparand, so what is
pre-registered is the conditions under which the resulting file must not be quoted. The run's cost
overran by about 10%, entirely explained by review-loop retries on two datasets rather than dataset
size, confirming the cost model was fine and the missing factor was loop rate.

## 2026-09-09: the walkthrough replays the final state; it does not checkpoint the run
The interactive pipeline walkthrough derives its step-by-step view from the final `PipelineState`,
using each field's single-writer contract, rather than adding a LangGraph checkpointer. Trades some
fidelity, labelled recorded, reconstructed, or not recorded, for not adding a persistence layer to
the system under test.

## 2026-09-21: the writeup's leakage numbers cannot be attributed to a commit, and the draft says so rather than quietly not mentioning it
The 145 fixture rows behind every leakage finding predate the `commit` column and several other
current results columns, so they cannot be tied to a specific tree. Chose to state the limitation in
the README rather than re-run everything or silently omit the caveat.

## 2026-09-21 (second entry): the cost model does not get a loop-rate term, because six events across three datasets is a record of what happened and not a rate
Declined to add a loop-rate term to the cost estimator after the full run's overrun, because the six
observed loops are concentrated in three specific datasets rather than spread uniformly, so a global
rate would misprice both the looping and non-looping datasets. Recorded as a note and a contingency
budget instead.

## 2026-09-21 (third entry): the reviewer objects on importance concentration, and the row records nothing the reviewer is shown
Worked out, for $0, why only 3 of 13 benchmark datasets drew reviewer objections: the reviewer
cannot see the verified or baseline score columns at all, and objects only when a profiler-nominated
column also dominates permutation importance. Added two new columns so this hypothesis is testable
going forward, since none of the facts the reviewer actually sees had been recorded before.

## 2026-09-21 (fourth entry): a reproduction check is a subset, not a set of retyped flags
Checked a standing risk about a README-quoted result by first reading the code that produced the
doubt, finding it used a different, untested arm, then running a true reproduction using the literal
same cell object rather than a retyped copy, so a later edit cannot silently turn "reproduction"
into an unnoticed comparison of different arms. The result reproduced.

## 2026-09-21 (fifth entry): a provenance helper that fails silently is worse than one that fails
Fixed `git_commit()` silently returning null on a local git configuration issue with no diagnostic,
so a run meant to close a provenance gap recorded none and reported success anyway. It now prints
the reason once per process and the harness warns before spending money, while still never raising
and still not falling back to searching for a different git binary.

## 2026-09-21 (sixth entry): the profiler's opaque recall moved, and it is recorded as a flag
Found, not pre-registered, that the profiler's opaque-naming recall on one fixture shifted across
three sessions with no code change that plausibly explains it. Recorded as a flagged caveat on the
README's published number rather than a finding, since it is post-hoc on a ranking already observed.

## 2026-09-23: docs cleanup, the walkthrough rebuild chain is gone and the archive holds the full history
This session dropped the module that captured a run's final state for the walkthrough, the CLI
flags on `ds-agents run` that fed it, the two scripts that rebuilt the walkthrough page from that
captured data, the HTML template they filled, and the captured data directory itself. The built
`docs/explainers/pipeline-walkthrough.html` page stays as a static artifact. The history/measurement
tests were cut down to one guard test each; the two opt-in timing guards (baseline fit and
dataset registration against their real timeouts) were kept because they check live code, not
recorded numbers. The full text of every decision moved to
`docs/decisions-archive.md`, leaving this file to one-line summaries. Pre-registrations and full
reasoning for every decision live there; run results and per-run narratives live in
`evals/results/LOG.md`.
