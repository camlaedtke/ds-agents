# ds-agents

How much of a data scientist's tabular workflow can you delegate to a team of agents, and how do
you know when they got it wrong?

The second half of that question is what this project measures. Six LangGraph nodes (intake,
profiler, feature engineering, modeling, adversarial review, reporting) run against 13 public
tabular datasets and 3 synthetic datasets with planted leakage. Every run writes one wide row into
`evals/results/`, and the schema grows each time the eval learns something it failed to record. The
findings are the product here, and the pipeline is the instrument that produces them.

Summary of the results:

1. **The leakage detector reads column names.** The profiler finds 19 of 20 planted traps when the
   columns are called `days_to_close` and `adjuster_touches`, and 2 of 20 when the same columns are
   renamed `var_07` and `var_08` with nothing else about the data changed. Recall goes from 0.95 to
   0.10 on one line of one file.
2. **Catching a leak and fixing it fail independently.** In the arm where the reviewer names the
   trap in 10 of 10 runs, the trap still shipped in 3 of 10. The cause was dispatch. The reviewer
   addressed column-scoped objections to a node that has no lever to drop a column.
3. **The default reviewer rarely objects.** Across 52 runs of all 13 benchmark datasets in the
   shipped configuration, the reviewer raised 12 objections in total, on 6 rows, on 3 datasets.
   Nine datasets drew zero objections in 36 runs. The three that did object appear to be the three
   whose permutation importance is concentrated in one to three columns, which is the only
   per-column number the reviewer is shown. That one is a hypothesis with a mechanism and a
   falsification test, and it is not yet established.
4. **The cost model was accurate for runs that passed first time.** A 13-dataset run came in 10.2%
   over estimate, and all of the overrun sits in the 6 runs that reviewed more than once.

---

## Setup

### The pipeline

```text
intake -> profiler -> feature_eng -> modeler -> reviewer -> router --pass--> reporter
                          ^                                    |
                          +---------------- block -------------+
```

All node I/O goes through one Pydantic object, `PipelineState`. No node reads a file, an
environment variable, or the network. Agents execute code only through a `run_python` MCP tool
whose sandbox forks per snippet with a stripped environment, a pruned `sys.path`, and no API key,
so agent-authored code cannot see this repository, the run's state, or the answer key.

The default agent model is Claude Haiku throughout. Sonnet appears only as a reviewer ablation.

### What is graded, and by what

A run's claimed score is never trusted. The harness withholds 20% of rows *before the graph starts*,
re-applies the pipeline's own emitted feature-transform code to those rows, and scores them itself.
Two reference points are fit on the agents' own train split and scored on the same withheld rows: a
class-prior constant predictor at 0.0 and a tuned RandomForest (`rf-v1`) at 1.0. Published
third-party numbers from OpenML's evaluation API are recorded per dataset and are never used as a
denominator, because they come from a 10-fold CV protocol and dividing by them would compare two
protocols rather than two systems.

Leakage is graded as a set comparison against a ground-truth column list that only the synthetic
fixtures have.

### The benchmark

13 binary-classification datasets from the AutoML Benchmark (Gijsbers et al. 2024), selected by a
committed rule (binary, `roc_auc`, <=100k rows, <=200 features, <=5M cells, >=5 usable features)
from OpenML suite 271. 22 candidates were excluded by that rule or could not be fetched. Nothing in
`evals/datasets/manifest.yaml` was typed by a person. It is written by `ds-agents datasets refresh`,
and `.claude/settings.json` denies writes to the directory.

| dataset | rows | features | positive rate |
|---|---|---|---|
| australian | 690 | 14 | 0.445 |
| credit_g | 1,000 | 20 | 0.300 |
| kc1 | 2,109 | 21 | 0.155 |
| jasmine | 2,984 | 144 | 0.500 |
| kr_vs_kp | 3,196 | 36 | 0.478 |
| sylvine | 5,124 | 20 | 0.500 |
| phoneme | 5,404 | 5 | 0.293 |
| amazon_employee_access | 32,769 | 9 | 0.058 |
| nomao | 34,465 | 118 | 0.286 |
| bank_marketing | 45,211 | 16 | 0.117 |
| adult | 48,842 | 14 | 0.239 |
| numerai28_6 | 96,320 | 21 | 0.495 |
| higgs | 98,050 | 28 | 0.471 |

None of these has a labelled leak, so leakage is measured on three synthetic fixtures instead. Each
has a committed seeded generator and a manifest written in the same breath as the CSV:

| fixture | rows | trap | mechanism | NMI with target |
|---|---|---|---|---|
| toy | 200 | `account_status_code` | noisy copy of target | (0.91 agreement) |
| claims_timing | 200 | `days_to_close`, `adjuster_touches` | recorded after the label | 0.125, 0.184 |
| reissued_ids | 200 | `member_number` | identifier re-issued by outcome | 0.094 |

---

## Result 1: the benchmark run

`evals/results/2026-09-02_full.jsonl` holds all 13 datasets, 4 replicates each, 52 rows, one commit
(`70f7547`), **$1.2165 and 41 minutes of wall clock**. 0 runs refused, 0 failed, `halted_at` null on
all 52.

Configuration: Haiku throughout, `naming=descriptive`, `reviewer_prompt=base`,
`objection_routing=as_addressed`, `objection_closure=off`, `loop_cap=3`. **This is the shipped
default configuration, and Result 3 finds better arms.** Every reviewer number below measures the
default.

| dataset | verified | rf-v1 unit | separation | normalised | features kept | $/run |
|---|---|---|---|---|---|---|
| phoneme | 0.9466 | 0.9500 | 0.450 | 0.992 | 5.0 | 0.0105 |
| amazon_employee_access | 0.8154 | 0.8572 | 0.357 | 0.883 | 7.5 | 0.0124 |
| adult | 0.9244 | 0.9027 | 0.403 | 1.054 | 12.0 | 0.0161 |
| credit_g | 0.7476 | 0.7660 | 0.266 | 0.931 | 20.0 | 0.0162 |
| bank_marketing | 0.9359 | 0.9279 | 0.428 | 1.019 | 16.0 | 0.0155 |
| numerai28_6 | 0.5211 | 0.5101 | **0.010** | **2.089** | 21.0 | 0.0170 |
| higgs | 0.8006 | 0.7914 | 0.291 | 1.031 | 28.0 | 0.0240 |
| nomao | 0.9954 | 0.9934 | 0.493 | 1.004 | 117.0 | 0.0427 |
| jasmine | 0.8610 | 0.8870 | 0.387 | 0.933 | 144.0 | 0.0421 |
| australian | 0.8772 | 0.9237 | 0.424 | 0.890 | 13.2 | 0.0290 |
| sylvine | 0.9130 | 0.9739 | 0.474 | 0.871 | 19.2 | 0.0196 |
| kc1 | 0.7675 | 0.8055 | 0.306 | 0.876 | 21.0 | 0.0179 |
| kr_vs_kp | 0.9662 | 1.0000 | 0.500 | 0.932 | 35.2 | 0.0412 |

The pipeline lands near a tuned RandomForest and beats it on 2 datasets of 13. A six-agent pipeline
costing 1.2 to 4.3 cents per dataset reaches 0.87 to 1.05 of a baseline it was not allowed to see,
and the two wins (`adult` at 1.054, `higgs` at 1.031) are small.

Two numbers need their denominators quoted alongside them:

- **`numerai28_6`'s 2.089 is an artifact of a near-chance dataset.** Its separation, meaning the
  distance between the constant predictor and the RandomForest, is 0.010 roc_auc. Normalising by
  0.010 makes any noise look like a landslide. It reproduced identically across two files, which
  confirms it is a property of the dataset rather than of a run.
- **`kr_vs_kp`'s unit point is exactly 1.0000.** A tuned RandomForest on the raw frame solves the
  dataset perfectly and the pipeline gets 0.9662. There is no trap here and nothing to fix. The
  reference system is simply better, which is the price of measuring against a system the pipeline
  is independent of.

### Claimed versus verified

The agents' self-reported score is close to the truth and biased slightly optimistic: mean claim gap
**+0.0053**, median **+0.0022**, with 4 of 52 rows exceeding 0.02 in absolute value. The worst is
`kc1` at **+0.0432** claimed over verified. This is the one place the pipeline looks better than it
deserves, and the margin is small.

### Where the time and money go

| node | mean seconds | share |
|---|---|---|
| modeler | 31.3 | 66% |
| profiler | 6.8 | 14% |
| reviewer | 4.7 | 10% |
| feature_eng | 2.9 | 6% |
| intake | 1.7 | 4% |
| reporter, router | 0.01 | 0% |

Fitting candidate models dominates, and it is the part an agent contributes least to.

---

## Result 2: the leakage detector reads names

This is the finding the project exists for.

`evals/results/2026-08-27_naming-ablation.jsonl` covers `claims_timing`, n=10 per arm, Haiku,
identical in every respect except one: in the `opaque` arm the CSV header line is rewritten so all
non-target columns become `var_01 … var_NN`. Every other byte of the file is copied through. All
columns are renamed, including the innocent ones. An opaque name that only the traps carried would
hand the profiler a replacement cue for the one being taken away.

| arm | profiler recall | profiler caught | leak remediated | claimed score |
|---|---|---|---|---|
| descriptive | **0.950** | 10/10 | 8/10 | 0.840 |
| opaque | **0.100** | 2/10 | **0/10** | 0.974 |

The traps are the same traps, at the same NMI with the target (0.125 and 0.184), in the same rows.
The profiler is given exactly the same deterministic statistics either way. What it is actually
doing is recognising that a column called `days_to_close` sounds like it was recorded after the
claim closed. That is good reasoning, and it is a different thing from the statistical detection the
node's name implies.

The consequence is in the last column of that table. In the descriptive arm the pipeline drops the
traps and honestly claims 0.840. In the opaque arm it keeps them and claims **0.974** against a
legitimate ceiling near 0.82. A pipeline that reports 0.974 on this data has been fooled, and it
reports the number with no hedge.

Everything after this section runs in the opaque arm, because it is the only configuration that
reliably puts a leaky matrix in front of the reviewer at all.

---

## Result 3: catching a leak and fixing it are separate failures

The reviewer is the adversarial half of the design. It sees the model, its scores, and its top
permutation importances, and it may raise typed objections that route the graph back to an earlier
node, up to `loop_cap` times.

All rows below are `claims_timing --naming opaque` on the Haiku default model.

| condition | n | reviewer named the trap | leak remediated | file |
|---|---|---|---|---|
| `base` prompt | 10 | 1/10 | 0/10 | reviewer-ablation |
| `which_column` prompt | 10 | **9/10** | **1/10** | reviewer-ablation |
| `which_column` + `by_category` routing | 10 | 8/10 | **5/10** | objection-routing |
| ditto, second run | 10 | 9/10 | **9/10** | objection-closure |
| ditto, CI baseline | 10 | **10/10** | **9/10** | ci-baseline |

Three things happen down that table, and they are three different bugs.

**The default reviewer does not look.** Under the shipped `base` prompt it named a trap in 1 of 10
runs. `which_column` appends one rule, *say which top-importance column explains an implausible
score*, and detection goes to 9 of 10. The reviewer was capable the whole time and was never asked.

**Detection bought almost nothing on its own.** At 9 of 10 detection, remediation was 1 of 10. The
loop budget was the obvious suspect and it was ruled out: a pre-registered sweep at `loop_cap` 1 / 3
/ 5, n=10 each, moved remediation to **0, 1, 1 out of 10** while quadrupling the cost per run
($0.0141 to $0.0553). The cap was never the bottleneck.

**The cause was dispatch.** The reviewer addressed `implausible_importance` objections to `modeler`,
a node that chooses which candidate to promote and has no lever to drop a column, so `feature_eng`
never saw them. `objection_routing=by_category` gives a column-scoped objection an *effective*
target of `feature_eng` regardless of what the reviewer wrote, applied in exactly one place.
Remediation went from 1/10 to 5/10 to 9/10.

`Objection.target_node` is never rewritten, so the row still records the reviewer's own dispatch
judgement even in the arm that overrides it. The override is visible in the data it produced.

**The second trap mechanism confirms Result 2 from the other direction.** `reissued_ids` plants an
identifier re-issued by outcome (`member_number`) and places a genuine unique identifier
(`application_ref`) beside it as a control. Under opaque naming they become `var_02` and `var_01`.
In 10 runs the profiler nominated **`var_01`, the innocent control, in 10 of 10** and the guilty
`var_02` in **3 of 10**. It flagged the harmless identifier every single time and missed the leaking
one more often than it found it. Two columns that look identical to a name reader and differ only in
what generated them are exactly the case the profiler cannot handle, and this fixture was built to
contain that case. The reviewer recovered some of it, naming the trap in 5 of 10 with remediation in
8 of 10, and it also objected to `var_03` (`credit_score`, a legitimate strong feature) in 2 runs.

**Two costs, both recorded.** `by_category` turns a reviewer false positive into a real deletion:
2 of 10 runs dropped `prior_claims_12m`, a legitimate strong feature. And the routing arm came out
cheaper per run than the control, which contradicted its own pre-registration.

### The reviewer model matters less than the reviewer prompt

`evals/results/2026-08-28_reviewer-ablation.jsonl`, prompt crossed with model:

| reviewer | prompt | n | named the trap | remediated | $/run |
|---|---|---|---|---|---|
| Haiku | base | 10 | 1/10 | 0/10 | 0.0151 |
| Haiku | which_column | 10 | 9/10 | 1/10 | 0.0299 |
| Sonnet | base | 7 | 0/7 | 0/7 | 0.0809 |
| Sonnet | which_column | 7 | 5/7 | 3/7 | 0.0913 |

The prompt effect holds under both models. Sonnet under `base` found nothing in 7 runs at 5x the
price. There is a model effect, and it shows up in remediation: **Haiku names a trap more often and
Sonnet fixes it more often**, because Sonnet addresses its objection to `feature_eng` instead of
`modeler`. That is the same dispatch bug arrived at from the other side. The Sonnet cells are n=7
against a pre-registered n=8, so they are directional and not established.

---

## Result 4: what the default reviewer does on real data

On the 52 benchmark rows, in the shipped configuration:

- **12 objections in total, on 6 of 52 rows, on 3 of 13 datasets.** Nine datasets drew zero
  objections in 36 runs.
- The three datasets that drew any (`australian`, `sylvine`, `kr_vs_kp`) are among the four that had
  never been run before that session.
- First `exhausted` verdicts ever recorded on a benchmark dataset: 4.
- One `kr_vs_kp` run **reproduces the dispatch failure outside the fixtures for the first time**,
  with two `implausible_importance` objections addressed to `modeler` and a finish of
  `objected_columns_unremediated: ['bxqsq', 'rimmx', 'wknck']`.
- Another `kr_vs_kp` run raised two `leakage` objections on a dataset with no answer key, so nothing
  can say whether it was right.

**This file cannot distinguish "nothing to object to" from "not looking."** Those nine datasets
carry no planted trap, so zero objections may well be correct. Result 3 shows the same configuration
scoring 1/10 on data that definitely does contain a trap, which is the reason to suspect the second
reading.

### Why those three datasets, and what the reviewer can actually see

Two facts about the reviewer's input constrain the answer, and both come from reading
`reviewer.py:_user_message` rather than from the rows.

The reviewer is never shown the reference points. `verified_holdout_score`, `baseline_separation`,
`baseline_normalised_score` and `holdout_claim_gap` appear nowhere in its message. It cannot react
to the pipeline underperforming its own baseline, so "low normalised score" is ruled out as a
trigger on structural grounds. The scores confirm it empirically: ranked by normalised score, the
objectors and the silent datasets interleave (`sylvine` 0.871 objects, `kc1` 0.876 is silent,
`amazon_employee_access` 0.883 is silent, `australian` 0.890 objects).

The base prompt then imposes a two-part gate: raise an objection only if you can name the columns at
issue and state the number behind it. `_adjudicate` enforces the first half mechanically, rejecting
a column-scoped objection outright when its columns intersect nothing in the profile. **The only
per-column number anywhere in the reviewer's input is `top_importances`.**

That gives a candidate mechanism. The profiler's nominations reach the reviewer by name, through a
"Kept despite a profiler flag" line in the feature summary, and a concentrated importance
distribution supplies the number. Both are needed, and a nomination on its own is not enough: the
profiler nominated columns on 6 of 13 datasets and the reviewer objected on 3. Recomputing both
statistics offline, the number of columns needed to reach 80% of total positive permutation
importance is 1, 1 and 3 for `sylvine`, `australian` and `kr_vs_kp`, and 4 or more for all ten
silent datasets. The reviewer also named precisely the head of that distribution and cut it at the
cliff, taking the top three on `kr_vs_kp` where rank 4 is 6x smaller, and never naming a rank-4
column on any dataset. The same gate explains an oddity in the categories: all 4 `metric_mismatch`
objections are on `australian`, on the passes where the concentrated column had already been
dropped, leaving a flat importance list and no column-scoped objection that could survive
`_adjudicate`.

**This is a hypothesis with a mechanism, and it is not established.** The threshold separates 3
datasets from 10 on a single post-hoc ranking at n=13, worth roughly p = 0.02 after accounting for
the statistics tried. More importantly the importances were recomputed offline from a proxy
RandomForest, because **the pipeline's own `top_importances` was never recorded on the row.** Not
one of the 83 columns on those rows is a fact the reviewer is shown, which is why the committed
rows cannot settle this. `top_importance_share`, `top_importance_n80` and a `top_importance_status`
companion are now emitted so that future runs measure it directly. Read back off the five committed
run envelopes the numbers behave: `reissued_ids` reads a share of 1.00 at n80=1, where one planted
column dominates and every other importance is negative, and `kc1` reads 0.199 at n80=8, which is
the flat shape that leaves the reviewer nothing to cite. The falsification test is a redundancy
ablation: duplicate the dominant column so permutation importance collapses while per-column NMI
holds steady, which moves this trigger and nothing else, for about $0.40.

---

## Result 5: cost, and an assumption that cost more than the model

The cost estimate is `$0.010163 + $0.000230 * n_features + $0.005609 * (n_rows/1e5)`, fit on 9
datasets with residual standard deviation $0.0022, re-derived by test rather than pasted.

The 13-dataset run came in at **$1.2165 against a $1.1040 estimate, or +10.2%**, under a
pre-registered $1.60 cap. The tempting read is that the cost model failed. It held up:

- All **46** runs that reviewed exactly once landed within **0.95x to 1.04x** of their cell's price.
- All of the overrun is in two cells, `australian` at **+81.0%** and `kr_vs_kp` at **+87.4%**, each
  of which sent 2 of 4 runs to `review_loops=3`.
- `kc1`, also modelled from the fit and also never run before, but with four first-pass runs, came
  in at **-0.5%**. That control is what makes the rest readable.
- Restricted to first-pass runs, the two overrunning cells read **-7%** and **+7.7%**.

The error was in a sentence printed above the coefficients in `SUBSETS`, correct and committed for
two sessions: *every price assumes a run that passes first time.* It was a stated, load-bearing
assumption that nobody had costed.

**The loop multiplier, measured on benchmark data for the first time**, within dataset against that
dataset's own first-pass mean:

| review loops | n | cost multiplier |
|---|---|---|
| 1 | 46 | 1.00 (reference) |
| 2 | 2 | **x1.80** |
| 3 | 4 | **x2.46** |

That sits against a standing "about 2.5x" carried from the 200-row toy fixture and never checked
until now. It agrees, which counts as reassurance rather than replication.

**A 13-dataset refit is declined.** The residuals are a bimodal split on a variable the model does
not contain. Re-fitting `a + b*n_features + c*n_rows` on all 13 would launder a known mechanism into
a larger intercept, making the estimate worse-founded while making it look better.

**A loop-rate term is also declined, having been worked out first.** The obvious repair is to price
the expected multiplier into every cell. Two things rule it out. The looping is piled into three
datasets (`australian` 3/4, `kr_vs_kp` 2/4, `sylvine` 1/4, and 0/4 on the other ten), and a
goodness-of-fit against a single constant rate applied uniformly gives chi-square(12) = 27.5,
p = 0.0065, so the data reject one global rate on their own terms. And applying one anyway moves the
error around: a global expected multiplier of 1.1431 predicts $1.2620 against the actual $1.2165,
charging the ten quiet datasets a 17.3% premium for behaviour they have never shown while still
pricing the three noisy ones at 71% of what they cost. The loop rate is 6/52, or 11.5% with a 95%
interval of [4.3%, 23.4%], which is recorded as a contingency note beside the coefficients and
pinned by a test as history. `docs/DECISIONS.md`, 2026-09-21, has the full argument.

**Which datasets loop is a reviewer question, not a size question.** The two standing guesses, that
`australian` is the smallest dataset (690 rows) and `kr_vs_kp` the most categorical (36 columns, all
categorical), both fail on the ordering. Result 4 has the candidate mechanism and its falsification
test. Until that test runs, a cost estimate cannot predict which cells will loop, which is the
reason the contingency above stays a range instead of becoming a coefficient.

---

## Design notes

Choices that shaped what could be measured, one paragraph each. Full reasoning is in
`docs/DECISIONS.md`.

**One state object, and nodes that cannot reach around it.** Every node reads and writes
`PipelineState` and nothing else. That is the reason an ablation is a field on a frozen `RunConfig`
instead of a branch: `naming`, `reviewer_prompt`, `objection_routing`, `objection_closure`,
`loop_cap` and `reviewer_model` are all recorded on every row, so a results file says what condition
produced it and `evaldiff` can refuse to compare two files that differ in more than one.

**The name-transparency ablation is a load-time header rewrite.** It is not a manifest field or a
paired fixture. One line of one CSV changes and every other byte is copied through, so the two arms
differ in exactly the thing being tested. This is why Result 2 is a clean number.

**The fixture manifest is written by the generator, and no node can read it.** `generate.py` writes
the CSV and the ground-truth manifest in the same breath, so they cannot drift. Nothing in `nodes/`
imports the fixture registry, because a node that could load a manifest could read the answer key,
and `planted_leakage_columns` would stop measuring anything.

**A column whose null carries information gets a companion column that says why.**
`leakage_graded` says whether the nine leakage columns were measured at all, which saves a reader
from inferring it from nine separate nulls. `rescore_status` does the same for the verified score
and `baseline_status` for the baseline. The pattern was arrived at three times before it was named,
and `top_importance_status` is the fourth, separating "no importances at all" from "importances
that are all zero or negative". One variant of it is still unresolved (see Limits).

**The baseline publishes the length of its own scale.** Two raw points (constant predictor,
RandomForest) plus the normalisation, rather than one ratio, because a single number cannot say
which end of the scale it is on. `numerai28_6`'s 2.089 is legible only because 0.010 is printed
beside it. The normalised column is suppressed entirely on a dataset with a planted leak, where a
baseline fit on the raw frame keeps the trap and a correct pipeline therefore scores below it.

**The sandbox is a forked worker, not a container.** Same isolation per snippet, a twentieth
of the cost, and `pytest -m fast` does not depend on a Docker daemon. Docker stays deferred behind a
`SandboxPool` seam until model-authored code needs a memory cap and a network block. The docstrings
say plainly that this is not a jail.

**Pre-registration, including the parts that were boring.** Each funded run has its endpoints
written into `docs/DECISIONS.md` before the money is spent, including what would make the resulting
file uninterpretable. That is why "no count in the 52-row file may be quoted as an effect" is a
constraint stated in advance instead of a caveat discovered afterwards.

---

## Limits

Stated because a results file that overstates its reach is worse than a smaller one.

**On the benchmark rows.** n=2 per cell per replicate, so no count in the 52-row file earns a Wilson
interval and none is quoted as an effect. The withheld holdout is a random split, so it cannot catch
a temporal or grouped leak. All 52 rows come from one configuration, the shipped default, so its
reviewer numbers do not describe the pipeline's best known arm.

**On whether Result 3 still describes the pipeline.** The `claims_timing` arms were run at older
commits, and there is evidence in both directions about whether the fixture still behaves that way.
A later session recorded two fresh replicates that passed first-loop with zero objections, both
planted leaks already dropped by feature engineering's own plan, against 10 of 10 committed rows
that looped. If that shift is real, several of Result 3's numbers describe a pipeline that no
longer exists; if it is nondeterminism, they stand and are underpowered. Unresolved, and about
$0.30 of replicates would settle it.

**On the leakage rows.** 200 withheld rows put roc_auc's standard error near 0.04 on the fixtures,
which is wide next to several of the differences above. Three traps is not a taxonomy.
Duplicate-rows-across-split is structurally uncatchable today, because the reviewer is never shown
the split or any rows. And leakage is scored as a set comparison over guilty columns, so a trap with
no guilty column cannot be scored at all.

**On provenance, which is the one asymmetry in the results.** The 52 benchmark rows are uniform: 83
columns and one commit, `70f7547`. `src/` has changed since that commit, and the changes are
walkthrough capture tooling, a CLI flag, and three additive result columns. No node and no part of
the grading path was touched, so the rows still describe the pipeline that runs today. The **145
fixture rows carrying every leakage finding above are not uniform.** They were written by earlier
code and carry 36 to 62 columns, missing 23 to 49 of the 83 the benchmark rows have. `commit` is
among the missing ones, so **no leakage number in this README can be attributed to a specific
commit**, and its provenance is a filename and a date. Those rows also predate `leakage_graded`
and `verified_holdout_score`, with two consequences: no committed row anywhere carries
`leakage_graded: true`, so a table script has to
filter on a non-empty `leakage_planted` instead, and **the claimed-versus-verified check in Result 1
exists only on the benchmark side.** The fixture scores in Results 2 and 3 are the agents' own
claims, re-verified by the set comparison against ground truth but not by an independent rescore.
Those files also still carry `baseline_score` and `score_ratio`, retired on 2026-08-31 and pinned as
must-never-reappear by test. Nothing above quotes them.

**Open defects, named here.** `errored` is true on exactly one of 52 rows, and both of that row's
errors are a recovery that worked: the reviewer claimed `block` with nothing actionable, was
re-asked once, and the retry produced a usable objection. A successful recovery gets
recorded as an error, because `errored` has no way to say "and it was handled." `exhausted` is still
uninformative about whether a trap shipped, and `leakage_remediated` is the column that answers
that. `n_candidates_failed_to_fit` is 0 on all 52 rows, which shows the column is quiet rather than
showing that it works. And the reviewer only ever sees the finished model, so it detects late, which
is a structural defect that no prompt fixes.

---

## Reproducing

```console
uv sync
uv run pytest                              # 864 tests, includes the toy end-to-end run
uv run ds-agents run --dataset toy         # one pipeline run, prints the node trace
uv run ds-agents datasets verify --online  # check the manifest against OpenML
uv run ds-agents eval --subset ci          # the fixture subset, ~$0.73
uv run ds-agents eval --subset full        # all 13 benchmark datasets, ~$1.22
uv run ds-agents eval-diff A.jsonl B.jsonl # refuses to call an underpowered difference an effect
```

With no API key the pipeline falls back to `StubModel`, every event is stamped `model="stub"`, and
`PipelineState.publishable()` refuses to write a results row, because a row built from a stub would
look like a system that never finds anything.

Every table above is regenerated from the committed rows by `uv run python docs/readme_numbers.py`,
so no number here rests on prose in a log. Run it after adding results and update whatever moved.

`docs/ARCHITECTURE.md` is the contract, `docs/DECISIONS.md` the log of why, `docs/PLAN.md` the
phases, and `evals/results/LOG.md` a per-run narrative of every funded run.
