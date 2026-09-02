# Eval log

Dated entries, newest at the bottom. See /run-eval.

## 2026-08-27 — naming ablation, `claims_timing`, n=10 a side

`2026-08-27_naming-ablation.jsonl`, 20 rows. First committed results file in the project; produced
by `ds-agents run --dataset claims_timing --naming {descriptive,opaque} --repeat 10 --results ...`,
live Haiku over MCP, `loop_cap=3`, `random_seed=20260822` throughout (model nondeterminism is the
only source of variance). Total $0.343, ~7.7 minutes.

|                              | descriptive | opaque |
| ---------------------------- | ----------- | ------ |
| profiler nominated >=1 trap  | 10/10       | 2/10   |
| mean `profiler_recall`       | 0.95        | 0.10   |
| trap survived into matrix    | 2/10        | 10/10  |
| mean claimed roc_auc         | 0.840       | 0.974  |
| reviewer leakage objections  | 0/10        | 0/10   |
| mean `profiler_false_alarm`  | 1.00        | 1.20   |
| verdict pass / exhausted     | 9 / 1       | 8 / 2  |
| mean $ / seconds             | 0.0174 / 23.4 | 0.0169 / 22.5 |

Legitimate roc_auc ceiling on this fixture is near 0.82, so the opaque arm's 0.974 is the traps
being leaned on. Across both arms the trap survived in 12 runs and the reviewer raised a leakage or
contamination objection in 0 of them. The honest identifier `claim_ref` was nominated in 10/10
descriptive and 9/10 opaque runs, which is the control saying the profiler's *statistical* flagging
is name-invariant while its *semantic* flagging is not. See DECISIONS.md, same date.

Not comparable to the previous session's 3-of-3 vs 1-of-3: that ad hoc ablation renamed only the
trap columns, this one renames every non-target column.

## 2026-08-28 — reviewer 2x2: model arm x prompt arm, `claims_timing --naming opaque`

`2026-08-28_reviewer-ablation.jsonl`, 27 rows, all from commit `2d5c1fc`. Live over MCP, upstream
nodes Haiku in every cell, `loop_cap=3`, `random_seed=20260822`. Total $1.041, ~23 minutes.

```
ds-agents run --dataset claims_timing --naming opaque \
  --reviewer-model {haiku,sonnet} --reviewer-prompt {base,which_column} \
  --repeat N --results evals/results/2026-08-28_reviewer-ablation.jsonl
```

| cell | n | reviewer named >=1 trap | mean `reviewer_recall` | mean `reviewer_false_alarm` | verdicts (pass/block/exhausted) | mean $ |
| ---- | - | ----------------------- | ---------------------- | --------------------------- | ------------------------------- | ------ |
| haiku / base         | 10 | 1/10 | 0.05 | 0.00 | 9 / 0 / 1 | 0.015 |
| haiku / which_column | 10 | 9/10 | 0.70 | 0.00 | 0 / 2 / 8 | 0.030 |
| sonnet / base        | 4  | 0/4  | 0.00 | 0.00 | 1 / 0 / 3 | 0.078 |
| sonnet / which_column| 3  | 2/3  | 0.67 | 0.33 | 2 / 0 / 1 | 0.093 |

**The prompt is the binding constraint, not the model.** One appended rule moves Haiku from 1/10 to
9/10. Sonnet under the base prompt is 0/4 -- no better than Haiku, and by the pre-registered rule
(>=5/10 to call an effect shown) the model main effect is not resolved and is directionally absent.
The failure the prompt fixes is specific: under `base` both models raise objections about the
*number* (`metric_mismatch`, `overfit`) and never name a column; `objections_by_category` across the
two base cells is 3 metric_mismatch/overfit and 2 implausible_importance against 0 leakage, while
the two `which_column` cells raise 8 leakage, 1 contamination and 17 implausible_importance.

**Cell n is unequal and the Sonnet cells are underpowered.** Sonnet cost $0.078-0.093 per run
against a budgeted $0.035 -- it blocks, so it runs three reviewer passes at ~$0.03 each -- and the
pre-registered $1.20 hard stop bound the Sonnet half to n=4 and n=3. Cell C ran as 2+2 under the
cost rule; its outcome fields were not read between batches. Sonnet at n>=10 is the first thing to
buy next session, at roughly $0.85.

**Upstream control (identical in every cell, and it is).** `profiler_recall` 0.25 / 0.20 / 0.25 /
0.167, `profiler_false_alarm` 1.0 / 1.2 / 1.0 / 1.33, trap survived into the matrix 25 of 27 runs.
Consistent with the 2026-08-27 opaque arm (0.10, 1.20, 10/10), so no cell difference is an artefact
of something moving upstream of the reviewer.

**Two things the prompt arm costs.** Wall time and money roughly double (1.2 to 2.7 mean loops),
and 8 of 10 Haiku `which_column` runs end `exhausted` -- the reviewer names the trap, the loop cap
runs out before it is removed, so `reviewer_caught` 9/10 becomes `leakage_remediated` 1/10. Naming
the column is necessary and not sufficient. Two Haiku `which_column` runs also hit a recoverable
router error ("reviewer claimed block with no open objection"), which is the reviewer objecting in a
category the node then dropped; not seen in any other cell.

**One run worked end to end**, the first in the project: sonnet/which_column row 1 named both traps,
`feature_eng` dropped them, and the claimed roc_auc fell to 0.761 -- below the ~0.82 legitimate
ceiling -- with a final verdict of `pass`.

Not comparable field-for-field to `2026-08-27_naming-ablation.jsonl`: `reviewer_*` and
`objections_by_category` did not exist when those rows were written and cannot be back-filled.
Shared fields (`profiler_*`, `leakage_*`, verdicts, cost) are comparable.

---

## 2026-08-28 (second run of the day): loop-cap sweep -- `2026-08-28_loop-cap-sweep.jsonl`

20 live rows, $0.694, Haiku everywhere, `claims_timing --naming opaque --reviewer-prompt
which_column --tools mcp`. Ran at the working tree of the commit that carries this file; the code and
this file must land together, because the four `route_sequence`/`objections_by_target_node`/
`objected_columns_unremediated`/`new_objections_per_pass` fields in every row do not exist before it.

Commands:

```
uv run ds-agents run --dataset claims_timing --naming opaque --reviewer-prompt which_column \
  --loop-cap 1 --repeat 10 --tools mcp --results evals/results/2026-08-28_loop-cap-sweep.jsonl
# and the same with --loop-cap 5
```

**The `loop_cap=3` arm is not in this file.** It is the haiku/which_column cell of
`2026-08-28_reviewer-ablation.jsonl` (n=10, commit `2d5c1fc`), reused rather than re-run to keep the
sweep inside its pre-registered budget. Everything else about that cell is identical. The
consequence is that the middle row of the table below is missing the four new fields, so its
`route_sequence` and `objections_by_target_node` columns read as blanks and not as zeroes.

| `loop_cap` | n | named >=1 trap | mean `reviewer_recall` | **remediated** | exhausted | mean loops | mean $ |
| ---------- | - | -------------- | ---------------------- | -------------- | --------- | ---------- | ------ |
| 1 | 10 | 7/10 | 0.60 | **0/10** | 10/10 | 1.0 | 0.0141 |
| 3 | 10 | 9/10 | 0.70 | **1/10** | 8/10  | 2.7 | 0.0299 |
| 5 | 10 | 9/10 | 0.80 | **1/10** | 9/10  | 4.6 | 0.0553 |

**The cap is not the bottleneck.** Remediation is 0, 1, 1 out of 10 across a 5x range of the cap,
while cost per run quadruples. Buying four extra reviewer passes bought one extra tenth of a point
of recall and no additional remediated run at all. Pre-registered as the Phase 5 loop-cap ablation;
recording it as a null result on remediation, which is what it is.

**What the new fields say instead**, at `loop_cap=5`: 5 of 10 runs raised every one of their
objections against `modeler` and none against `feature_eng`, and only 4 of 10 runs ever routed to
`feature_eng` at all. A `modeler`-targeted objection cannot remove a column -- every candidate is
fit on the one transform `feature_eng` already froze -- so those runs had no path to remediation at
any cap. At `loop_cap=1` the same split is 5 of 10 `modeler`-only, and no run routes anywhere: a
single pass is spent before the loop can begin.

**Diagnostic runs (3 runs, $0.102, not written to any results file)** established the mechanism
before the sweep was designed; the sweep confirms it. Two causes, both independent of the cap:

1. **Misrouting.** In 2 of 3 diagnostic runs every objection was `implausible_importance` addressed
   to `modeler`. `feature_eng` force-drops an objected column before its own model is consulted, but
   only for objections addressed to it, so the traps were never removed.
2. **No closure.** In 3 of 3 the reviewer never dispositioned an objection `resolved`. The third run
   dropped both traps, saw the claimed roc_auc fall 0.986 -> 0.823, wrote that this "is consistent
   with removing leakage", and marked the objection `still_open` anyway because the columns "were
   never validated as non-leaking, only removed". A run that remediated fully still scored
   `exhausted`.

The second one matters for how this file is read: **`exhausted` is not the same as "the trap
shipped"**. `leakage_remediated` is the field that answers that, and it is the one quoted above.

---

## 2026-08-28 (third run of the day): Sonnet cells topped up -- `2026-08-28_reviewer-ablation.jsonl`

7 rows appended to the existing file, $0.613, bringing both Sonnet cells from n=4 and n=3 to **n=7
each**. Commands (`--loop-cap 3` stated explicitly so the new rows match the cell they join, though
3 is the default):

```
uv run ds-agents run --dataset claims_timing --naming opaque --reviewer-model sonnet \
  --reviewer-prompt base --loop-cap 3 --repeat 3 --tools mcp --results <this file>
# and --reviewer-prompt which_column --repeat 4
```

**n=7, not the pre-registered n=8.** The loop-cap sweep cost $0.694 against an estimate of $0.57,
and the session's $1.50 cap left $0.70. n was cut on cost alone, before any Sonnet outcome field was
read; the cells stay equal-n. Session total: $1.409.

The 7 new rows carry the four new loop fields; the 27 rows written at commit `2d5c1fc` do not, and
cannot be back-filled. Shared fields are comparable.

| cell | n | named >=1 trap | mean `reviewer_recall` | **remediated** | exhausted | mean $ |
| ---- | - | -------------- | ---------------------- | -------------- | --------- | ------ |
| haiku / base          | 10 | 1/10 | 0.05 | 0/10 | 1/10 | 0.0151 |
| haiku / which_column  | 10 | 9/10 | 0.70 | 1/10 | 8/10 | 0.0299 |
| sonnet / base         | 7  | 0/7  | 0.00 | 0/7  | 6/7  | 0.0809 |
| sonnet / which_column | 7  | 5/7  | 0.57 | **3/7** | 1/7 | 0.0913 |

**The prompt effect holds and strengthens**: Sonnet under `base` is now 0 of 7, at n=7 rather than
n=4. Nothing about the base prompt gets either model to name a column.

**The model main effect has separated from the prompt effect, and it is not about detection.**
Under `which_column`, Haiku names a trap in 9 of 10 runs and Sonnet in 5 of 7 -- Haiku detects
*more*. But Sonnet remediates 3 of 7 against Haiku's 1 of 10, and ends `exhausted` in 1 of 7 against
8 of 10. Reported as directional at this n, not shown: the pre-registered rule is >=5/10.

**Why, in one number.** Across all 27 rows that carry `route_sequence` (this file's 7 plus the
sweep's 20):

| | n | remediated |
| - | - | ---------- |
| routed to `feature_eng` at least once | 6 | 3 (50%) |
| never routed to `feature_eng` | 21 | **0 (0%)** |

Reaching `feature_eng` is necessary for remediation and is not sufficient. The cell differences
above are almost entirely differences in how often a run gets there: sonnet/which_column 2 of 4,
haiku/which_column at `loop_cap=5` 4 of 10, sonnet/base 0 of 3, haiku/which_column at `loop_cap=1`
0 of 10. Sonnet's advantage is that it addresses its objection to the node that can act on it.

## 2026-08-28 (fourth run of the day): objection-routing arm -- `2026-08-28_objection-routing.jsonl`

**Pre-registered before the cell was run.** Everything from here to the results table below was
written first; the commit that carries this file carries the pre-registration and the rows together.

The claim under test: the binding constraint on remediation is the pipeline's contract, not the
reviewer's competence. `objection_routing="by_category"` gives a column-scoped objection an
effective target of `feature_eng` whatever the reviewer addressed it to, without rewriting what the
reviewer said. See DECISIONS.md 2026-08-28 (third entry).

```
uv run ds-agents run --dataset claims_timing --naming opaque --reviewer-prompt which_column \
  --loop-cap 3 --objection-routing by_category --repeat 10 --tools mcp \
  --results evals/results/2026-08-28_objection-routing.jsonl
```

**Control, not re-run:** the haiku/`which_column` cell of `2026-08-28_reviewer-ablation.jsonl`
(n=10, commit `2d5c1fc`), which is `--objection-routing as_addressed` in all but name -- that arm
is the default and is byte-identical to it, and the 445-test suite passes with zero edits to the
router's own tests, which is the evidence for that claim. Those rows predate `route_sequence`,
`objections_rerouted`, `objection_routing` and `n_final_features`, so **only `leakage_remediated`
crosses the boundary cleanly**; the mechanism and harm columns below read as blanks on the control,
not as zeroes.

Endpoints, fixed in advance:

- **Primary: `leakage_remediated`.** House rule, unchanged since the 2x2: **>=5/10 is shown, 3-4/10
  is directional, below that is not resolved at this n.** Control is 1/10. The mechanical ceiling is
  about 8/10 -- of the 9 control runs that caught a trap, 5 named both planted columns, and 3 of the
  4 that named only `var_07` had `var_08` already dropped upstream.
- **Mechanism: `route_sequence` contains `feature_eng`** (control: 0 of 10 haiku rows ever did) and
  **`objections_rerouted` > 0**. Without these a remediation number is a coincidence, not a result.
- **Harm: `n_final_features`, mean `claimed_holdout_score` against the ~0.82 legitimate ceiling,
  `reviewer_false_alarm`.** This arm converts a reviewer false positive into a really dropped
  feature -- under `as_addressed` a wrong objection sent to `modeler` was harmless. The one cap=5
  run that remediated also objected to `var_04`, which is `prior_claims_12m` and a legitimate strong
  feature. A remediation number quoted without a matrix width beside it is not honest.
- **Descriptive, explicitly NOT an endpoint: `review_verdict`.** **`exhausted` is expected to stay
  high in both arms**, because the closure half of the diagnosis is deliberately unfixed: the
  reviewer never dispositions an objection `resolved`. A flat verdict column is the predicted
  result here and must not be read as the routing fix failing. It is also the cleanest evidence that
  the two causes are independent -- routing should move `leakage_remediated` and leave
  `review_verdict` alone, and closure should do the opposite.
- **Budget: $0.65 cap, $0.47 expected** (~$0.047/run: one pass plus two `feature_eng` returns,
  decomposed from the committed sweep rows, against $0.030/run for the control). If the cell exceeds
  the cap it stops where it stops and n is reported as what actually ran.

### Results -- 10 rows, $0.272, well inside the $0.65 cap

Ran at the working tree of the commit carrying this file. The `objection_routing`,
`objections_rerouted` and `n_final_features` fields do not exist before it.

| | control `as_addressed` (n=10, `2d5c1fc`) | **`by_category` (n=10)** |
| - | - | - |
| **`leakage_remediated`** | **1/10** | **5/10** |
| `reviewer_caught` | 9/10 | 8/10 |
| mean `reviewer_recall` | 0.70 | 0.65 |
| route reached `feature_eng` | *(field absent)* | **8/10** |
| `objections_rerouted` > 0 | *(field absent)* | 7/10 |
| verdict pass / exhausted / block | 0 / 8 / 2 | 3 / 4 / 3 |
| mean loops | 2.7 | 2.2 |
| mean `claimed_holdout_score` | 0.9228 | 0.8812 |
| mean `reviewer_false_alarm` | 0.0 | 0.4 |
| `n_final_features` | *(field absent)* | 4-7 (median 5) |
| mean $/run, wall | 0.0299, 40.6s | 0.0272, 34.5s |

**Primary endpoint met at the pre-registered threshold: 5/10 against 1/10, which is exactly the
">=5/10 is shown" line.** Not above it. Read it as the threshold being reached, not cleared.

**The mechanism fired.** 8 of 10 runs routed to `feature_eng`; 7 of 10 contained at least one
objection the graph re-addressed. The control rows cannot be compared directly on this because they
predate the field; the standing cross-cell figure is that 0 of the 21 runs that never routed to
`feature_eng` remediated, against 3 of 6 that did.

**It got cheaper, not dearer, and this contradicts the pre-registration.** $0.0272/run against a
predicted $0.047 and a control of $0.0299. The estimate assumed `by_category` would spend the whole
cap; instead it *shortens* runs -- the objection gets acted on, so the loop terminates rather than
grinding to `exhausted`. Mean loops fell 2.7 to 2.2 and three runs ended `pass`, which no control
run did. The prediction that `exhausted` would stay flat was wrong in the run's favour and should be
recorded as a wrong prediction, not quietly dropped: fixing routing turns out to partly relieve the
closure symptom, because a reviewer whose objection is actually satisfied sometimes does close it.

**Every one of the 5 non-remediating runs failed for a reason that is not routing.**

- **3 runs (`new_objections_per_pass == [1,1,1]`, all `exhausted`) named the surviving trap on the
  *final* reviewer pass.** A block at the cap becomes `exhausted` and routes to the reporter, so
  `feature_eng` never runs again and the drop cannot land. `feature_eng` had re-run after passes 1
  and 2 in each of these, and forced drops cover every open column-scoped objection, so a trap named
  earlier would have gone -- which is what dates the naming to the last pass. **The effective number
  of actionable reviewer passes is `loop_cap - 1`, not `loop_cap`.** That is a new finding and it is
  not what the loop-cap sweep measured, because under `as_addressed` no pass was actionable at all.
- **2 runs claimed `block` with zero objections raised** (`errored: true`, `route_sequence:
  ["reporter"]`, nothing dropped, claimed 0.9858). This is the parking-lot error from the reviewer
  ablation, which had appeared twice there and in none of the 27 rows after it. It is now reproduced
  twice more and is a reviewer-side fault, independent of this arm.

So of the 8 runs where the reviewer raised anything at all, 5 remediated; of the runs that named a
trap before the final pass, effectively all did.

**Harm: real, and it is the price named in the pre-registration.** 2 of 10 runs dropped `var_04` =
`prior_claims_12m`, which the fixture lists as a legitimate strong feature. Under `as_addressed`
that same false objection was inert because it went to a node with no lever; under `by_category` it
costs a real column. `n_final_features` stayed 4-7, so no run remediated by gutting the matrix, and
mean `claimed_holdout_score` fell 0.9228 -> 0.8812 toward the ~0.82 legitimate ceiling, which is the
direction that indicates leakage leaving rather than damage. The honest summary is that the arm
buys 4 extra remediated runs and spends 2 legitimate features doing it.

**Language rule, per DECISIONS.md:** this 5/10 is *reviewer detection plus scripted dispatch*. It is
not "the agent team remediated 5 of 10". The claim it supports is that the pipeline's contract, not
the reviewer's competence, was the binding constraint.

### A latent bug found while diagnosing this cell: a forced drop is not sticky

Not the cause of anything above -- the three surviving-trap runs never resolved an objection, so it
could not have fired -- but it is reachable and it was found by chasing them, so it is recorded here
rather than lost. `_forced_drops` recomputes from `open_objections` on every invocation, so once the
reviewer dispositions its objection `resolved`, a *later* return to `feature_eng` for some other
objection silently puts the leaked column back in the matrix. Verified directly against the node:
with the objection open the snippet reads `DROP = ['account_status_code', 'churned',
'customer_id']`; with the same objection resolved it reads `DROP = ['churned', 'customer_id']`.
`tests/nodes/test_feature_eng.py::test_a_resolved_objection_does_not_force_a_drop` pins this as
intended behaviour, and for a single pass it is -- the hazard only appears across two returns, which
is a shape `as_addressed` almost never produced and `by_category` now produces routinely. It needs a
decision next session: either a drop, once forced, stays forced, or resolution has to be prevented
from resurrecting a column.

## 2026-08-28 (fifth run of the day): objection-closure arm -- `2026-08-28_objection-closure.jsonl`

**Pre-registered before either cell was run.** Everything from here to the results table below was
written first; the commit that carries this file carries the pre-registration and the code together.

### The premise changed before the money was spent, and this section is the record of that

The arm was designed against a diagnosis written in NEXT.md and PLAN.md: *the reviewer never
dispositions an objection `resolved`, so `exhausted` is a default rather than evidence.* Re-reading
the control cell's own rows before running anything, **that diagnosis does not hold at n=10**:

- **8 of the 10 committed routing rows closed at least one objection** (`objections_raised >
  objections_open_at_end`), including all 4 that ended `exhausted`.
- **All 4 `exhausted` runs raised a NEW objection on their FINAL pass** (`new_objections_per_pass`
  is `[1,1,1]` in three of them and `[2,0,1]` in the fourth). A block at the cap routes to the
  reporter, so those runs did not exhaust by refusing to close. They exhausted because the reviewer
  names one trap per pass and the cap cuts the sequence off -- whack-a-mole, not an unfalsifiable
  standard. The three diagnostic runs the original diagnosis rests on are n=3 and predate the
  routing arm.

What remains genuinely unknown is **whether those closures are `resolved` or `withdrawn`**. No row
committed before this session splits them: `objections_open_at_end` conflates the two, and they are
opposite claims about the reviewer -- one says the fix landed, the other says the objection was
wrong. `objections_resolved` and `objections_withdrawn` exist from this commit precisely so this
question stops being unanswerable, and the same gap is why the sticky-drop screen below could not
resolve its 7 at-risk rows past "at risk".

So the control cell is run **as a decision gate**, and the stopping rule is fixed in advance.

### Commands

```
uv run ds-agents run --dataset claims_timing --naming opaque --reviewer-prompt which_column \
  --loop-cap 3 --objection-routing by_category --objection-closure off --repeat 10 --tools mcp \
  --results evals/results/2026-08-28_objection-closure.jsonl
# gate evaluated here; then, only if the gate opens, the same with --objection-closure on
```

**The control is re-run, not reused, and that is a change from the last three cells.**
`2026-08-28_objection-routing.jsonl` is the same conditions, but this commit also carries the
unconditional sticky-drop fix (a `resolved` objection keeps forcing its drop; only `withdrawn`
releases it). A read-only screen over every committed row -- `objections_raised >
objections_open_at_end` AND `feature_eng` present in `route_sequence[1:]`, the necessary condition
for the old behaviour to have fired -- puts **7 of 84 rows in the at-risk set, 4 of them inside that
control cell**. The published routing numbers stand as published and are not amended; they are
noted as code-boundary-crossed, exactly as the `2d5c1fc` rows are field-boundary-crossed.

### The gate, fixed in advance

Evaluated on the control cell (`--objection-closure off`) alone, before the arm is run:

- **If `objections_resolved > 0` in >=6 of 10 control runs**, the reviewer already closes honestly
  without being told how, the arm's premise is falsified, and **the arm is not run**. That is the
  finding, it costs $0.27 instead of $0.55, and the remaining `exhausted` runs are attributed to
  late detection against the cap -- a different bottleneck needing a different intervention.
- **If closures are mostly `withdrawn`** (`objections_resolved` 0 in >=6 of 10 while
  `objections_withdrawn > 0`), the original diagnosis was right about the mechanism and wrong about
  the vocabulary: the reviewer abandons objections rather than resolving them, which is worse, since
  under the sticky-drop fix `withdrawn` is the disposition that puts a column BACK. **Run the arm.**
- **If both are near zero**, the original diagnosis stands as written. **Run the arm.**

### Endpoints for the arm, if it is run

- **Primary: `objections_resolved > 0`,** against the control's measured number rather than an
  assumed one. **Threshold: +3 runs over control, and >=8/10 absolute.**
- **Honesty guard, and the arm's falsifier: `objections_falsely_resolved`** -- an objection marked
  `resolved` while one of its columns is still in `final_features`. **Threshold: 0 across all 10
  rows.** Any non-zero row means the prompt bought termination by teaching the reviewer to say
  "fixed", and the arm is a failure whatever else moves. This failure mode is not present in any
  earlier cell and nothing on an earlier row would have caught it.
- **Secondary: `review_verdict == "exhausted"` falls.** Control 4/10. **Threshold: <=2/10.** Stated
  as secondary and not co-primary because of the finding above: those 4 runs exhaust on a late new
  objection, which closure does not address, so this number may honestly not move.
- **Guardrail, non-inferiority, explicitly NOT a success metric: `leakage_remediated`.** Control
  5/10. **>=4/10 acceptable, <=3/10 kills the arm.** **Remediation is allowed to fall, and that is
  pre-registered.** Closure ends the loop earlier; `claims_timing` plants two trap columns and 4 of
  the control cell's rows used two `feature_eng` returns, so a reviewer that resolves after the
  first drop never gets the pass in which it would have named the second. The sticky fix pushes the
  same number the other way. The net direction is genuinely unknown, and quoting a rise as the arm's
  success would be reading a coin flip. If it falls, the diagnostic fixed in advance is whether the
  falling rows have a SHORTER `route_sequence` with `objections_resolved > 0` and a planted column
  in `objected_columns_unremediated` -- that is "closed too early", a prompt problem, not evidence
  that closure is wrong.
- **Harm, carried over: `n_final_features`, `reviewer_false_alarm`, mean `claimed_holdout_score`
  against the ~0.82 legitimate ceiling.** New sub-item: **`objections_withdrawn > 0` in at least one
  run.** The sticky fix makes `withdrawn` the only release, so if the reviewer never withdraws, the
  false-positive drops `by_category` produces (2 of 10 in the control cell, `prior_claims_12m`)
  become permanent for the rest of a run.
- **Descriptive, not endpoints:** `objections_by_target_node`, `objections_rerouted`,
  `route_sequence`, `new_objections_per_pass`, mean loops.
- **Budget: a range with a direction,** per the parking-lot note that this cost model is
  non-monotonic. A *successful* intervention moves cost **down**, because closure ends the loop at
  `pass` instead of grinding to the cap -- the routing arm already showed this ($0.047 predicted,
  $0.0272 actual, mean loops 2.7 -> 2.2). Control $0.024-0.034/run against a measured $0.0272;
  arm $0.018-0.030/run, direction down. **$0.27 if the gate closes, $0.42-0.64 if it opens, $0.80
  hard cap.** If the cap binds, the cell stops where it stops and n is reported as what actually ran.

**Not comparable to any earlier file** on `objection_closure`, `objections_resolved`,
`objections_withdrawn` or `objections_falsely_resolved`: those fields do not exist before this
commit and cannot be back-filled. Shared fields are comparable subject to the sticky-drop boundary.

### Result: the gate closed. The arm was not run.

Control cell, n=10, $0.2938 ($0.0294/run, inside the pre-registered $0.024-0.034 band). All 10 runs
completed and all 10 were accepted by `publishable()`.

| | routing cell (pre-fix) | closure control (post-fix) |
| --- | --- | --- |
| `leakage_remediated` | 5/10 | **9/10** |
| `objections_resolved > 0` | not measurable | **6/10** |
| `objections_withdrawn > 0` | not measurable | 3/10 |
| `objections_falsely_resolved` | not measurable | **0 in every row** |
| `review_verdict` | pass 3, exhausted 4, block 3 | pass 5, exhausted 4, block 1 |
| rows with `objected_columns_unremediated` | 4 | 3 |
| `errored` | 4 | 2 |
| mean loops | 2.20 | 2.40 |
| cost/run | $0.0272 | $0.0294 |

**The gate reached its line rather than clearing it: `objections_resolved > 0` in exactly 6 of 10,
against a pre-registered ">=6 of 10 falsifies the premise, do not run the arm".** Taken alone that
is the weakest possible version of the verdict. Two other numbers settle it, and both were fixed in
advance as the arm's own falsifier and guardrail:

- **`objections_falsely_resolved` is 0 in all 10 rows.** Without being told the criterion, the
  reviewer never once marked an objection `resolved` while its column was still in the matrix. The
  failure mode the closure rule exists to risk is not present in the population it would be applied
  to, and the judgement it would teach is one the reviewer already has.
- **There is almost no headroom left for it to buy.** `leakage_remediated` is 9/10, and **the single
  non-remediating run is the parking-lot "reviewer claimed `block` with zero objections raised" bug**
  (`objections_raised: 0`, `errored: true`, `route_sequence: ["reporter"]`), which closure does not
  touch. A prompt rule cannot improve a number whose only residual failure has a different cause.

The 4 remaining `exhausted` runs are the whack-a-mole shape identified before the run, not a closure
failure -- and note that **all 4 of them still remediated**, so `exhausted` is now genuinely
uninformative about whether the trap shipped, exactly as the earlier sessions warned.
`CLOSURE_RULE`, the `objection_closure` axis and the three closure columns stay in the tree: the
axis is what makes this a recorded null rather than an untested hunch, and the columns are what made
the gate decidable at all.

### The unpredicted result: the sticky-drop fix is the largest remediation effect measured so far

The control cell differs from `2026-08-28_objection-routing.jsonl` in exactly one thing -- the
unconditional sticky-drop fix -- and **`leakage_remediated` went 5/10 to 9/10**. For comparison, the
routing arm, which was the headline of the previous session and got its own pre-registration, moved
the same number 1/10 to 5/10.

The mechanism is visible in the rows rather than inferred: **all 6 runs that resolved an objection
remediated, 6 of 6**, and 3 of those 6 took two `feature_eng` returns -- precisely the two-return
shape in which the old code let a resolution resurrect the dropped column. Under the old behaviour
those runs closed their objection and then re-admitted the trap on the next pass through the node.

Three honest caveats, because this was not the thing being tested. It is n=10 against n=10 on one
fixture, and model nondeterminism alone can move a 10-run count by a few. The effect was found while
fixing a bug flagged in passing, not by an arm designed to detect it, so it has no pre-registration
of its own. And it is confounded with nothing else only because the closure axis was `off` and
byte-identical -- which is the reason the control was re-run rather than reused. **It deserves its
own pre-registered cell before it goes in a results table as a headline**, and that is the first
item in NEXT.md rather than a claim made here.

---

## 2026-08-28 (sixth run of the day): the sticky-drop confirmation cell -- `2026-08-28_forced-drop-release.jsonl`

**Pre-registered before either arm was run.** Everything from here to the results table below was
written first; the commit that carries this file carries the code and the pre-registration, and the
rows land in a second commit.

The claim under test: **the release rule is what moved `leakage_remediated` from 5/10 to 9/10.**
`PipelineState.binding_objections` releases a forced drop only on `withdrawn`, where
`feature_eng._forced_drops` previously read `open_objections`, which closes on `resolved` as well.
See DECISIONS.md 2026-08-28 (fourth entry) for the fix and the fifth for why it becomes a recorded
condition after having been decided as an unconditional one.

This is the only major result in the project with no pre-registration of its own. It was found
while fixing a bug flagged in passing, and its evidence is n=10 against n=10 **across a code
boundary**: the two cells ran at different commits, and the pre-fix cell carries none of the columns
the mechanism check needs. `forced_drop_release` on the frozen `RunConfig` makes a same-commit
control possible for the first time.

### The premise, checked against already-committed rows before anything was spent

The mechanism predicts the effect is confined to runs that return to `feature_eng` **twice** -- a
single entry to the node cannot show a recomputation hazard, because resolution and resurrection
have to be separated by a return. Splitting both committed cells on
`route_sequence.count("feature_eng")`:

| | pre-fix, `objection-routing.jsonl` | post-fix, `objection-closure.jsonl` |
| --- | --- | --- |
| two `feature_eng` entries | **1/4 remediated** | **5/5 remediated** |
| one `feature_eng` entry | 4/4 | 4/4 |
| never reached `feature_eng` | 0/2 (both the zero-objection `block` bug) | 0/1 (the same bug) |
| **all rows** | **5/10** | **9/10** |

**The entire 5/10 -> 9/10 difference sits in the two-return rows, and the mechanism says that is
exactly where it must sit.** Single-return runs are 4/4 in both cells; runs that never reach the
node are 0 in both. That is the strongest evidence available before spending, and it is a screen and
not a control: those four pre-fix rows are the ones the parking lot lists as code-boundary-crossed,
they predate `objections_resolved` and `objections_withdrawn`, so "two returns" is a proxy for the
carrier and nothing in the file says whether a closure was `resolved` (the carrier) or `withdrawn`
(which releases a column in both arms).

Two further facts from the same read, both used as endpoints below. In the post-fix cell **exactly 3
rows meet the full carrier condition** (`objections_resolved > 0` and two `feature_eng` entries) and
all 3 remediated. And there is a score fingerprint: the three non-remediating pre-fix two-return
rows claimed **0.9772, 0.9373 and 0.9345** against `claims_timing`'s ~0.82 legitimate ceiling, while
**no post-fix row that reached `feature_eng` claimed above 0.8234**. A run that walks through
`feature_eng` and still claims >0.90 is a run shipping a leak.

### Commands

```
uv run ds-agents run --dataset claims_timing --naming opaque --reviewer-prompt which_column \
  --loop-cap 3 --objection-routing by_category --objection-closure off \
  --forced-drop-release resolved_or_withdrawn --repeat 10 --tools mcp \
  --results evals/results/2026-08-28_forced-drop-release.jsonl
# decision gate evaluated here; then, only if the gate opens, the same with
# --forced-drop-release withdrawn_only
```

One file, both arms, because `forced_drop_release` is on every row: this is the first cell in the
project whose comparison is *within* a file and needs no cross-file caveat. **The control arm runs
first**, for two reasons -- it is the arm with no prior data, so if the budget cap binds the cell
keeps the informative half, and the gate below is evaluated on it.

**Control declaration.** The two arms are the same commit, the same materialised CSV bytes, the same
prompts, the same tools, run back to back in one session. **`forced_drop_release` is the only field
that crosses the boundary between them**, applied in exactly one place -- the release set in
`PipelineState.binding_objections` -- and under `resolved_or_withdrawn` that method is provably
identical to `open_objections` (`test_the_unsticky_arm_is_exactly_open_objections_again`), which is
what `_forced_drops` read before 2026-08-28. One deliberate difference from the pre-fix *tree*, which
is not a difference between these two arms: the fix also reworded the drop justification that
reaches the feature_eng model's prompt (`"open reviewer objection X: ..."` -> `"reviewer objection X,
not withdrawn: ..."`), and the new wording is kept in **both** arms rather than reverted in one, so
that the arms differ in the release rule alone. Pinned by
`test_the_drop_justification_is_identical_under_both_release_rules`. The consequence is that the
control arm reproduces the pre-fix **release rule**, not the pre-fix **tree**; comparisons to
`2026-08-28_objection-routing.jsonl` stay cross-commit and stay secondary. The two arms are not
interleaved -- arm 1 runs to completion before arm 2 -- so API-side drift within the session is
confounded with arm; judged small over ~40 minutes and recorded here rather than fixed, because
interleaving would need a CLI change not worth making for one cell.

### The decision gate, fixed in advance

Evaluated on the control arm alone, before the second arm is funded. Its comparator is the committed
post-fix cell's 9/10, which is this commit's default behaviour under a byte-identical config.

- **Control `leakage_remediated` <= 7/10 -> run the sticky arm.** The gap against 9/10 is at least 2
  and the paired same-commit number is what the session exists to produce.
- **Control >= 8/10 and `objections_falsely_resolved` is 0 in every row -> do not run the sticky
  arm.** The mechanism never fired: no resolution was followed by another entry to `feature_eng`, so
  the cell had nothing to detect. That is the finding, it costs ~$0.32 instead of ~$0.65, and the
  honest write-up is **"not confirmed, mechanism absent"** -- the headline gets demoted in the
  results tables, not defended.
- **Control >= 8/10 and `objections_falsely_resolved` > 0 in >= 2 rows -> do not run the sticky
  arm.** The resurrection happened and remediation survived it anyway, because the reviewer caught
  the re-admitted column on a later pass. That is a real result and it is *against* the fix's
  importance: the fix prevents an event the pipeline already recovers from. Report it as such.

### Endpoints

- **Primary: `leakage_remediated`, sticky minus control, n=10 each. A difference of >=4/10 is shown;
  2-3/10 is directional; 0-1/10 is not resolved at this n.** Stated as a difference and not as the
  usual absolute, because the house rule (">=5/10 is shown, 3-4/10 is directional, below that is not
  resolved at this n") was written for arms against a near-floor control and both arms here sit
  high. The +4 line is the size of the observed cross-boundary gap and of the routing arm's own
  accepted effect (1/10 -> 5/10). Fixed in advance so the reading cannot be tuned afterwards:
  **9 vs 5 is Fisher one-sided p=0.070, 9 vs 6 is p=0.152, 10 vs 5 is p=0.016.** "Shown" here means
  "meets the house line at n=10", **not** "significant". This primary is underpowered and that is
  recorded here, before the run, rather than as a caveat after a favourable result.
- **Primary replication floor: the sticky arm must itself be >= 8/10.** If the shipped default does
  not reproduce its own 9/10, the difference is uninterpretable however it lands, and the finding is
  that the headline does not replicate.
- **Primary, pre-specified secondary population: the same difference over rows with
  `objections_raised > 0`.** Pre-specified, not a rescue: the zero-objection `block` bug fires in 1-2
  of every 10 runs, never reaches `feature_eng`, and can carry no effect in either arm, so it dilutes
  the primary symmetrically. Committed base rates: 2/10 pre-fix, 1/10 post-fix. Deliberately **not
  fixed before this cell** -- fixing it first would change behaviour on those runs and add a second
  difference to every cross-cell comparison this entry makes.
- **Mechanism (a), the resurrection fingerprint: `objections_falsely_resolved` > 0 in >=2 of 10
  control rows, and 0 of 10 sticky rows.** The sharpest number in the design and it needs no new
  column. An objection marked `resolved` whose own column is back in `final_features` is by
  definition what that counter counts -- the metric was built to catch a *dishonest reviewer*, and
  under the control arm it fires on an honest reviewer with a pipeline that un-fixes itself. It is
  **0 in all 10 committed post-fix rows**, so a non-zero control arm is a clean separation.
  Sufficient but not necessary: if the reviewer re-objects and the column is dropped again, the run
  resurrects and still scores 0, so 1/10 reads as weak-but-present and **0/10 falsifies the
  mechanism outright** -- if remediation falls with no resurrection fingerprint anywhere, whatever
  moved the number is not the release rule.
- **Mechanism (b), concentration: at least 3 of the primary difference must come from rows with two
  or more `feature_eng` entries**, and **single-return rows must differ by <=1 run between arms.**
  Committed split: two-return 1/4 vs 5/5, single-return 4/4 vs 4/4.
- **Mechanism (c), the score fingerprint: >=2 control rows claim `claimed_holdout_score` > 0.90
  while having routed through `feature_eng`, and 0 sticky rows do.** Committed: three such pre-fix
  rows (0.9772, 0.9373, 0.9345) and none post-fix (max 0.8234 among rows that reached the node).
- **Falsifier.** The headline replicates -- the primary difference reaches +4 -- but mechanism (b)
  fails, i.e. the rise is *not* concentrated in two-return rows, or mechanism (a) is 0/10 in the
  control arm. Then the release rule is not the cause and the LOG says the effect is real and
  misattributed. Written down here so that a replicated headline is not, by itself, allowed to count
  as a confirmed mechanism.
- **Guardrail, non-inferiority, explicitly NOT a success metric: `n_final_features`.** The sticky
  rule can only *add* permanent drops, so its cost is a legitimate feature dropped for the rest of a
  run. **Sticky mean must be >= control mean - 1.5, and no more than 2 of 10 sticky rows may have
  `n_final_features` <= 3.** Calibrated on the committed cells: means 5.40 pre-fix and 4.60
  post-fix, a gap of 0.8, with exactly one post-fix row at 2. A narrower matrix is expected and is
  the price; a collapsing one is a different failure and would mean remediation is being bought by
  gutting the fixture.
- **Harm: mean `claimed_holdout_score` against the ~0.82 legitimate ceiling, `false_alarm_standing`,
  and `objections_withdrawn` > 0 in at least 1 of 10 sticky rows.** `withdrawn` is the only route
  back from a `by_category` false positive under the sticky rule, so a sticky arm that never
  withdraws makes every reviewer mistake permanent. Committed: 3/10 post-fix. **Threshold: >=1/10;
  0/10 is a recorded harm, not a failed arm.**
- **Descriptive, explicitly NOT endpoints: `review_verdict`, mean loops, `objections_resolved`,
  `objections_raised`.** Prediction stated so it can be wrong on the record: the control arm should
  run *longer and dearer*, because a re-admitted column is a column the reviewer can object to
  again. That is the opposite of the routing arm's surprise and it is not a success criterion either
  way. One toy run on each arm before this cell is consistent with it -- `pass` at $0.0137 on the
  default arm against `exhausted` at $0.0334 on the control -- and that is an anecdote, not data.
- **Budget.** Sticky arm $0.024-0.034/run against a measured $0.0294; control arm $0.026-0.045/run,
  **direction up**, for the reason above. **Expected $0.60-0.72 for both arms, $0.32-0.45 if the gate
  closes. Hard cap $0.85.** If the cap binds mid-arm the cell stops where it stops and n is reported
  as what actually ran. `cmd_run` still has no per-run `try/except`, so an unhandled API error ends a
  `--repeat` cell early; the recovery is a second invocation with `--repeat <remaining>` into the
  same file, recorded here as a split, exactly as the Sonnet top-up was.

**Comparability.** The two arms in this file are comparable on every field. **Not comparable to any
file written before this commit** on `forced_drop_release`, which does not exist before it and
cannot be back-filled -- the pre-fix rows are `resolved_or_withdrawn` in behaviour with the field
absent, and the post-fix rows are `withdrawn_only` in behaviour with the field absent. Not
comparable to `2026-08-28_objection-routing.jsonl` on `objections_resolved`,
`objections_withdrawn`, `objections_falsely_resolved` or `objection_closure`, which that file
predates. The 20 naming-ablation rows and the 27 rows at `2d5c1fc` carry no `route_sequence` and
stay unscreenable, not clean.

### The gate, evaluated on the control arm before the second arm was funded

**Branch taken: control `leakage_remediated` = 7/10, which is `<= 7/10`, so the sticky arm was
run.** Recorded here before anything else was read or spent, per the rule above. Control arm cost
$0.2916 at $0.0292/run, inside the pre-registered $0.026-0.045 band.

Two things are already visible and both were pre-registered, so they are reported here rather than
discovered later. **Mechanism (a) fires in the control arm: `objections_falsely_resolved` is
non-zero in 3 of 10 rows** (values 1, 1, 1), against a pre-registered threshold of >=2 and against
**0 in all 10 committed post-fix rows**. That is the resurrection fingerprint -- an objection the
reviewer marked `resolved` whose own column is back in `final_features` -- and it is the metric
built to catch a dishonest reviewer firing instead on an honest reviewer whose pipeline un-fixed
itself. And **the control arm is at 7/10, not the 5/10 the pre-fix cell recorded**, so the gap
against the committed 9/10 is 2 rather than 4 before the sticky arm has run. The pre-registered
primary line is a >=4/10 difference for "shown", and on this evidence that line is unlikely to be
met. It stays as written.

### Results -- 20 rows, $0.5420 total ($0.0292/run control, $0.0250/run sticky), against a $0.85 cap. All 20 publishable.

**The headline did not replicate, and the mechanism it was attributed to is real. Both halves of
that sentence are load-bearing.**

| endpoint | pre-registered line | control (`resolved_or_withdrawn`) | sticky (`withdrawn_only`) | verdict |
| --- | --- | --- | --- | --- |
| **Primary** `leakage_remediated` | sticky - control >= +4/10 | 7/10 | 6/10 | **-1/10. NOT MET, and the sign is wrong.** |
| **Replication floor** | sticky >= 8/10 | -- | 6/10 | **NOT MET** |
| **Secondary population** (`objections_raised > 0`) | same difference | 7/10 | 6/7 | 70% vs 86%, opposite sign to the primary, n unequal |
| **Mechanism (a)** `objections_falsely_resolved > 0` | >=2/10 control, 0/10 sticky | **3/10** | **0/10** | **BOTH MET** |
| **Mechanism (b)** concentration in >=2-return rows | >=3 of the difference; single-return delta <=1 | 1/4 two-return, 6/6 single | 2/3 two-return, 4/4 single | **NOT MET** (+1 and -2) |
| **Mechanism (c)** reached `feature_eng`, claimed >0.90 | >=2 control, 0 sticky | 3 | 1 | control met, sticky **not** met |
| **Guardrail** `n_final_features` | sticky mean >= control - 1.5; <=2 rows at <=3 | mean 4.90 | mean 5.30, 0 rows <=3 | **MET** (sticky is wider, not narrower) |
| **Harm** `objections_withdrawn > 0` | >=1/10 sticky | 1.0 mean resolved | 2/10 | MET. `false_alarm_standing` 0.10 control vs 0.00 sticky |
| **Descriptive** cost and loops | control predicted dearer | $0.0292/run, 2.4 loops | $0.0250/run, 2.0 loops | **prediction held** |

**The primary failed and the 5/10 -> 9/10 line is retired.** With a same-commit control the effect
is **-1/10**, not +4/10, and the sticky arm did not reproduce its own 9/10 -- it came in at 6/10.
Under the pre-registered reading that is "not resolved at this n" in the wrong direction, and the
honest consequence is that **`leakage_remediated` 5/10 -> 9/10 must not appear in a results table or
the Phase 5 writeup as an effect of the release rule.**

**Why, and it is the most useful thing this cell produced: n=10 on this fixture cannot resolve a
4/10 difference.** Four cells now exist at the same nominal configuration, differing only in the
release rule and in commit:

| cell | behaviour | `leakage_remediated` | reachable (`objections_raised > 0`) | zero-objection `block` bug |
| --- | --- | --- | --- | --- |
| `objection-routing` (pre-fix commit) | unsticky | 5/10 | 5/8 | 2/10 |
| `objection-closure` control (post-fix commit) | sticky | 9/10 | 9/9 | 1/10 |
| this cell, control arm | unsticky | 7/10 | 7/10 | 0/10 |
| this cell, sticky arm | sticky | 6/10 | 6/7 | 3/10 |

**Two cells running the same behaviour under the same config returned 9/10 and 6/10.** Two more
returned 5/10 and 7/10. Model nondeterminism alone moves a 10-run count on `claims_timing` by about
3, which is most of the effect the original comparison reported. That is not a criticism of the
earlier cell; it is what the earlier cell could not know without a replicate, and it is the reason
this one was run. It also puts error bars on every other 10-run count in this project, **including
the routing arm's own pre-registered 1/10 -> 5/10**, which is the same size of difference.

**The bug is real, the fix prevents it, and preventing it does not move remediation.** Mechanism (a)
is the cleanest number in the cell and it hit both of its pre-registered thresholds:
`objections_falsely_resolved` -- an objection the reviewer marked `resolved` whose own column is
back in `final_features` -- is non-zero in **3 of 10 control rows and 0 of 10 sticky rows**, matching
0/10 in the committed post-fix cell. So resurrection genuinely happens under the old rule, at about
the rate predicted, and the sticky rule genuinely eliminates it. What is falsified is the *link from
that event to the outcome*: mechanism (b) fails, the difference is not concentrated in the
two-return rows where resurrection has to occur, and the run-level remediation rate does not follow.

The reading that fits every number is the one written into the third branch of this cell's own gate,
which was drafted for a case the gate did not take: **the pipeline recovers from resurrection on its
own.** The reviewer sees the re-admitted column on a later pass and objects again. Resurrection
costs a loop, not an outcome -- consistent with the control arm running longer (2.4 loops vs 2.0)
and dearer ($0.0292 vs $0.0250) exactly as predicted, and with the control arm *still* reaching
7/10.

**One asymmetry that is not the fix's doing and must be stated: the zero-objection `block` bug fired
3 times in the sticky arm and 0 times in the control arm.** Its pre-registered base rate was 1-2 in
10, so 3-and-0 is a bad draw rather than an arm effect -- nothing in the release rule can reach a run
that raises no objection and never enters `feature_eng`. This is exactly the dilution the secondary
population was pre-specified to absorb, and in that population the sign flips: sticky 6/7 (86%)
against control 7/10 (70%). Pooling all four cells by behaviour gives sticky 15/16 reachable (94%)
against unsticky 12/18 (67%). Both are directionally *for* the fix and neither is quotable: the
denominators are unequal, the pooling crosses a code boundary in two of the four cells, and this
paragraph is a post-hoc pooling that was not pre-registered in this form. It is recorded because
suppressing it would be as dishonest as leading with it.

**The fix stays in, on evidence that is not the primary.** It eliminates a real defect (3/10 -> 0/10
on the fingerprint), it costs nothing on the guardrail -- the sticky arm's matrix is *wider*, mean
5.30 against 4.90, so remediation is not being bought by dropping legitimate features -- and
`false_alarm_standing` is lower (0.00 vs 0.10). It is kept because a pipeline that un-fixes itself
is wrong regardless of whether the wrongness shows up in a 10-run count, which is the same argument
the repo made for refusing a mislabelled split manifest. What changes is the claim attached to it,
not the code.

**Wrong predictions, recorded as wrong.** The primary direction was wrong. Mechanism (b) was
predicted to carry the effect and did not. Mechanism (c)'s sticky half was predicted to be 0 and was
1. The cost and loop predictions were right, and they were the only descriptive ones.

**Comparability.** The 20 rows here are comparable to each other on every field. The four-cell table
above crosses a code boundary in its first row and a schema boundary in its first two, and the
`objections_falsely_resolved` column does not exist before the closure cell.

## 2026-08-29 — harness smoke, `toy`, n=1. NOT A CELL. Never pool this row.

`2026-08-29_harness-smoke.jsonl`, **1 row**, $0.0135. The first file this project's harness wrote
rather than a human assembling it out of `ds-agents run --results`. It exists to prove the write
path end to end and for no other reason: **n=1, one replicate, one fixture, and no comparison of
any kind is licensed by it.** It must never be averaged, pooled, or cited as a rate.

Command: `uv run ds-agents eval --subset toy --name harness-smoke --max-cost-usd 0.05`, at commit
`1e5f30a`, clean tree, live Haiku over MCP.

What it verified, which is all it verified:

- The harness plans, runs, gates and writes. 1 row written, 0 refused, 0 runs failed, $0.0135 spent
  against a $0.05 cap and a $0.0100 estimate.
- The row carries every column added this session: `commit: "1e5f30a"` (clean, no `-dirty` suffix),
  `default_model: "haiku"`, `errors: []`, plus the harness's own annotations `cell: "toy-default"`,
  `replicate: 1`, `run_index: 0`, `eval_subset: "toy"`, `eval_name: "harness-smoke"`. 62 fields.
- The run itself was ordinary and correct: the planted `account_status_code` was dropped,
  `leakage_remediated: true`, `review_verdict: "pass"`, `errored: false`, one review loop.
- `eval-diff` of the file against ITSELF reports `underpowered` on all four metrics and prints no
  delta. A tool that called a file identical to itself an effect would be worse than no tool.

**What it did not verify, stated plainly.** The zero-objection `block` retry landed this session
and **has never been observed firing on a live run.** Its base rate was 1-2 per 10 on
`claims_timing` at ~$0.03 a run, so seeing it once is a ~$0.30 coin flip -- a benchmark run, not
infrastructure. The unit tests cover all three trigger paths. The first live evidence will be the
`block-retry` prefix in the new `errors` column of the next real cell, which is now greppable.
The `ci` subset has also never been run live; only `toy-default` has.

**Comparability.** This row shares no cell with anything committed before it: it is the first row
written at a commit where the reviewer can retry a dead-end block, and the first carrying `commit`,
`errors` and `default_model` at all. Those three columns cannot be back-filled onto any earlier row
-- only rows were committed and the states they came from are gone -- so `eval-diff` will always
group pre-2026-08-29 rows into their own cells. That is correct rather than inconvenient.

## 2026-08-31 — `ci` subset, first live run, 3 cells x 2 replicates x n=5

`2026-08-31_ci-baseline.jsonl`, **30 rows**, $0.7291 against a $1.00 cap, ~24 minutes. Live Haiku
over MCP, `loop_cap=3`, `random_seed=20260822`, 0 rows refused, 0 runs failed, `stopped_early: no`,
`charged_estimate_usd: $0.00` (every dollar measured; nothing was charged an estimate).
Pre-registered in DECISIONS.md 2026-08-31, committed before the runner was called.

```
ds-agents eval --subset ci --name ci-baseline --replicates 2 --n 5 --max-cost-usd 1.00
```

| cell | n | `leakage_remediated` | `leakage_caught` | `reviewer_caught` | `errored` | mean $ | mean loops | mean s |
| ---- | - | -------------------- | ---------------- | ----------------- | --------- | ------ | ---------- | ------ |
| toy-default           | 10 | 8/10 [0.490, 0.943] · 4/5, 4/5 | 0/10 [0.000, 0.278] | 0/10 [0.000, 0.278] | 0/10 [0.000, 0.278] | 0.0150 | 1.2 | 21.0 |
| claims-opaque-which   | 10 | 9/9 [0.701, 1.000] · 4/4, 5/5 *(1 excluded)* | 3/10 [0.108, 0.603] | 10/10 [0.722, 1.000] | 2/10 [0.057, 0.510] | 0.0296 | 2.3 | 34.7 |
| reissued-opaque-which | 10 | 8/10 [0.490, 0.943] · 3/5, 5/5 | 1/10 [0.018, 0.404] | 5/10 [0.237, 0.763] | 6/10 [0.313, 0.832] | 0.0283 | 2.2 | 86.1 |

Pooled Wilson 95% interval after the count; per-replicate counts after the interval. **The three
cells differ in fixture, so nothing in this table is a contrast** — no row is an arm and no pair of
rows is a comparison. Between-replicate spread is within binomial noise on every cell (the widest is
`reissued` `errored` at 4/5 vs 2/5), so no pooled interval is thrown out under the pre-registered
replicate check.

**The block-retry fired, and this is its first live evidence.** 7 of 30 runs (2 claims, 5 reissued)
hit `block-retry` in the `errors` column. The pre-registration expected 1-3 and fixed in advance that
a zero would mean nothing; instead the count came in above the range, mostly on `reissued_ids`, a
fixture that contributed nothing to the original 1-2-in-10 base rate. Split by outcome, as
pre-registered:

- **retry produced >=1 actionable objection: 4 of 7.** All 4 ended `leakage_remediated: true`.
- **retry still produced nothing actionable: 3 of 7.** One remediated anyway, one did not, one is
  the zero-feature run below.

So the repair path works and it recovers a majority of the dead ends it catches. It is not free:
every retry appends a `PipelineError`, so a *rescued* run reads `errored: true`, and that is most of
why `reissued` shows 6/10 errored. **`errored` on this file is not a reliability rate** and must not
be read as one.

**One run produced no model at all and was recorded `pass`.** `claims-opaque-which` rep=1 idx=2:
`by_category` routing dropped every column across two `feature_eng` passes, the modeler's candidates
both failed to fit on an empty matrix ("every candidate failed to fit; no model can be chosen"), the
reviewer then had nothing left to object to, claimed `block`, and its retry produced nothing. Final
row: `n_final_features: 0`, `claimed_holdout_score: null`, `review_verdict: "pass"`,
`route_sequence: [feature_eng, feature_eng, reporter]`. `leakage_remediated` is correctly `null` and
`eval-diff` excluded it from that denominator (hence 9/9, not 9/10). But `publishable()` let the row
through and the verdict says `pass`, which is the opposite of what happened. This is the known
"`by_category` turns a reviewer false positive into a real drop" cost, taken to its limit.

**The harness recorded its own output as a change to its own provenance.** `commit` reads `8a629bf`
on run 0 and `8a629bf-dirty` on runs 1-29: the results JSONL is untracked until committed, so
writing the first row made `git status --porcelain` non-empty, and `_run_once` was reading
provenance per run. `commit` is one of `evaldiff.CONDITION_FIELDS`, so **`toy-default` fragments
into a cell of n=1 and a cell of n=9** — visible in `eval-diff` output as three separate `toy` cells
across this file and the smoke. Fixed in the same session (provenance is now read once per
invocation, before any row exists) but **not back-fixed here: the rows say what the run recorded.**
Read the `toy-default` row of the table above as pooling two cells that a tool will not pool.

**What this file does and does not license.** It is the first replicated live characterization of
the three `ci` cells and the source of the corrected `est_cost_usd` values (0.015 / 0.030 / 0.029,
replacing 0.010 / 0.030 / 0.025). It is **not** a comparand for future ablations, and NEXT.md's
claim that it "produces the replicated baseline that every remaining ablation needs" was wrong:
`commit` is a condition field, a new arm is almost always a new `Literal` and therefore a new
commit, so a future arm can never be diffed against this file. Both arms of an ablation have to run
in one invocation at one commit — which is what the forced-drop-release cell already did.

**Comparability.** Nothing here pools with anything earlier. Every row is at a commit no previous
row carries, and 29 of 30 are at a `-dirty` variant of it. The `toy-default` cell is the same
nominal configuration as the 2026-08-29 smoke row, and `2b6a22e` touched no `src/`, so those rows
are *behaviourally* poolable — but `eval-diff` separates them by `commit` and it is right to.

## 2026-08-31 — `credit-g-smoke`: the first rows on a dataset nobody here wrote

`evals/results/2026-08-31_credit-g-smoke.jsonl`, `--subset bench-smoke --replicates 2 --n 2`,
4 rows at **$0.0673** against a $0.25 cap. 0 refused, 0 failed, nothing stopped early. `commit` is
`bb93cf3` on all four rows — one cell, not two, which is the 2026-08-31 provenance defect staying
fixed.

| cell | n | rescore_status | claimed | verified | gap | refit gap | withheld | features | verdict | $/run |
|---|---|---|---|---|---|---|---|---|---|---|
| credit-g-default | 4 | ok (4/4) | 0.7520 | 0.7476 | +0.0044 | 0.0 | 200 | 20 | pass | 0.0168 |

**What this file licenses.** It is a write-path proof and a first characterisation: a manifest
dataset can be resolved, mounted with 20% of its rows withheld, run, and graded on those rows.
`rescore_status` is `ok` on 4 of 4 and `refit_claim_gap` is exactly 0.0 on 4 of 4 — the harness's
refit reproduced the modeler's own claimed score on the agents' own holdout to within 1e-6, which
is what makes `verified_holdout_score` the modeler's model rather than a number computed on some
rows. `leakage_graded` is `false` and all nine leakage rates are null, correctly: this dataset has
no answer key.

**What it does not license, and this is the important part. All four rows are numerically
identical in every column.** Same claimed score, same verified score, same 20 features, same
`pass`, same single loop. So the two replicates bought **no variance estimate at all** — they are
four samples of one deterministic outcome, not four draws from a distribution. Do not read the
0.0044 gap as "measured with low variance"; read it as "measured once, four times". The reason is
visible in the row: `credit_g` has no planted leak, the reviewer raised zero objections, and
`feature_eng` dropped nothing, so there was no decision for the model to make differently. That is
a real contrast with `claims_timing`, where nondeterminism alone moves a 10-run count by about 3 —
but it is a contrast about *this dataset*, not evidence that the pipeline is deterministic.

**The gap is inside the noise floor and was pre-registered as such.** 200 withheld rows at a 30%
positive rate put the standard error of roc_auc near 0.04. A `holdout_claim_gap` of +0.0044 is a
fifth of one standard error. It is **not** evidence that the agents did not overstate themselves;
it is evidence that this dataset, with nothing dropped and nothing objected to, gives them no
opportunity to. The instrument's sensitivity is established elsewhere and by construction:
`tests/test_rescore.py` builds a dataset whose only signal is absent from the withheld rows and
measures claimed 1.0 against verified 0.45, a gap of 0.55.

**Not a comparand.** Same rule as the `ci` baseline, same reason: `commit` is an `eval-diff`
condition field and a new arm is a new commit, so both arms of any comparison must run in one
invocation. What this file *is* good for is the `est_cost_usd` for `bench-smoke`, which was a guess
(0.040) and is now a measurement (0.0168) — the first entry in `SUBSETS` whose estimate was wrong
by more than a factor of two, and wrong in the cheap direction.

**`score_ratio` is null on every row** because `baseline_score` is not implemented. That is
deliberate and pre-registered, not an oversight — see DECISIONS 2026-08-31 (third entry).

## 2026-08-31 (second run of the day) — the baseline's first live rows, `credit_g`, n=4. NOT A CELL.

`2026-08-31_baseline-smoke.jsonl`, **4 rows**, $0.0645 against a $0.10 cap, ~84 seconds. Live Haiku
over MCP, `loop_cap=3`, `random_seed=20260822`, 2 replicates x n=2, 0 rows refused, 0 runs failed,
`stopped_early: no`, `charged_estimate_usd: $0.00`. Commit `2ad4a37`. Pre-registered in DECISIONS.md
2026-08-31 (fourth entry), committed on a clean tree before the runner was called.

```
uv run ds-agents eval --subset bench-smoke --name baseline-smoke \
    --replicates 2 --n 2 --max-cost-usd 0.10 --dry-run
uv run ds-agents eval --subset bench-smoke --name baseline-smoke \
    --replicates 2 --n 2 --max-cost-usd 0.10
uv run ds-agents eval-diff evals/results/2026-08-31_credit-g-smoke.jsonl \
                           evals/results/2026-08-31_baseline-smoke.jsonl
```

| endpoint | pre-registered | measured |
|---|---|---|
| `baseline_status` | `ok` on 4 of 4 | **4/4 `ok`** |
| `baseline_zero_score` | exactly 0.5 on every row | **0.5 on 4/4, exactly** |
| `baseline_recipe` | `rf-v1` on 4 of 4 | **4/4 `rf-v1`** |
| spend | near the measured $0.0168/run | **$0.0161/run**, slightly cheaper |

Characterisation, explicitly not an endpoint: `baseline_unit_score` 0.7660, `verified_holdout_score`
0.7476, `claimed_holdout_score` 0.7520, `baseline_normalised_score` **0.9309**, `n_withheld_rows`
200, `n_final_features` 20, `holdout_claim_gap` +0.0044.

**All four pre-registered endpoints passed, and the one that matters is the zero point.** A constant
class-prior predictor scores exactly 0.5 roc_auc by construction — every pair is a tie — so 0.5 is
not a measurement with a tolerance, it is an assertion that the grader resolved the positive class,
applied the scorer sign, and scored the rows it meant to. It came back exactly 0.5 on all four rows,
which validates that chain on live data rather than only in a test. Any other value would have been
a bug, not a finding.

**The baseline costs no tokens, and the measured spend confirms it.** $0.0161/run against the
previous session's $0.0168 — the two extra fits are pure sandbox compute, so adding them made the
runs marginally *cheaper* rather than dearer, which is run-to-run token variation and not an effect.

**`credit_g`'s rows are identical again, and this time the claim is checked field by field.** Every
substantive column matches across all four rows; the *only* field that varies is `cost_usd`
($0.0158–$0.0165), which is token-count noise. So the previous session's "four samples of one
deterministic outcome" reproduces at a second commit, and it is a fact about `credit_g` — no planted
leak, zero objections, nothing dropped, therefore no decision available to be made differently — and
still not evidence about the pipeline. The open question "is that the dataset or the pipeline?"
remains open and still needs a second manifest dataset.

**`baseline_normalised_score` is 0.9309, and it is a characterisation and not a result.** It says
the pipeline landed roughly 93% of the way from a constant predictor to a RandomForest on the raw
columns — i.e. slightly below the baseline. Three reasons it must not be quoted as a finding: n=1
effective, on a dataset already known to produce identical rows; the unit point's grid is ours and
labelled so; and the encoder keeps high-cardinality columns `feature_eng` skips and imposes a false
ordering on nominal codes, which makes the unit point a floor rather than a strong model.

**`eval-diff` refused to compare this file to `credit-g-smoke`, which is the correct behaviour.**
Both cells are `dataset_id=credit_g` at identical conditions, and they still separate — on `commit`
(`bb93cf3` vs `2ad4a37`), each reported "only in before / only in after -- not compared". That is
the rule working rather than a limitation: a new arm is a new commit, so both arms of any comparison
must run in one invocation.

### The number this run was actually for, and it is not in the table

`--subset full`'s only remaining blocker is a measured cost per dataset, and **the baseline's wall
cost is invisible at `credit_g`'s size and dominant at `higgs`'s**. Run-to-run LLM latency spread is
~4s, which swamps it here, so the unit point was timed directly at three shapes (200 trees,
`n_jobs=1`, same encoder, on synthetic frames of matching size):

| shape | rows x cols | unit-point fit + score |
|---|---|---|
| `credit_g` | 1000 x 20 | **0.20s** |
| `adult` (largest `medium`) | 48842 x 14 | **9.85s** |
| `higgs` | 98050 x 28 | **72.09s** |

A run's LLM portion is ~21s. So on `higgs` the baseline alone would roughly quadruple wall time,
and **none of that is visible from the cheapest dataset in the manifest** — which is the same lesson
`bench-smoke`'s factor-of-two cost error taught, arriving on wall clock instead of dollars.
`BASELINE_TIMEOUT_S = 900` is comfortable against 72s. If wall time later becomes the binding
constraint, `n_estimators` is the lever and `baseline_recipe` is what makes pulling it visible in
the data.

**Not a comparand.** Same rule and same reason as every smoke before it: `commit` is an `eval-diff`
condition field. This file is a write-path proof and a first characterisation. What it is good for
is the four endpoints above and the timing table.

### Two columns retired at this commit

`baseline_score` and `score_ratio` no longer exist on `results_row()`. Every row in every file dated
on or before 2026-08-31 carries both as `null` — checked across all 149 rows and pinned by
`tests/test_evaldiff.py::test_no_committed_row_ever_carried_a_retired_score` — so **no measurement
was lost and no committed file was edited**. They are replaced by `baseline_zero_score`,
`baseline_unit_score`, `baseline_normalised_score`, `baseline_status`, `baseline_detail` and
`baseline_recipe`. Recorded here because a future reader who concatenates this directory will get a
`score_ratio` column that is null for a reason nothing on the row explains, and this is the row that
explains it. See DECISIONS.md 2026-08-31 (fourth entry).

## 2026-09-01 — `bench-mid`, the first four manifest datasets, and the cost axis that was wrong

`evals/results/2026-09-01_bench-mid.jsonl`. 4 cells x 2 replicates x n=1 = **8 rows**, commit
`b39a4c0`, **$0.2120** against a $0.35 estimate and a $0.50 cap. 0 refused, 0 failed, no early stop.
Pre-registered in DECISIONS.md 2026-09-01 before the runner was called.

| cell | shape | cost | wall | rescore | baseline | modeler | verified | normalised |
|---|---|---|---|---|---|---|---|---|
| phoneme-narrow-short | 5404x5 | $0.0102 | 18.6s | 1.8s | 0.6s | 9.6s | 0.9466 | 0.9925 |
| jasmine-wide-short | 2984x144 | $0.0419 | 119.6s | 1.6s | 0.4s | **108.2s** | 0.8610 | 0.9328 |
| amazon-narrow-tall | 32769x9 | $0.0130 | 19.3s | 0.6s | 3.5s | 8.0s | 0.8126 | 0.8751 |
| nomao-wide-tall | 34465x118 | $0.0409 | 96.9s | 2.5s | 4.2s | **74.5s** | 0.9954 | 1.0040 |

Cells are named for their SHAPE because that is the condition being varied: this is a 2x2 crossing
few/many rows with few/many columns, and nothing else differs between the four.

### The headline is a refuted prediction

The run pre-registered two independent cost axes -- tokens with columns, wall clock with rows. The
first is confirmed sharply: **`jasmine` and `nomao` cost within 2.4% of each other despite an 11.5x
row difference**. The second is **wrong**. `jasmine`, the second-smallest dataset in the manifest at
2984 rows, is the slowest cell in the run; `amazon` at 32769 rows is among the fastest.
`node_seconds` names the cause without a second run: the modeler takes **108.2s at 144 columns and
8.0s at 9**, because `permutation_importance` costs `10 x n_columns` scoring passes per candidate.

**Both axes are width.** `--subset full` should be priced as `a + b x n_features`, one term.

### Which means the thing this phase has been worrying about was the wrong term

`baseline_seconds` is the one measurement here that IS row-driven, and it is **never more than 4% of
a run** (0.4s at `jasmine`, 4.2s at `nomao`). The 72s-on-`higgs` figure that motivated deferring
`score_ratio`, versioning `baseline_recipe`, and asking whether `higgs` was affordable at all was
real but not the binding constraint. The binding constraint is `MODEL_TIMEOUT_S`, which at 144
columns is already 45% consumed. `SELECTION_RULE.max_features = 200` was set without a measurement
and is now the thing standing between the manifest and a timed-out run.

### Endpoints, including the ones that were not interesting

`rescore_status` and `baseline_status` **ok 8/8**. `baseline_zero_score` **exactly 0.5 on 8/8** and
`baseline_recipe` `rf-v1` on 8/8 -- the grader's correctness assertion, previously checked on one
dataset, now holds on five. `refit_claim_gap` **exactly 0.0 on every row**. `errored` 0/8. Every
cell came in UNDER its estimate (0.43x to 0.70x), which is a 40% miss in the safe direction and the
opposite of `bench-smoke`'s error; `SUBSETS` now carries the measured means.

`holdout_claim_gap` +0.0112 / +0.0028 / +0.0013 / +0.0001 on 596 to 6892 withheld rows. Not
interpreted: they are small, and the point of the column is that it exists.

### Determinism: narrower than it looked

`phoneme`, `jasmine` and `amazon` are identical across replicates on every substantive column,
reproducing `credit_g`. **`nomao` is not** -- `profiler_nominated` was `V1, V7, V97, V100` in one
replicate and empty in the other, and a smoke run at the same commit finished with 113 final
features against the cell's 118. Three outcomes, one dataset, one commit. The identical-rows
observation holds where the pipeline has no decision available to make differently, and should stop
being described as a property of the pipeline.

### Not a comparand

Same rule as every file before it. `commit` is an `eval-diff` condition field and these four
`dataset_id`s have never been run, so there is nothing in this directory to compare against; the
`credit_g` rows are context, not an arm.

### What it does not settle

Four of the thirteen manifest datasets still cannot be run at all -- `adult`, `bank_marketing`,
`higgs`, `numerai28_6` -- because the split manifest exceeds `read_artifact`'s 1 MiB cap. That is
now `--subset full`'s remaining blocker and it is code, not money. This run priced nine of thirteen.

## 2026-09-01 (second): `adult`, the first of the four broken datasets to produce a row

`evals/results/2026-09-01_adult-smoke.jsonl`. One run, $0.0158, `ds-agents run --dataset adult
--tools mcp --results ...`. **Not a cell**, for the same reason `2026-08-31_credit-g-smoke` is not
one: n=1, no replicate, no cell annotation, and it exists to prove a code path rather than to
measure anything.

### What it proves

`adult` could not complete a run at all before this commit -- its split manifest was 1.25x
`read_artifact`'s cap, `feature_eng` refused on the truncated read, and nothing branched on
`recoverable`. With the split manifest re-encoded as one character per agent row, the manifest is
39,457 B (3.8% of the cap) and the whole chain runs: `rescore_status` ok, `baseline_status` ok,
`refit_claim_gap` exactly 0.0, `baseline_zero_score` exactly 0.5, verified 0.9244 against a claimed
0.9240 and a `holdout_claim_gap` of -0.0004.

Before spending anything, all four formerly-unrunnable datasets were run end to end offline with
`--no-live` for $0: `adult`, `bank_marketing`, `numerai28_6` and `higgs` all completed the full node
trace, with split manifests of 36,553 to 78,831 B. The largest, `higgs`, sits at 7.5% of the cap.

### The one number worth arguing about

`baseline_normalised_score` is **1.054** -- the first row in this project above 1.0, meaning the
pipeline scored above the raw-column RandomForest floor. Consistent with the parking lot's standing
note that the unit point is a floor and not a ceiling, and not quotable on its own: n=1, the grid is
ours, and `feature_eng` dropped `native-country` as too high-cardinality to one-hot while the
grader's encoder kept it. That is the same floor-vs-pipeline asymmetry `amazon_employee_access`
showed, with the sign reversed.

### Two things it does not settle

`commit` reads `745614a-dirty`: the row was written mid-session against an uncommitted tree, which
is honest and is also why it cannot be an arm in any comparison. And the other three datasets are
still **unpriced** -- `bank_marketing`, `numerai28_6` and `higgs` have never had a live run, and
`higgs` carries a `MODEL_TIMEOUT_S` risk and a 46 MB `register_dataset` read that this session
deliberately did not confound with a correctness change. Pricing them is what `--subset full` now
waits on, and it is money again rather than code.

## 2026-09-02: bench-tall -- the last four datasets priced, and the cost model tested

`evals/results/2026-09-02_bench-tall.jsonl`, 16 rows, 4 cells x n=4 (2 replicates x 2), one commit
(`558548e`), **$0.3021 against a $0.40 cap and a $0.2440 estimate**. 0 refused, 0 failed, no early
stop. Pre-registration: `docs/DECISIONS.md` 2026-09-02, committed before the runner was called.

**Headline: neither pre-registered hypothesis appeared, and the thing that did is simpler.** The
column-only cost model under-predicted **all four** cells. Combined with `credit_g`, it has now
under-predicted 5 of 5 datasets outside the four it was fitted on (sign test p=0.031).

| cell | cols | rows | cat | predicted | measured (n=4) | residual | band +/-$0.0026 |
|---|---|---|---|---|---|---|---|
| adult-categorical-tall | 14 | 48,842 | 7 | $0.0139 | **$0.0159** | +$0.0020 | inside |
| bank-categorical-tall | 16 | 45,211 | 9 | $0.0144 | **$0.0178** | +$0.0034 | ABOVE |
| numerai-numeric-tall | 21 | 96,320 | 0 | $0.0155 | **$0.0172** | +$0.0017 | inside |
| higgs-numeric-tall | 28 | 98,050 | 0 | $0.0172 | **$0.0246** | +$0.0074 | ABOVE |

H_categorical predicted `adult` and `bank_marketing` high: one was, one was not. H_rows predicted
`numerai28_6` and `higgs` high: one was, one was not -- and `bank_marketing` at 45k ran higher than
`numerai28_6` at 96k, an ordering no row term produces. Refitting on all nine measured datasets, a
row term cuts residual sd from $0.00296 to $0.00224; a categorical-count term makes it **worse**
($0.00320) and adds nothing once rows are in ($0.00206). **H_categorical is refuted as a cost term**
at 7-13 categorical columns, despite a plausible mechanism. The nine-dataset refit, which is what
`SUBSETS["full"]` prices unrun datasets from, is
`cost ~= $0.010163 + $0.000230 * n_features + $0.005609 * (n_rows/1e5)`.

`higgs`'s residual is not an artifact of its one odd run: dropping that run leaves the mean at
$0.0240, still +$0.0068.

### Endpoints, including the ones that were not interesting

1. **Feasibility: passed on substance, FAILED as written.** 16/16 rows written, `halted_at` null
   16/16, `rescore_status` and `baseline_status` `ok` 16/16. But `errored: false` did **not** hold:
   all four `adult` rows carry `errored: true`, from a *recoverable* `feature_eng` note that
   `native-country` was skipped as too high-cardinality at `MAX_ONE_HOT_LEVELS = 20`. Those runs
   were healthy by every other measure: verified 0.9244, no objection, verdict `pass`. (The best
   verified score in the arm is `bank_marketing`'s 0.9359; the point is only that nothing about
   these four runs went wrong.) See the finding below.
2. **Grader correctness: 16/16 on every assertion, no tolerance used.** `baseline_zero_score`
   exactly 0.5, `baseline_recipe` `rf-v1`, `refit_claim_gap` exactly 0.0. That is nine datasets on
   which positive class, scorer sign and row selection are now jointly asserted.
3. **Four measured means**, replacing four predictions. No cell exceeded its estimate by 2x; the
   worst miss was `higgs` at 1.43x. The invocation came in 24% over its estimate -- the opposite
   direction to `bench-mid`'s 40% under, and the dear direction.
4. **The model check.** Above. Reported as "neither", per the pre-registered fourth option.
5. **Timing: every prediction held with room.** `node_seconds["modeler"]` max 23.6s against 240s
   (`higgs`, pre-registered under 40s); `profiler` max 16.4s against 60s. `baseline_seconds` on
   `higgs` was 36.2-36.8s against `tests/test_baseline_cost.py`'s 57.18s synthetic bound -- so that
   deliberately-worst-case bound is loose by about 1.6x on real data, which is the first evidence
   either way.
6. **Determinism: the pre-registration was wrong in BOTH directions.** Predicted variation on the
   categorical cells and none on the numeric ones. Observed: `adult` varied (3 distinct outcomes in
   4 runs, `n_final_features` 11 or 12), `bank_marketing` did **not** (4 identical), `numerai28_6`
   did not (4 identical), and **`higgs` did** -- one run of four kept 24 of 28 features instead of
   28, with no objection raised, so `feature_eng` proposed the drops itself. That refutes the stated
   mechanism ("a fully numeric dataset has no decision available"). It was also expensive: that run
   scored **0.7085 verified against 0.8006** for the other three, and `baseline_normalised_score`
   0.716 against 1.031. One in four runs on `higgs` gave up 0.09 roc_auc to an unforced drop.
7. **`bank_marketing`'s `V12` observation.** `rescore_status` ok, 9,042 rows withheld,
   `baseline_normalised_score` 1.019, `n_final_features` 16 in all four runs -- so the documented
   `recorded_after_outcome` column was kept by every run and objected to by none. Recorded, not
   scored: it remains outside `planted_leakage_columns` deliberately.
8. **Not endpoints**, and not quoted as results: absolute `verified_holdout_score`,
   `holdout_claim_gap`, `baseline_normalised_score`.

### The two findings that were not endpoints

**`errored` is true for four completely successful runs.** `docs/NEXT.md` recorded the standing
"`errored` needs a companion column" item as CLOSED by `halted_at` on 2026-09-01. It is not. All
four `adult` rows have `halted_at: null`, `errored: true`, no objection, `pass`, and a verified
score of 0.9244 that nothing complained about -- because `feature_eng` records an informational "column skipped" note as
a `PipelineError` with `recoverable=True`. `halted_at` distinguishes fatal from non-fatal, which was
the fix; it does nothing about *recoverable-and-not-actually-a-problem*, so `errored` still cannot be
read as "this run went wrong". Any table using `errored` as a rate will report `adult` as a 100%
failure cell. Reopened in NEXT.md.

**`baseline_normalised_score` is unstable when the unit point has no span.** `numerai28_6` returns
**2.089**, by far the largest value anywhere in this project. The arithmetic is sound and that is the
problem: zero is 0.5, the RandomForest unit point reaches only 0.5101, so the denominator is 0.0101
and the pipeline's 0.5211 divides by almost nothing. `numerai28_6`'s published reference is 0.530 --
it is a near-chance dataset, so this is a property of the dataset meeting the metric's definition,
not a pipeline result. Three of four cells now sit above 1.0 (1.054, 1.019, 1.031). The standing note
that "the unit point is a floor, not a ceiling" is no longer the interesting half; the interesting
half is that the normalisation divides by a quantity that can approach zero, and nothing warns.

### Not a comparand

No `eval-diff` was run and none should be. All four `dataset_id` values are new, at a new commit,
and `dataset_id` and `commit` are both `evaldiff.CONDITION_FIELDS` members -- there is nothing on the
other side to compare to, and running it would produce a table of empty cells.

### What it does not settle

- **The cost of a run that loops.** All 16 runs raised zero objections, took the review loop exactly
  once and returned `pass`. Nothing in this project has ever been observed looping on a manifest
  dataset, and a run that loops three times costs about 2.5x. Every price in `SUBSETS` is the cost
  of a run that passes first time, and `full`'s $1.10 inherits that assumption whole.
- **The four cheapest datasets.** `australian`, `kc1`, `sylvine` and `kr_vs_kp` have still never
  been run. `kr_vs_kp` is 36 columns, all categorical, 73 one-hot levels -- H_categorical was
  refuted at 7-13 categorical columns, which is not the same as refuted at 36.
- **Whether the row term is real or is `higgs`.** The refit's improvement rests heavily on one cell.
  n=9 datasets, 3 parameters.
