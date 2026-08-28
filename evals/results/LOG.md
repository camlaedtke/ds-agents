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
