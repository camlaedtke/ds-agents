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
