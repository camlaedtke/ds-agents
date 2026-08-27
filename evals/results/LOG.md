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
