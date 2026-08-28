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
