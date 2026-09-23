# ds-agents

How much of a data scientist's tabular workflow can a team of agents do, and how do you tell when
they got it wrong?

Six LangGraph nodes (intake, profiler, feature engineering, modeler, adversarial reviewer, reporter)
run on 13 public binary-classification datasets and 3 synthetic datasets with planted leakage. The
harness withholds 20% of rows before the pipeline starts and scores them itself, so the agents'
claimed score is never trusted. The default model is Claude Haiku throughout.

Full tables, caveats and method are in [docs/RESULTS.md](docs/RESULTS.md).

## Findings

1. **Model quality is close to a tuned RandomForest, at 1 to 4 cents per dataset.** Across the 13
   benchmark datasets the pipeline reaches 0.87 to 1.05 of a RandomForest baseline it never sees,
   and beats it on 2. Its self-reported scores run slightly optimistic (mean gap +0.005).

2. **The leakage detector reads column names.** With descriptive names (`days_to_close`) the
   profiler caught 19 of 20 planted leaks. With the same columns renamed `var_07` and nothing else
   changed, it caught 2 of 20, and the pipeline reported a leaked score of 0.974 with no warning.
   The opaque-arm number has since drifted up (8 of 20 on a later run), so the exact rate needs a
   re-run, but the gap is large either way.

3. **Catching a leak and fixing it are separate failures.** The default reviewer named the leak in
   1 of 10 runs. One added prompt rule raised that to 9 of 10, but the leak still shipped in 9 of
   10, because the reviewer sent its objection to the modeler, which cannot drop columns. Routing
   column objections to feature engineering fixed it (9 of 10 remediated, reproduced three weeks
   later). The reviewer prompt mattered more than the reviewer model: Sonnet cost 5x and found
   nothing under the default prompt.

4. **On real data the default reviewer rarely objects.** 12 objections in 52 runs, on 3 of 13
   datasets. This data can't tell "nothing to object to" from "not looking". A candidate
   explanation is that the reviewer only objects when feature importance is concentrated in 1 to 3
   columns. That is a hypothesis, not a result yet.

5. **The cost model held; an unpriced assumption didn't.** The full benchmark run came in 10% over
   estimate ($1.22 vs $1.10). Every run that passed review first time was within 5% of its
   estimate. The whole overrun came from runs that looped back through the reviewer, which cost
   about 1.8x to 2.5x.

## Main caveats

- Fixture results use n=10 per arm and 200-row holdouts, so small differences are not effects.
- The leakage rows were written by older code and don't record their commit.
- Three planted traps are not a taxonomy, and a random holdout split can't catch temporal leaks.

## Running it

```console
uv sync
uv run pytest                              # full suite, includes the toy end-to-end run
uv run ds-agents run --dataset toy         # one pipeline run, prints the node trace
uv run ds-agents eval --subset ci          # leakage fixtures, ~$0.73
uv run ds-agents eval --subset full        # all 13 benchmark datasets, ~$1.22
uv run ds-agents eval-diff A.jsonl B.jsonl # compare two results files
```

Without an API key the pipeline uses a stub model and refuses to write result rows.

`docs/ARCHITECTURE.md` is the contract, `docs/DECISIONS.md` the log of why, and
`evals/results/LOG.md` a narrative of every funded run.
