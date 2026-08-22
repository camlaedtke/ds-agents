# ds-agents

How much of a data scientist's tabular workflow can you delegate to a team of agents, and how do
you know when they got it wrong?

A LangGraph pipeline (intake, profiling, feature engineering, modeling, adversarial review,
reporting) that runs against a benchmark of public tabular datasets with published baselines, plus
planted leakage traps. Tools are exposed through a custom MCP server. Results and ablations live in
`evals/results/`.

Work in progress. See `docs/PLAN.md`.
