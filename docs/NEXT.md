# Next session

## Start here
**The manifest landed, and the question that blocked it since session 0 is answered.**
`evals/datasets/manifest.yaml` holds **13 binary-classification datasets** from the AutoML
Benchmark (Gijsbers et al., JMLR 25(101) 2024; OpenML suite 271), each pinned by OpenML data id +
task id + upstream md5, each carrying a `published_reference` citable to a stable OpenML run id.
AMLB was chosen because it is the only candidate that publishes *baseline framework definitions*
rather than only a dataset list. Cost: $0 in API spend to build it.

**The manifest is generated, and that is enforced rather than promised.** `.claude/settings.json`
denies Edit and Write under `evals/datasets/`, so `ds-agents datasets refresh` is the only writer.
Every number in the file is an API response field or a measurement on the fetched CSV; the prose
lives in `src/ds_agents/benchmark.py`. Verified two ways: 27 opt-in `network` tests against the
live APIs, and `datasets verify --online`, which re-fetches and diffs clean.

**A published number is recorded and is deliberately NOT `baseline_score`.** An OpenML score comes
from OpenML's 10-fold CV run by a third-party flow; `verified_holdout_score` will come from our own
withheld holdout. Dividing them would attribute a *protocol* difference to a *system* difference,
silently. So `published_reference` is informational and labelled with its protocol on the value,
and `baselines` is a *definition with no number in it* naming what `baseline_score` must be
computed from. An AST test asserts no code path from either module reaches `baseline_score`.

**The rule chose the datasets, and it disagreed with the plan twice — both times correctly.**
`SELECTION_RULE` (binary, <=100k rows, <=200 features, <=5M cells, >=5 features surviving
`feature_eng`'s filters) was applied to measurements, not to expectations. `amazon_employee_access`
scored 0 usable features on the first build and 9 on the second, because the first measured the
frame `fetch_openml` returns (ID columns as high-cardinality `category`) and the second measured
the CSV as re-read (`int64`). **The CSV is what gets mounted at `$DS_DATASET`, so all measured
fields now come from the re-read file.** The same round trip moved `kc1`'s `positive_class` from
`'true'` to `'True'` — a manifest recording the former would have named a class no run could match.

**Nine results-row columns stopped lying about datasets with no answer key.**
`graded_for_leakage = bool(planted)` returns `None` from `leakage_caught`, `leakage_precision`, the
`false_alarm*` columns and the four `profiler_/reviewer_` grades, generalising the rule
`leakage_remediated` and the `*_recall` columns already followed. A new `leakage_graded` column
says so outright rather than making a reader infer it from nine nulls. **No published number
moves** — the precondition was proved, not assumed: all 145 committed rows carry a non-empty
`leakage_planted`, and a test fails if that stops being true.

The floor: **681 tests pass offline** (up from 580) plus 27 network tests, ruff clean,
`uv run ds-agents run --dataset toy` green live at $0.0129. Session spend ~$0.07 all in.

## First prompt
Read CLAUDE.md, docs/PLAN.md Phase 4, and the "Start here" above. **The next box is the re-scorer,
and it is what `--subset full` now waits on**: nothing withholds a holdout and nothing computes
`verified_holdout_score` or `baseline_score`, so running the 13 datasets today writes 13 rows whose
headline column is null.

**Wire ONE dataset end to end, not thirteen.** `credit_g` (1,000 rows) is the cheapest. In order:
a `Runnable` protocol over `Fixture | DatasetEntry`; `_dataset_state` beside `_fixture_state`; the
holdout split *before* the graph starts (never in `split_artifact`); `verified_holdout_score` by
re-applying `feature_code_artifact` to the withheld rows — which is the whole reason Phase 1 made
the feature artifact *code*; then `baseline_score` from AMLB's two points (constant class-prior
predictor, tuned RandomForest) fit on the same train split and scored on the same holdout.

Budget warning before anyone types `--subset full`: `dataset_id` is an `eval-diff` condition field,
so 13 datasets is **13 cells**, and these are real datasets — the measured $0.015-$0.030 and
21-86s per run on 200-row toys are floors, not estimates.

## Open questions

- **Is `amazon_employee_access` a real dataset or a degenerate one?** It passes the rule with 9
  usable features, but all nine are integer-encoded high-cardinality IDs (resource, manager,
  role_family...). It may still produce the zero-signal run the `min_usable_features` criterion
  exists to prevent — the criterion measures *survivability*, not *learnability*. Worth one cheap
  run once the run path exists, before it goes in any table.
- **Is a partially-labelled dataset gradeable at all?** `bank_marketing` carries one documented
  leak (`V12`) and `leakage_labelled: false`. It is deliberately NOT in `planted_leakage_columns`,
  because that field is read as a complete list and claiming completeness would score every other
  suspicious column a false alarm. So today the one real, externally-documented leak this project
  has access to is unscored. Some third state between "complete answer key" and "nothing" may be
  worth inventing — but only with a run to test it against.
- **`published_reference` picks the max over uploaded runs, which is not a protocol-stable anchor.**
  `kr_vs_kp` reads 1.000 and `numerai28_6` reads 0.530. Both are real and both are the best anyone
  uploaded, which mostly measures who uploaded. Fine as a realism sanity check, not as a target.
- **The AMLB self-signed certificate is the reason PLAN.md's box is `[~]` and not `[x]`.** AMLB's
  *own* per-dataset numbers are not vendored because `openml1.win.tue.nl` cannot be verified from
  here. If it ever gets a valid cert, those numbers are worth adding as a second reference.
- **Two AMLB candidates could not be fetched at all** — `guillermo` and `Robert`, both md5
  mismatches from OpenML itself. Recorded as `fetch_failed` in `excluded`. Both would have failed
  `max_features` anyway, so nothing was lost, but a *recurring* upstream md5 mismatch is a finding.
- **A run produced no model at all and was recorded `review_verdict: "pass"`.** Unchanged from last
  session; fixing verdict derivation changes a column every committed row carries.
- **The `ci` baseline is NOT a comparand for future ablations.** Unchanged. Both arms of any
  comparison must run in one invocation at one commit.
- **No CI job and no thresholds.** Unchanged; the cheap deterministic gate is the better buy.
- **`errored` needs a companion column.** Unchanged — and `leakage_graded` is now the worked
  example of what that fix looks like.
- **`exhausted` is still uninformative.** Unchanged.
- **Late detection against the cap** is the remaining structural defect. Unchanged.
- **Should `by_category` become the default?** Unchanged.
- **`customer_id` as a false positive is every run.** Unchanged.
- **Duplicate-rows-across-split is still unbuilt.** Unchanged.
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.

## Parking lot

- **`.mcp.json` still hardcodes the toy dataset on argv.** Now more visibly wrong: there are 13
  other datasets it cannot see, and Phase 5's generalist arm needs it parameterised.
- **`benchmark.py` and `benchmark_build.py` are split so reading the manifest never imports
  sklearn or urllib.** Worth keeping if a third module ever wants the registry.
- **`csv_sha256` is recorded but deliberately not asserted in the always-on test tier** — it
  depends on pandas' float formatting, so a version bump would turn it red with nothing wrong.
- **`network` tests are gated on `DS_AGENTS_NETWORK_TESTS=1`, NOT on `-m "not network"` in
  addopts.** pytest's `-m` is a single option, so the hook's `pytest -m fast` would *replace* an
  addopts filter and silently re-enable them.
- **The `min_usable_features` criterion re-implements `feature_eng`'s filters** using its imported
  constants, so the two cannot drift. `blood-transfusion` (4 usable) is what it excludes today.
- **OpenML data ids and task ids are different namespaces and coincide for `credit-g` (31/31).**
  A test asserts at least one entry where they differ, so the pair stays evidence.
- **`positive_class` is the minority level, named explicitly**, because "the second one
  alphabetically" is not a definition anyone can rely on.
- **13 datasets, 22 excluded, every exclusion recorded with the measurement that caused it** — so
  "why is `christine` not in here" is answerable from the file rather than a transcript.
- **The harness's `SUBSETS` still has no `full` entry** and `_resolve_subset`'s message now names
  the re-scorer rather than the manifest.
- **A failed run is charged its cell's ESTIMATE, not its real cost.** Unchanged.
- **The cost cap is invocation-level by choice.** Unchanged.
- **`ds-agents eval` appends to a same-day, same-name file rather than refusing.** Unchanged.
- **`forced_drop_release` is closed to further use.** Unchanged.
- **`feature_eng.py`'s comment on the justification string is still wrong** for a future reader.
- **`harness.py` and `cli.py` still import each other inside functions.** Deferred again.
- **The loop-cap default `3` is written in three places.**
- **`cmd_run` has no per-run `try/except`.** Unchanged.
- **Docker is deferred, not rejected, and `SandboxPool` is the seam.**
- **`ArtifactStore` copies the dataset per run and chmods it 0444** — at 98k rows (`higgs`) that
  copy is no longer free, and `register_dataset` also runs `pd.read_csv` + per-column `nunique`
  **per run**. Worth measuring before the first `full` invocation.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
