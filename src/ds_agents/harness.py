"""The Phase 4 benchmark harness: cells, a plan, and one loop that turns a plan into rows.

A `Cell` is one point in the benchmark grid -- a dataset plus the run conditions `_fixture_state`
needs to reproduce it. `SUBSETS` names groups of cells worth running together. `plan()` expands a
subset into an ordered list of individual runs, replicated and indexed. `run_eval()` walks that
plan, calling a `Runner` for each entry and writing every publishable result through
`cli._append_results_row` -- the same gate `--results` uses, so a results file from this harness
and a results file from `ds-agents run --results` cannot silently disagree about what counts as a
result.

`cli.py` imports this module inside `cmd_eval`, and this module imports `cli` inside its own
functions rather than at module scope. `cli` will need `run_eval`, `run_eval` needs `cli._run_once`
and `cli._append_results_row`, and a top-level `import ds_agents.cli` on either side of that would
be circular. Deferring the import to call time breaks the cycle at no real cost, since both call
sites are function bodies, not import-time code.

`random_seed` never appears as a `Cell` field, a `plan()` parameter, or a `run_eval()` argument, and
it should not gain one. It is a DATA seed -- it fixes the train/test split and the estimators'
internal seeding on `RunConfig`, not anything about which model answered. Moving it between cells
would fold split variance into a number this project reports as model variance, and it would break
comparability with all 84 rows already committed under `random_seed=20260822`. `replicates` exists
instead: a replicate reruns the SAME seed, so the spread it measures is model nondeterminism alone.
That distinction is not academic -- it is the entire finding of 2026-08-28's forced-drop-release
session, where two cells at identical configuration and the identical seed returned
`leakage_remediated` 9/10 and 6/10, and a single 10-run cell turned out to be unable to resolve a
4/10 difference from noise.
"""

import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ds_agents.holdout import PreparedDataset
from ds_agents.naming import Naming
from ds_agents.state import (
    DEFAULT_LOOP_CAP,
    ObjectionClosure,
    ObjectionRouting,
    PipelineState,
    ReviewerPrompt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# Where results land unless a caller says otherwise. Named so the CLI can pass the same default
# explicitly rather than keeping a second copy of the path that could drift from this one.
RESULTS_DIR = REPO_ROOT / "evals" / "results"

# How many runs may fail back to back before the invocation gives up. Named rather than inlined,
# beside `loop_cap` and `max_cost_usd` as the third bound on how much a single invocation can
# consume: a config broken in a way every run hits should cost three runs, not a whole cap.
MAX_CONSECUTIVE_FAILURES = 3

# Keeps a path separator (and anything else shell- or filesystem-unfriendly) out of the results
# filename, and keeps every filename this harness has ever produced grep-consistent.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True)
class Cell:
    """One point in the benchmark grid: a dataset plus the run conditions that reproduce it.

    Every field except `name`, `dataset` and `est_cost_usd` mirrors a keyword `_fixture_state`
    accepts, so `conditions()` can hand them straight to `cli._run_once` with no translation layer
    to drift out of sync. `est_cost_usd` is planning-only -- it feeds the cost cap check before a
    run and the printed plan, and it is never written to a results row, because a row that carried
    an estimate next to `cost_usd` would invite averaging the two.

    Deliberately absent: `forced_drop_release`. It is a `Literal["withdrawn_only",
    "resolved_or_withdrawn"]` on `RunConfig` whose off value reproduces a known defect rather than
    offering a second design (see `docs/NEXT.md`'s parking lot and DECISIONS.md 2026-08-28, fifth
    entry) -- it exists for exactly one same-commit control, already run. A `Cell` field for it
    would invite a second use, so `Cell.conditions()` never mentions it and `_fixture_state` falls
    back to its correct default (`withdrawn_only`) every time.
    """

    name: str
    dataset: str
    model: str = "haiku"
    reviewer_model: str | None = None
    naming: Naming = "descriptive"
    reviewer_prompt: ReviewerPrompt = "base"
    loop_cap: int = DEFAULT_LOOP_CAP
    objection_routing: ObjectionRouting = "as_addressed"
    objection_closure: ObjectionClosure = "off"
    est_cost_usd: float = 0.03

    def conditions(self) -> dict[str, Any]:
        """Keyword arguments for `cli._fixture_state` / `cli._run_once`.

        Named `model_name` / `reviewer_model_name` here because that is what `_fixture_state`
        calls them -- `Cell.model` stays short because it is read constantly while defining
        `SUBSETS` below.
        """
        return {
            "model_name": self.model,
            "reviewer_model_name": self.reviewer_model,
            "naming": self.naming,
            "reviewer_prompt": self.reviewer_prompt,
            "loop_cap": self.loop_cap,
            "objection_routing": self.objection_routing,
            "objection_closure": self.objection_closure,
        }


# The `ci` subset: one cell per registered fixture, all at loop_cap=3 and objection_closure="off"
# (both are the dataclass defaults, so neither is written out below). `claims-opaque-which` and
# `reissued-opaque-which` both run the naming and routing conditions together because that is the
# combination `docs/NEXT.md` flags as the one worth a regression gate: opaque naming is what made
# the trap findable at all, and `by_category` routing is what let a caught trap actually get
# dropped. Costs are rough per-run estimates for the cost-cap check, not a promise.
#
# The three estimates below are MEASURED means from the 2026-08-31 `ci` baseline (n=10 a cell,
# `evals/results/2026-08-31_ci-baseline.jsonl`), replacing the guesses they shipped with. They are
# rounded up to the nearest $0.001, because this number's job is to stop the cap being overrun and
# a mean that is right half the time is the wrong side to be wrong on. The originals were 0.010 /
# 0.030 / 0.025: the whole subset came in at $0.7291 against a $0.65 estimate, a 12% under-count of
# which `toy-default` is about 63% and `reissued-opaque-which` about 42% (`claims-opaque-which` was
# the one good guess, over by 1%). The spread within a cell is what makes a mean a poor guarantee
# here -- a run that takes the review loop three times costs around 2.5x one that passes first
# time. `est_cost_usd` is planning-only and never reaches a results row, so changing it revises no
# published number -- it only changes where a future cap truncates.
SUBSETS: dict[str, tuple[Cell, ...]] = {
    "toy": (Cell(name="toy-default", dataset="toy", est_cost_usd=0.015),),
    "ci": (
        Cell(name="toy-default", dataset="toy", est_cost_usd=0.015),
        Cell(
            name="claims-opaque-which",
            dataset="claims_timing",
            naming="opaque",
            reviewer_prompt="which_column",
            objection_routing="by_category",
            est_cost_usd=0.030,
        ),
        Cell(
            name="reissued-opaque-which",
            dataset="reissued_ids",
            naming="opaque",
            reviewer_prompt="which_column",
            objection_routing="by_category",
            est_cost_usd=0.029,
        ),
    ),
    # The first subset that names a dataset nobody here wrote. One cell, one dataset, on purpose:
    # `credit_g` is the cheapest thing in the manifest at 1000 rows, and the point is to prove the
    # run-and-grade path end to end before thirteen of them are paid for. `est_cost_usd` is a
    # measurement like every number above it: the 4-row 2026-08-31 `credit-g-smoke` invocation
    # averaged $0.0168 per run. The pre-run guess was 0.040, wrong by more than a factor of two and
    # wrong in the cheap direction -- which is why the other twelve get measured before `full`.
    "bench-smoke": (Cell(name="credit-g-default", dataset="credit_g", est_cost_usd=0.017),),
    # The subset that exists to price the rest of the manifest. Four cells forming a 2x2 over the
    # TWO cost axes this project had been treating as one: wall clock tracks ROWS (fits), token cost
    # tracks COLUMNS (schema, profile summaries and importance lists all reach a prompt). `credit_g`
    # is small on both, which is why one measurement of it could price nothing else.
    #
    #                    few columns              many columns
    #   few rows         phoneme   5404 x 5       jasmine  2984 x 144
    #   many rows        amazon   32769 x 9       nomao   34465 x 118
    #
    # Crossed on purpose. With only a wide-and-tall cell the two axes stay confounded and a timeout
    # there is uninterpretable; with the cross, `jasmine` vs `nomao` separates width from length and
    # `phoneme` vs `amazon` separates length from width.
    #
    # `adult` is NOT here. Not because it can't be run any more -- the split manifest fix
    # (2026-09-01, `ds_agents/split_manifest.py`) landed after this cell was already run, and
    # `adult`, `bank_marketing`, `higgs` and `numerai28_6` are all runnable now. It is absent
    # because this is a measurement, not a subset definition: these four numbers are what
    # `bench-mid`'s 2026-09-01 invocation actually cost on the datasets it actually ran, and
    # `adult` was not one of them. Pricing it, and the other three, is what `bench-tall` below does.
    #
    # MEASURED means from the 2026-09-01 run (n=2 a cell,
    # `evals/results/2026-09-01_bench-mid.jsonl`), rounded up to the nearest $0.001, replacing the
    # guesses they shipped with. The guesses were 0.020 / 0.060 / 0.030 / 0.065 and every one was
    # HIGH -- the invocation came in at $0.2120 against a $0.35 estimate. That is the opposite of
    # `bench-smoke`'s error and it is the safe direction to be wrong in, but it is still a 40% miss.
    #
    # One caveat that a mean cannot carry: all 8 runs took the review loop exactly once and none
    # raised an objection, so these are the cost of a run that passes first time.
    #
    # `amazon_employee_access` was $0.013 until 2026-09-02 and is $0.014 now. Not a re-measurement:
    # its mean is $0.013007, and rounding that UP -- the rule this comment has always stated, for
    # the stated reason that an estimate must never be optimistic -- gives $0.014. It had been
    # rounded to nearest instead, which put it $0.000007 below the mean. Found by
    # `tests/test_cost_model.py`, which derives every price here from the results files rather than
    # trusting the literal, on the first run of that test. `est_cost_usd` never reaches a results
    # row, so no published number moves. The `ci` comment
    # above records that a run looping three times costs about 2.5x one that does not, and nothing
    # here has been observed looping.
    "bench-mid": (
        Cell(name="phoneme-narrow-short", dataset="phoneme", est_cost_usd=0.011),
        Cell(name="jasmine-wide-short", dataset="jasmine", est_cost_usd=0.042),
        Cell(name="amazon-narrow-tall", dataset="amazon_employee_access", est_cost_usd=0.014),
        Cell(name="nomao-wide-tall", dataset="nomao", est_cost_usd=0.041),
    ),
    # The subset that prices the last four datasets, and the one that tests the cost model rather
    # than just applying it. `bench-mid` fitted `cost ~= a + b * n_features` on four points and the
    # project started quoting new runs against it; this is the arm that asks whether that model
    # survives contact with anything it was not fitted on.
    #
    # WHAT THIS IS, LITERALLY: a line along the ROW axis. 45k to 98k rows, extending 2.9x beyond the
    # tallest thing the model was fitted on (`nomao`, 34,465) at widths of 14 to 28, which is
    # interpolation on the fitted 5-to-144 range. Within these four, rows and column count are
    # nearly rank-identical (45k/16, 49k/14, 96k/21, 98k/28), so they are CONFOUNDED and no residual
    # computed inside `bench-tall` alone separates them.
    #
    # WHAT NOBODY CHOSE, AND IT IS THE INTERESTING PART: all four datasets the cost model was fitted
    # on have ZERO categorical columns. `phoneme`, `jasmine`, `amazon_employee_access` and `nomao`
    # are entirely numeric, so `feature_eng` one-hot encodes nothing on any of them and `LEVELS` in
    # the emitted transform is empty in all four. That fell out of picking a 2x2 on rows x columns,
    # because dtype was not one of the two axes anyone was thinking about. Both out-of-sample points
    # measured since are categorical and both came in ABOVE the model: `credit_g` (13 categorical,
    # 54 one-hot levels) +$0.0015, `adult` (7 categorical + 1 skipped, 58 levels) +$0.0019.
    # `credit_g` has 1000 rows, so that residual cannot be a row term.
    #
    # There is a mechanism. The reviewer reads `feature_code_artifact` into its prompt, and that
    # artifact carries `LEVELS`, `FEATURE_ORDER` and `COLUMN_SOURCE`, which grow with one-hot level
    # count and not with rows -- and `feature_eng` additionally reasons in prose about cardinality,
    # which a fully numeric dataset never triggers.
    #
    # These four separate the two hypotheses, because rows and categoricals are ANTI-CORRELATED
    # across them:
    #
    #                     categorical                     numeric
    #   ~45-49k rows      adult 7 cat / 58 levels
    #                     bank_marketing 9 cat / 44 levels
    #   ~96-98k rows                                      numerai28_6 0 cat
    #                                                     higgs 0 cat
    #
    # So H_rows (the model needs a row term) predicts `numerai28_6` and `higgs` run high, and
    # H_categorical predicts `adult` and `bank_marketing` do. Opposite orderings, one $0.24 run.
    # That is luck rather than design and it is worth saying so: had the four been correlated, this
    # would have cost the same and settled nothing.
    #
    # A 2x2 over rows x categoricals DOES exist, but only by pooling with committed cells
    # (short+numeric = phoneme/jasmine, tall+numeric = amazon/nomao/numerai/higgs,
    # short+categorical = credit_g, tall+categorical = adult/bank_marketing -- these two fill the
    # empty corner). It crosses a `commit` boundary, `commit` is an `evaldiff.CONDITION_FIELDS`
    # field, and `eval-diff` will refuse the comparison. It is a table you may draw by hand with the
    # boundary named, never one the tooling blesses.
    #
    # MEASURED means from the 2026-09-02 run (n=4 a cell, 16 rows, $0.3021,
    # `evals/results/2026-09-02_bench-tall.jsonl`), rounded up to the nearest $0.001, replacing the
    # four predictions they shipped with. The predictions were $0.014 / $0.014 / $0.016 / $0.017
    # from `a + b * n_features`; measured $0.0159 / $0.0178 / $0.0172 / $0.0246.
    #
    # WHAT THE ARM SETTLED, and it is not either pre-registered hypothesis.
    #
    #   cell               cols  rows   cat  predicted  measured   residual   band +/-$0.0026
    #   adult                14  48.8k    7    $0.0139   $0.0159    +$0.0020  inside
    #   bank_marketing       16  45.2k    9    $0.0144   $0.0178    +$0.0034  ABOVE
    #   numerai28_6          21  96.3k    0    $0.0155   $0.0172    +$0.0017  inside
    #   higgs                28  98.1k    0    $0.0172   $0.0246    +$0.0074  ABOVE
    #
    # H_categorical predicted the two categorical cells high; one was and one was not.
    # H_rows predicted the two tall cells high; one was and one was not, and `bank_marketing` at
    # 45k ran higher than `numerai28_6` at 96k, which a row term cannot order. So neither
    # pre-registered ordering appeared.
    #
    # What DID appear is simpler and was not one of the two options: **all four residuals are
    # positive**, as is `credit_g`'s +$0.0015, so the column-only model has under-predicted 5 of 5
    # datasets outside the four it was fitted on (sign test p=0.031). Refitting on all nine measured
    # datasets, a row term cuts the residual sd from $0.00296 to $0.00224 while a categorical-count
    # term makes it WORSE ($0.00320) and adds nothing once rows are in ($0.00206 for both).
    # **H_categorical is refuted**: the mechanism was plausible -- the reviewer reads a transform
    # artifact whose `LEVELS` grow with one-hot count -- and the data says it is not worth a term.
    # H_rows survives as a term but not as the clean ordering it was registered as.
    #
    # The nine-dataset refit, which is what `full` below is priced from for the datasets that have
    # never been run:  cost ~= $0.010163 + $0.000230 * n_features + $0.005609 * (n_rows / 1e5).
    #
    # The caveat a mean cannot carry, and it is the same one `bench-mid` recorded: all 16 runs
    # raised ZERO objections, took the review loop exactly once and returned `pass`. Nothing in this
    # project has ever been observed looping on a manifest dataset, and the `ci` comment above
    # records that a run that loops three times costs about 2.5x. Every number in this dict is the
    # cost of a run that passes first time.
    "bench-tall": (
        Cell(name="adult-categorical-tall", dataset="adult", est_cost_usd=0.016),
        Cell(name="bank-categorical-tall", dataset="bank_marketing", est_cost_usd=0.018),
        Cell(name="numerai-numeric-tall", dataset="numerai28_6", est_cost_usd=0.018),
        Cell(name="higgs-numeric-tall", dataset="higgs", est_cost_usd=0.025),
    ),
    # Every binary dataset in the manifest, one cell each. This entry exists as of 2026-09-02,
    # when the last four datasets were priced; before that `_resolve_subset` raised a bespoke error
    # for the name, and the blocker that error described moved four times.
    #
    # It does NOT move a fifth time, because it is no longer a code or a costing problem. Nine of
    # the thirteen numbers below are measured means from real invocations. The other four are
    # modelled, and they are the four SMALLEST shapes in the manifest -- `australian` (690 rows),
    # `kc1` (2,109), `sylvine` (5,124), `kr_vs_kp` (3,196) -- priced from the nine-dataset refit
    # recorded in the `bench-tall` comment plus one residual sd ($0.0022) and rounded up, because
    # the one thing that arm established about this model is that it under-predicts out of sample.
    # The cap, not the estimate, is what protects the budget from them being wrong.
    #
    # `kr_vs_kp` is the one to watch: 36 columns, ALL categorical, 73 one-hot levels, which is more
    # than any dataset anyone has run. H_categorical was refuted as a cost term at 7 to 13
    # categorical columns and that is not the same as refuted at 36, so its $0.022 is the softest
    # number here.
    #
    # WHAT RUNNING THIS COSTS, which is the thing that was never written down while the blocker was
    # always something else: 13 cells at $0.276 a full pass. At the `--replicates 2 --n 2` that
    # /run-eval requires for anything comparative, that is 52 runs, about **$1.10 and 45-70
    # minutes**. Nothing technical stands in the way of that. It is a budget decision and it has
    # never been put to anyone, which is now the honest content of the blocker.
    "full": (
        # measured
        Cell(name="phoneme", dataset="phoneme", est_cost_usd=0.011),
        Cell(name="amazon-employee-access", dataset="amazon_employee_access", est_cost_usd=0.014),
        Cell(name="adult", dataset="adult", est_cost_usd=0.016),
        Cell(name="credit-g", dataset="credit_g", est_cost_usd=0.017),
        Cell(name="bank-marketing", dataset="bank_marketing", est_cost_usd=0.018),
        Cell(name="numerai28-6", dataset="numerai28_6", est_cost_usd=0.018),
        Cell(name="higgs", dataset="higgs", est_cost_usd=0.025),
        Cell(name="nomao", dataset="nomao", est_cost_usd=0.041),
        Cell(name="jasmine", dataset="jasmine", est_cost_usd=0.042),
        # modelled, never run
        Cell(name="australian", dataset="australian", est_cost_usd=0.016),
        Cell(name="sylvine", dataset="sylvine", est_cost_usd=0.018),
        Cell(name="kc1", dataset="kc1", est_cost_usd=0.018),
        Cell(name="kr-vs-kp", dataset="kr_vs_kp", est_cost_usd=0.022),
    ),
}


def _resolve_subset(subset: str) -> tuple[Cell, ...]:
    """`SUBSETS[subset]`, or a `ValueError` listing what would have resolved.

    `full` used to get a bespoke message of its own, rewritten four times as its blocker moved --
    the manifest, then the run-and-grade path, then the split manifest, then cost. As of 2026-09-02
    it is an ordinary entry in `SUBSETS` and needs no special case, because the last four datasets
    are priced. What is left is not a blocker this function can describe: running `full` properly is
    52 runs at about $1.10, and whether to spend that is a decision for a person. The comment on the
    entry itself says so, which is where a reader who is about to spend the money will be looking.
    """
    if subset in SUBSETS:
        return SUBSETS[subset]
    available = ", ".join(sorted(SUBSETS))
    raise ValueError(f"unknown eval subset {subset!r}. Available subsets: {available}.")


@dataclass(frozen=True)
class PlannedRun:
    """One entry in an executed plan: which cell, which replicate, which index within it, and
    where it fell in the invocation's overall execution order."""

    cell: Cell
    replicate: int  # 1..replicates
    index: int  # 0..n-1 within (cell, replicate)
    seq: int  # 0..len(plan)-1, execution order across the whole invocation


def plan(cells: Sequence[Cell], *, replicates: int, n: int) -> list[PlannedRun]:
    """Expand `cells` into an ordered list of runs: every cell's replicate 1 first (n runs each),
    then every cell's replicate 2, and so on.

    Replicate-major, not cell-major. `run_eval`'s cost cap can bind partway through a plan, and
    whichever ordering is chosen decides who gets shortchanged when it does. A cell-major plan --
    finish cell A's N runs, then cell B's, ... -- means a cap that binds mid-run leaves every cell
    before the cut at full strength and drops every cell after it to zero. That is exactly what
    happened to the Sonnet reviewer cells in the 2026-08-28 reviewer-ablation run: budgeted at n=8,
    they landed at n=4 and n=3 because the cap bound inside the first cell's replicate loop and the
    later cell never started, and an unequal n=4-vs-n=3 pairing could not be quoted as a clean 2x2.
    Replicate-major ordering means a cap that binds after k runs drops roughly the same fraction
    from every cell, because it walks one full pass across all cells before starting the next
    replicate of any of them.
    """
    runs: list[PlannedRun] = []
    seq = 0
    for replicate in range(1, replicates + 1):
        for cell in cells:
            for index in range(n):
                runs.append(PlannedRun(cell=cell, replicate=replicate, index=index, seq=seq))
                seq += 1
    return runs


Runner = Callable[[PlannedRun, Path], PipelineState]


@dataclass
class CellTally:
    """Running counts for one cell across an invocation."""

    rows_written: int = 0
    rows_refused: int = 0  # publishable() said no -- a stub run, most likely
    runs_failed: int = 0  # raised before producing a state at all
    spend_usd: float = 0.0  # actual for completed runs, estimated for failed ones -- see `charged`
    charged_estimate_usd: float = 0.0  # the part of spend_usd nobody measured
    leakage_remediated: int = 0  # written rows where results_row()["leakage_remediated"] is True
    errored: int = 0  # written rows where results_row()["errored"] is True


@dataclass
class HarnessReport:
    """What one `run_eval` invocation did."""

    path: Path | None  # None only for a dry run -- nothing was written anywhere
    rows_written: int
    rows_refused: int
    runs_failed: int
    spend_usd: float
    # How much of `spend_usd` is an estimate rather than a measurement. A run that raises took its
    # partial cost with it -- the tokens were spent, but the `PipelineState` holding the count never
    # came back -- so the failed run is charged its cell's estimate instead. Reported separately
    # because a spend figure that silently mixes measured and guessed dollars is the kind of number
    # this project refuses to publish, and because under-counting here would let a run of failures
    # walk straight through the cost cap.
    charged_estimate_usd: float
    stopped_early: str | None  # None | "cost cap" | "consecutive failures"
    per_cell: dict[str, CellTally]


def _live_run(artifacts_root: Path, *, transport: str, no_live: bool) -> Runner:
    """The real runner: materializes each (dataset, naming) pair once, then calls `cli._run_once`.

    Caching is keyed on (dataset, naming) rather than done once per cell, because two cells in a
    subset can share both -- `claims-opaque-which` and a hypothetical second `claims_timing` cell
    at the same naming would otherwise rewrite the same file twice. Materializing once per
    invocation, not once per run, is the same rule `cmd_run --repeat` follows for `--naming opaque`:
    every run that is supposed to see the same bytes has to actually see the same bytes, and
    rewriting the file N times is N chances for it not to.

    `git_commit()` is read here, once, under exactly that rule, and the 2026-08-31 `ci` baseline is
    what proved it belongs here rather than inside `_run_once`. The results file is untracked until
    someone commits it, so writing row 0 dirties the tree, and a per-run read recorded run 0 at
    `8a629bf` and runs 1..29 at `8a629bf-dirty`. `commit` is one of `evaldiff.CONDITION_FIELDS`, so
    that fragmented `toy-default` into cells of n=1 and n=9 -- the harness contaminating its own
    provenance with its own output. One read, before any row exists, cannot.
    """
    # Deferred import: see the module docstring for why this cannot be a top-level import.
    from ds_agents import cli
    from ds_agents.holdout import prepare
    from ds_agents.provenance import git_commit
    from ds_agents.runnable import Runnable, resolve
    from ds_agents.state import RunConfig

    commit = git_commit()

    runnables: dict[str, Runnable] = {}
    prepared_by_key: dict[tuple[str, str], PreparedDataset] = {}

    def runner(planned_run: PlannedRun, root: Path) -> PipelineState:
        cell = planned_run.cell
        # `resolve` rather than `load_fixture`: a Cell's dataset may now name a manifest dataset,
        # and the two registries share one namespace.
        dataset = runnables.setdefault(cell.dataset, resolve(cell.dataset))
        key = (cell.dataset, cell.naming)
        if key not in prepared_by_key:
            # Once per (dataset, naming) per invocation, for the reason `materialize` already was
            # and one more: every replicate in a cell must be graded against the SAME withheld
            # rows, or the runs in it are not measuring the same question and the cell cannot be
            # pooled. `withheld` is a sibling of `input`, so it is outside every run root.
            prepared_by_key[key] = prepare(
                dataset,
                cell.naming,
                into=artifacts_root / "input" / f"{cell.dataset}-{cell.naming}",
                withheld_into=artifacts_root / "withheld" / f"{cell.dataset}-{cell.naming}",
                seed=RunConfig().random_seed,
            )
        return cli._run_once(
            dataset,
            root=root,
            prepared=prepared_by_key[key],
            commit=commit,
            transport=transport,
            no_live=no_live,
            **cell.conditions(),
        )

    return runner


def _print_plan(planned: list[PlannedRun], cells: Sequence[Cell], max_cost_usd: float) -> None:
    estimated = sum(r.cell.est_cost_usd for r in planned)
    print(
        f"plan: {len(planned)} runs across {len(cells)} cells (cells={[c.name for c in cells]})",
        file=sys.stderr,
    )
    for r in planned:
        print(
            f"  seq={r.seq:>4}  cell={r.cell.name:<24} replicate={r.replicate} index={r.index}",
            file=sys.stderr,
        )
    print(f"estimated cost: ${estimated:.4f}  (cap ${max_cost_usd:.2f})", file=sys.stderr)


def _print_report(report: HarnessReport) -> None:
    print(
        f"\ndone: {report.rows_written} rows written, {report.rows_refused} refused, "
        f"{report.runs_failed} runs failed, ${report.spend_usd:.4f} spent"
        + (
            f" (${report.charged_estimate_usd:.4f} of it estimated, from failed runs)"
            if report.charged_estimate_usd
            else ""
        )
        + (f", stopped early: {report.stopped_early}" if report.stopped_early else ""),
        file=sys.stderr,
    )
    for name, tally in report.per_cell.items():
        print(
            f"  {name:<24} written={tally.rows_written} refused={tally.rows_refused} "
            f"failed={tally.runs_failed} spend=${tally.spend_usd:.4f} "
            f"remediated={tally.leakage_remediated} errored={tally.errored}",
            file=sys.stderr,
        )
    if report.path is not None:
        print(f"results: {report.path}", file=sys.stderr)


def run_eval(
    *,
    subset: str,
    name: str,
    replicates: int = 1,
    n: int = 1,
    max_cost_usd: float = 0.50,
    out_dir: Path = RESULTS_DIR,
    artifacts_dir: Path | None = None,
    transport: str = "mcp",
    no_live: bool = False,
    dry_run: bool = False,
    today: date | None = None,
    runner: Runner | None = None,
) -> HarnessReport:
    """Run `subset` and append every publishable result to a dated JSONL file.

    `name` must match `^[a-z0-9][a-z0-9-]*$`, checked before anything else runs, so a name with a
    path separator or a typo'd flag is caught before a cent is spent rather than surfacing as a
    write to some unintended path.

    Prints the plan and the cost estimate before running anything, live or not. `dry_run` returns
    right after that print, with `path=None` and every counter at zero -- it calls `runner` zero
    times, which is the property that makes `--dry-run` actually free.

    The cost cap is invocation-level, not per-cell: before each run, if `spend + cell.est_cost_usd`
    would exceed `max_cost_usd`, the run does not start and `stopped_early` becomes `"cost cap"`.
    `spend` is tracked in ACTUAL dollars for every run that completes, off `state.total_cost_usd`.
    A run that RAISES is the exception, and it is charged its cell's estimate instead: the tokens it
    burned before failing are unrecoverable, because the state holding the count never came back.
    Guessing high is the safe direction -- ignoring failed runs would let a cell that fails late,
    after paying for most of a pipeline, walk straight through the cap -- and the guessed portion is
    reported separately as `charged_estimate_usd` so no one reads a mixed figure as a measurement.

    Each run is wrapped in its own `try/except Exception`: a run that raises is printed, counted in
    `runs_failed`, and the loop moves on to the next one. `SystemExit` is not caught here and
    propagates, because `cli._run_once` raises it to mean the run never started at all (an unknown
    fixture, a tool server that would not launch) -- and whatever stopped run 7 from starting will
    stop run 8 the same way, so there is nothing to gain by continuing. Three CONSECUTIVE ordinary
    failures abort the whole invocation with `stopped_early="consecutive failures"`, so a broken
    config cannot silently burn the entire cost cap one exception at a time.

    Every written row carries the harness's own annotations under `extra=`: `cell`, `replicate`,
    `run_index`, `eval_subset`, `eval_name`. A row that `publishable()` refuses is not written, but
    it still increments `rows_refused` -- a refused row is load-bearing, because it changes a
    cell's denominator just as surely as a written one changes its numerator.
    """
    if not _NAME_RE.match(name):
        raise ValueError(
            f"eval name {name!r} must match {_NAME_RE.pattern!r} (lowercase, digits, hyphens, no "
            "path separators) -- it becomes a filename"
        )

    cells = _resolve_subset(subset)
    planned = plan(cells, replicates=replicates, n=n)
    per_cell = {cell.name: CellTally() for cell in cells}

    _print_plan(planned, cells, max_cost_usd)

    if dry_run:
        print("dry run: no runner called, nothing written", file=sys.stderr)
        return HarnessReport(
            path=None,
            rows_written=0,
            rows_refused=0,
            runs_failed=0,
            spend_usd=0.0,
            charged_estimate_usd=0.0,
            stopped_early=None,
            per_cell=per_cell,
        )

    # Deferred import: see the module docstring.
    from ds_agents import cli

    today = today or date.today()
    path = out_dir / f"{today:%Y-%m-%d}_{name}.jsonl"
    if path.exists():
        print(f"note: {path} already exists, appending to it", file=sys.stderr)

    artifacts_root = artifacts_dir or Path(tempfile.mkdtemp(prefix="ds-agents-eval-"))
    if runner is None:
        runner = _live_run(artifacts_root, transport=transport, no_live=no_live)

    spend = 0.0
    charged_estimate = 0.0
    rows_written = 0
    rows_refused = 0
    runs_failed = 0
    consecutive_failures = 0
    stopped_early: str | None = None

    for planned_run in planned:
        cell = planned_run.cell
        if spend + cell.est_cost_usd > max_cost_usd:
            print(
                f"stopping: cost cap ${max_cost_usd:.2f} would be exceeded by "
                f"seq={planned_run.seq} (cell={cell.name}, spent so far ${spend:.4f})",
                file=sys.stderr,
            )
            stopped_early = "cost cap"
            break

        run_root = artifacts_root / cell.name / f"seq-{planned_run.seq:04d}"
        try:
            state = runner(planned_run, run_root)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - any run failure is counted the same way
            print(
                f"run failed (seq={planned_run.seq}, cell={cell.name}, "
                f"replicate={planned_run.replicate}, index={planned_run.index}): {exc!r}",
                file=sys.stderr,
            )
            runs_failed += 1
            per_cell[cell.name].runs_failed += 1
            # The tokens this run spent before it raised are unrecoverable -- the state that
            # counted them never came back -- so it is charged its cell's estimate. Guessing high
            # is the safe direction: a cap that ignored failed runs entirely would let a cell that
            # fails late, after paying for most of a pipeline, spend without limit.
            spend += cell.est_cost_usd
            charged_estimate += cell.est_cost_usd
            per_cell[cell.name].spend_usd += cell.est_cost_usd
            per_cell[cell.name].charged_estimate_usd += cell.est_cost_usd
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"stopping: {MAX_CONSECUTIVE_FAILURES} consecutive failures", file=sys.stderr)
                stopped_early = "consecutive failures"
                break
            continue

        consecutive_failures = 0
        spend += state.total_cost_usd
        per_cell[cell.name].spend_usd += state.total_cost_usd

        wrote = cli._append_results_row(
            state,
            path,
            extra={
                "cell": cell.name,
                "replicate": planned_run.replicate,
                "run_index": planned_run.index,
                "eval_subset": subset,
                "eval_name": name,
            },
        )
        if wrote:
            rows_written += 1
            per_cell[cell.name].rows_written += 1
            row = state.results_row()
            if row.get("leakage_remediated"):
                per_cell[cell.name].leakage_remediated += 1
            if row.get("errored"):
                per_cell[cell.name].errored += 1
        else:
            rows_refused += 1
            per_cell[cell.name].rows_refused += 1

    report = HarnessReport(
        path=path,
        rows_written=rows_written,
        rows_refused=rows_refused,
        runs_failed=runs_failed,
        spend_usd=spend,
        charged_estimate_usd=charged_estimate,
        stopped_early=stopped_early,
        per_cell=per_cell,
    )
    _print_report(report)
    return report
