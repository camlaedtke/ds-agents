"""The benchmark registry: external datasets with published references, and how to read them.

`evals/datasets/manifest.yaml` names 10 to 15 real datasets drawn from the AutoML Benchmark
(AMLB) -- Gijsbers et al., JMLR 25(101), 2024 -- whose classification tasks are OpenML suite 271.
It is the counterpart to `fixtures.py`: the fixtures are small, synthetic, and carry a planted
answer key, while these are real, unlabelled for leakage, and carry a citation instead.

The manifest is a GENERATED file. `.claude/settings.json` denies Edit and Write under
`evals/datasets/`, so it can only be written by `benchmark_build.refresh()` running as a program.
That is not an inconvenience to route around -- it is the mechanism that makes every number in the
file traceable. A manifest is exactly the shape of file that invites a plausible-looking number to
be typed into it, and a wrong published baseline is unfalsifiable and would silently poison every
table in the Phase 5 writeup. So: every number in the manifest is either a field of an API
response or a measurement taken on the fetched frame, and every sentence a person chose is a
constant in THIS file, where review and the fast-test hook can see it.

Two rules this module exists to keep, both pinned by test in `tests/test_benchmark_manifest.py`:

  1. Nothing in `nodes/` imports it. Same reason as `fixtures.py` -- a node that can read a
     registry can read the answer key.
  2. Nothing in it can reach `PipelineState.baseline_score`. A published score was produced on
     OpenML's own 10-fold CV by somebody else's flow; `verified_holdout_score` will be produced by
     this pipeline on a holdout this repo withholds. Dividing one by the other and calling it
     `score_ratio` would attribute the difference between two PROTOCOLS to the difference between
     two SYSTEMS. `published_reference` is therefore informational and deliberately has no path
     into the results row. `baseline_score` must be computed alongside the run it is compared
     against, from the baseline DEFINITION recorded in the manifest.

This module is deliberately dumb in the way `fixtures.py` is: it resolves paths, validates shape,
and hands back a typed object. It does no network I/O and imports neither sklearn nor urllib --
that is `benchmark_build.py`, which is the only thing that writes the manifest.
"""

import hashlib
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "evals" / "datasets" / "manifest.yaml"
# Gitignored. The manifest pins each dataset by OpenML data id and upstream md5, so the CSVs are
# reproducible from it and do not belong in git -- several are tens of thousands of rows, against a
# committed data footprint that is currently three fixtures under 40KB each.
CACHE_ROOT = REPO_ROOT / ".cache"
OPENML_CACHE = CACHE_ROOT / "openml"
DATASET_CACHE = CACHE_ROOT / "datasets"

SCHEMA_VERSION = 1

# Pinned to an immutable commit rather than a branch or a tag, because a tag can be moved and the
# whole point of recording it is that `test_every_entry_is_a_member_of_its_declared_amlb_list`
# checks the same bytes next year that it checked today.
AMLB_REF = "dfe8d21871e2dec2b7cab6c9fbfe3575ce4dadc6"


class SelectionRule(BaseModel):
    """Which AMLB datasets are admitted, as data rather than as taste.

    The rule is a constant here, written into the manifest by the generator, and enforced against
    the manifest's own recorded measurements by a fast test. Stating it in prose only would make
    "why is `christine` missing" unanswerable in six months.
    """

    model_config = ConfigDict(extra="forbid")

    rule_version: int
    task_types: list[str]
    metric: str
    max_rows: int
    max_features: int
    max_cells: int
    min_usable_features: int
    target_count: tuple[int, int]


SELECTION_RULE = SelectionRule(
    rule_version=1,
    # One task type and one metric across the whole set. `score_ratio` is direction-aware, and a
    # mixed set would also drag in the r2-baseline-of-exactly-0.0 hole that `score_ratio`'s own
    # docstring already flags as a real gap.
    task_types=["binary"],
    metric="roc_auc",
    # Wall time. `reissued_ids` already averages 86s a run at 2,000 rows.
    max_rows=100_000,
    # `permutation_importance` costs n_source_columns x n_repeats scoring passes PER CANDIDATE, so
    # width is the expensive axis, not length.
    max_features=200,
    # The bound that actually tracks permutation cost. `max_features` alone admits a 100k x 200
    # dataset that is 20x the work of a 100k x 10 one.
    max_cells=5_000_000,
    # The criterion with the sharpest teeth, and the reason it exists: `feature_eng` mechanically
    # drops identifier-like columns and skips categoricals above MAX_ONE_HOT_LEVELS. A dataset that
    # is mostly high-cardinality categoricals arrives at the modeler as an empty matrix, which is
    # exactly the zero-feature `pass`-with-no-model run docs/NEXT.md carries as an open question.
    # Without this bound we would reproduce that run a dozen times and call it a benchmark.
    min_usable_features=5,
    target_count=(10, 15),
)


class GeneratedWith(BaseModel):
    """Library versions the measurements were taken under.

    Recorded because `n_usable_features` and `positive_rate` are measured by pandas and sklearn,
    and a future disagreement between the manifest and a re-fetch should be attributable.
    """

    model_config = ConfigDict(extra="forbid")

    python: str
    pandas: str
    scikit_learn: str


class SourceBlock(BaseModel):
    """Where the set came from, in enough detail to re-fetch it."""

    model_config = ConfigDict(extra="forbid")

    benchmark: str
    citation: str
    paper_url: str
    openml_suite_id: int
    openml_suite_name: str
    amlb_repo: str
    amlb_ref: str
    amlb_lists: dict[str, str]
    openml_data_api: str
    openml_eval_api: str
    # A string, not a date. YAML 1.1 parses an unquoted 2026-08-31 into a datetime.date, and typing
    # this as `date` would let the model quietly accept an unquoted scalar and hide the coercion
    # that the same parser applies to a column literally named `no`.
    retrieved: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    generated_with: GeneratedWith


CITATION = SourceBlock(
    benchmark="AutoML Benchmark (AMLB)",
    citation=(
        "Gijsbers, P., Bueno, M. L. P., Coors, S., LeDell, E., Poirier, S., Thomas, J., "
        "Bischl, B., & Vanschoren, J. (2024). AMLB: an AutoML Benchmark. Journal of Machine "
        "Learning Research, 25(101), 1-65."
    ),
    paper_url="https://jmlr.org/papers/v25/22-0493.html",
    openml_suite_id=271,
    openml_suite_name="AutoML Benchmark All Classification",
    amlb_repo="https://github.com/openml/automlbenchmark",
    amlb_ref=AMLB_REF,
    amlb_lists={
        "small": "resources/benchmarks/small.yaml",
        "medium": "resources/benchmarks/medium.yaml",
    },
    openml_data_api="https://www.openml.org/api/v1/json/data/{data_id}",
    openml_eval_api=(
        "https://www.openml.org/api/v1/json/evaluation/list/task/{task_id}/function/{function}"
    ),
    retrieved="2026-08-31",
    generated_with=GeneratedWith(python="", pandas="", scikit_learn=""),
)


class BaselineDefinition(BaseModel):
    """AMLB's calibration convention, recorded as a DEFINITION rather than as numbers.

    This is the field that says what `PipelineState.baseline_score` must be computed from. It
    carries no number on purpose: see the note, and the module docstring's rule 2.
    """

    model_config = ConfigDict(extra="forbid")

    definition_source: str
    zero_point: str
    unit_point: str
    note: str


BASELINE_DEFINITION = BaselineDefinition(
    definition_source="Gijsbers et al. 2024, framework score normalisation",
    zero_point="constant class-prior predictor",
    unit_point="tuned RandomForest",
    note=(
        "This is a definition, not a measurement. AMLB's own per-dataset per-framework scores "
        "live in an object store at openml1.win.tue.nl which presents a self-signed TLS "
        "certificate, so they cannot be re-fetched or re-verified from this repo and are not "
        "quoted here -- a number this repo cannot re-check is a number it will not publish. What "
        "IS quoted, per dataset, is `published_reference`: a real score from OpenML's own "
        "evaluation API, citable to a stable run id. That number is informational. It was "
        "produced on OpenML's 10-fold CV by another flow, so dividing this pipeline's "
        "verified_holdout_score by it would compare two protocols rather than two systems. "
        "`baseline_score` must instead be computed by fitting the two points above on this "
        "repo's own train split and scoring them on the same withheld holdout as the run."
    ),
)


class KnownLeak(BaseModel):
    """A leak somebody else documented in a real dataset, with the citation that documents it.

    Emphatically NOT ground truth, and `in_planted_leakage_columns` is a `Literal[False]` so that
    cannot change without a schema change and a reviewer noticing. `planted_leakage_columns` is
    read by `results_row()` as a COMPLETE list: everything flagged and not in it is scored a false
    alarm. Promoting one documented column into it would claim we know every leak in a real
    dataset, and would turn every other genuinely-suspicious column the reviewer names into a
    counted mistake. Recorded so a run can be read by a human, not scored by a machine.
    """

    model_config = ConfigDict(extra="forbid")

    column: str
    kind: str
    why: str
    source: str
    source_url: str
    in_planted_leakage_columns: Literal[False] = False


# Hand-authored, and here rather than in the manifest for the reason the module docstring gives:
# prose a person chose lives in `src/` where it is reviewed, so that everything in the generated
# file can be machine-derived. Keyed by our `dataset_id`.
#
# `bank_marketing`'s `duration` is the first non-synthetic, independently documented leak this
# project has had access to. Every leakage finding to date is on a fixture this repo wrote itself,
# which is the standing weakness of the whole thesis -- a reviewer that catches our traps may only
# be catching our habits.
KNOWN_LEAKAGE: dict[str, list[KnownLeak]] = {
    "bank_marketing": [
        KnownLeak(
            column="V12",
            kind="recorded_after_outcome",
            why=(
                "Last contact duration in seconds. It is not known before a call is placed, and "
                "the call's outcome is known the moment it ends, so a non-zero duration already "
                "implies the label. UCI says so explicitly. The column is V12 rather than "
                "`duration` because OpenML data 1461 ships anonymised headers (V1..V16), and the "
                "identification is measured rather than recalled: V12 is the twelfth feature, "
                "matching UCI's documented column order, and its observed range is 0 to 4918 -- "
                "4918 being exactly the duration maximum UCI documents for this dataset. The "
                "surrounding columns corroborate the same offset: V6 takes negative values "
                "(balance) and V14 bottoms out at the -1 sentinel (pdays). "
                "That the header is opaque makes this a BETTER probe, not a worse one -- the "
                "reviewer cannot catch it off the name and has to notice the column's behaviour."
            ),
            source=(
                "UCI Bank Marketing dataset description: 'this attribute highly affects the "
                "output target (e.g., if duration=0 then y=no). Yet, the duration is not known "
                "before a call is performed ... Thus, this input should only be included for "
                "benchmark purposes and should be discarded if the intention is to have a "
                "realistic predictive model.'"
            ),
            source_url="https://archive.ics.uci.edu/dataset/222/bank+marketing",
        )
    ],
}


class PublishedReference(BaseModel):
    """A real, citable score somebody else published on this OpenML task.

    Informational. `protocol` is a required field rather than a comment because it is the reason
    this cannot be `baseline_score`, and a reader who skips the module docstring should still hit
    the caveat on the value itself.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    openml_run_id: int
    openml_run_url: str
    flow_name: str
    metric: str
    value: float
    protocol: str


class DatasetEntry(BaseModel):
    """One external dataset: how to fetch it, what is in it, and what has been published about it.

    Deliberately NOT shaped like `Fixture`. If this were duck-substitutable, `cli._fixture_state`
    would accept it and build a run with `planted_leakage_columns=[]`, and every leakage column on
    the results row would score the reviewer as having missed a leak nobody planted. The divergent
    shape is the guardrail. A `Runnable` protocol over the two is a later session's job, and wants
    the harness-side re-scorer to exist first.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    openml_data_id: int
    # Recorded, never derived from `openml_data_id`. They are different namespaces that happen to
    # coincide for credit-g (data 31, task 31), which is exactly the coincidence that would teach
    # the next reader the wrong rule.
    openml_task_id: int
    openml_name: str
    openml_version: int
    openml_format: str
    openml_url: str
    md5_checksum: str
    amlb_list: str
    target: str
    task_type: str
    metric: str
    positive_class: str
    n_rows: int
    n_features: int
    n_usable_features: int
    positive_rate: float
    csv_sha256: str
    # This set carries no ground-truth leak list, and says so per entry rather than only in the
    # header, because the field is read by tests that would otherwise have to assume it.
    leakage_labelled: bool
    known_leakage: list[KnownLeak] = Field(default_factory=list)
    published_reference: PublishedReference | None = None

    @property
    def task_description(self) -> str:
        """Matches `Fixture.task_description`: what a person would say, not a spec."""
        return f"Predict {self.target} and report {self.metric}."


class ExcludedDataset(BaseModel):
    """A candidate the rule rejected, with the measurement that rejected it.

    Recorded so the selection is auditable rather than merely stated. "Why is `christine` not in
    here" should be answerable from the file, not from a session transcript.
    """

    model_config = ConfigDict(extra="forbid")

    openml_name: str
    openml_data_id: int
    openml_task_id: int
    amlb_list: str
    failed: str
    measured: dict[str, float | int | str]


class DatasetManifest(BaseModel):
    """What `evals/datasets/manifest.yaml` is allowed to contain.

    `extra="forbid"` throughout, for `FixtureManifest`'s stated reason: a stray key in a ground
    truth file must fail loudly rather than become a guarantee nothing reads.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    source: SourceBlock
    baselines: BaselineDefinition
    selection: SelectionRule
    datasets: list[DatasetEntry]
    excluded: list[ExcludedDataset] = Field(default_factory=list)


def load_manifest(path: Path = MANIFEST_PATH) -> DatasetManifest:
    """Parse and validate the manifest, or raise `SystemExit` with something actionable.

    `yaml.safe_load` only. `SystemExit` rather than an exception for `load_fixture`'s reason: the
    callers are entry points, and a traceback for "the manifest has not been generated" reads like
    a pipeline bug when it is a missing setup step.
    """
    if not path.exists():
        raise SystemExit(
            f"no benchmark manifest at {path}. It is a generated file -- run "
            f"`uv run ds-agents datasets refresh` to build it. See docs/PLAN.md Phase 4."
        )
    return DatasetManifest.model_validate(yaml.safe_load(path.read_text()))


def available(path: Path = MANIFEST_PATH) -> list[str]:
    """Every `dataset_id` in the manifest, sorted. Empty if it has not been generated yet.

    Empty rather than raising, so `--dataset` help text and the fixture/benchmark disjointness
    test work on a fresh checkout where nobody has run the fetch.
    """
    if not path.exists():
        return []
    return sorted(entry.dataset_id for entry in load_manifest(path).datasets)


def load_dataset(dataset_id: str, path: Path = MANIFEST_PATH) -> DatasetEntry:
    """Resolve one dataset by id, or raise `SystemExit` listing the ones that do exist."""
    manifest = load_manifest(path)
    for entry in manifest.datasets:
        if entry.dataset_id == dataset_id:
            return entry
    known = ", ".join(sorted(e.dataset_id for e in manifest.datasets)) or "(none)"
    raise SystemExit(
        f"no benchmark dataset named {dataset_id!r} in {path}. Known datasets: {known}."
    )


def cached_csv_path(entry: DatasetEntry, root: Path = DATASET_CACHE) -> Path:
    """Where the fetched CSV for this dataset lands. Gitignored; may not exist."""
    return root / entry.dataset_id / f"{entry.dataset_id}.csv"


def require_cached_csv(entry: DatasetEntry, root: Path = DATASET_CACHE) -> Path:
    """The fetched CSV for `entry`, or `SystemExit` naming the command that produces it.

    `cached_csv_path` answers "where would it be"; this answers "where is it", and the difference
    matters at a run's entry point. Without it a fresh checkout fails deep inside
    `ArtifactStore.register_dataset` with a `FileNotFoundError` on a path nobody typed, which reads
    like a broken pipeline rather than a missing download.

    A `csv_sha256` mismatch **warns and proceeds** rather than refusing. The hash is a function of
    pandas' float formatting, so a pandas bump changes it with nothing wrong; refusing would make
    every benchmark run in the repo unavailable on an upgrade. The warning still goes to stderr
    because the other cause of a mismatch -- a hand-edited or half-written cache file -- is one
    nobody should discover from a results table.
    """
    path = cached_csv_path(entry, root)
    if not path.exists():
        raise SystemExit(
            f"benchmark dataset {entry.dataset_id!r} is in the manifest but its CSV is not at "
            f"{path}. The cache is gitignored and rebuilt by fetching: "
            f"run `uv run ds-agents datasets refresh`."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry.csv_sha256:
        print(
            f"warning: {entry.dataset_id} csv_sha256 does not match the manifest "
            f"(cache {digest[:12]}..., manifest {entry.csv_sha256[:12]}...). Proceeding. If you "
            f"have not just upgraded pandas, re-run `ds-agents datasets refresh`.",
            file=sys.stderr,
        )
    return path
