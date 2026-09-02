"""The single object every node reads and writes.

Nothing in `nodes/` touches the filesystem, the network, or the environment. A node receives a
`PipelineState`, calls MCP tools, and returns a narrow dict of the fields it changed -- never the
state object itself, or the `operator.add` fields below concatenate the accumulated history onto
itself. If something is not on this object, a node cannot know it.

Two rules shape most of what follows, both of them consequences of the project's thesis:

1. The system under test does not get to report its own grade. Anything an agent claims is stored
   as a claim, and the harness records what it independently measured alongside it.
2. Anything the eval counts must be a field, never prose. If a metric can only be computed by
   string-matching a model's sentence, the metric is not real.
"""

import operator
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

ArtifactId = str


def _strip_computed(model: type[BaseModel], data: Any) -> Any:
    """Remove serialization-only computed fields from a dumped payload, recursively.

    `extra="forbid"` is load-bearing here: a node inventing a field must fail loudly. The cost is
    that `model_dump()` output cannot be fed straight back to `model_validate()`, because computed
    fields look like unexpected extras. This walks the declared field types and drops them.
    """
    if not isinstance(data, dict):
        return data
    computed = set(model.model_computed_fields)
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in computed:
            continue
        field = model.model_fields.get(key)
        out[key] = _strip_value(field.annotation, value) if field else value
    return out


def _strip_value(annotation: Any, value: Any) -> Any:
    """Descend into lists and optionals looking for nested Contract models."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _strip_computed(annotation, value)
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            if isinstance(value, list):
                return [_strip_computed(arg, item) for item in value]
            return _strip_computed(arg, value)
    return value


def utc_now() -> datetime:
    """Timezone-aware. Naive timestamps make committed JSONL inconsistent across machines."""
    return datetime.now(UTC)


NodeName = Literal[
    "intake",
    "profiler",
    "feature_eng",
    "modeler",
    "reviewer",
    "reporter",
    # The router writes `review_iterations` and `review_verdict`, which an edge function cannot
    # do, so it is a node and needs a name here or its cost and failures are unrecordable.
    "router",
    # The single-agent ablation arm runs one generalist instead of the team. It still has to
    # produce a valid trace, or the two arms are not comparable.
    "generalist",
]

RoutableNode = Literal["feature_eng", "modeler"]

TaskType = Literal["binary", "multiclass", "regression"]

Metric = Literal["roc_auc", "accuracy", "f1", "log_loss", "rmse", "mae", "r2"]

# Which direction counts as better. `holdout_claim_gap` and `baseline_normalised_score` are both
# meaningless without it: the same raw difference means "the agents overstated themselves" for
# roc_auc and "they understated themselves" for rmse.
GREATER_IS_BETTER: dict[str, bool] = {
    "roc_auc": True,
    "accuracy": True,
    "f1": True,
    "r2": True,
    "log_loss": False,
    "rmse": False,
    "mae": False,
}

# Kept small and enumerable on purpose: the eval counts objections by category, so an open-ended
# string here would make "did the reviewer catch the planted leak" unanswerable. `subcategory` is
# the escape hatch, so a reviewer catching something unanticipated is not forced to mislabel it.
ObjectionCategory = Literal[
    "leakage",
    "contamination",
    "overfit",
    "metric_mismatch",
    "implausible_importance",
    "spec_violation",
    "other",
]

# Categories whose whole meaning is "this column is the problem". An objection in one of these
# that names no column cannot be scored, so it is rejected at the contract.
COLUMN_SCOPED_CATEGORIES = frozenset({"leakage", "contamination", "implausible_importance"})

Severity = Literal["low", "medium", "high"]

# Model names that mean "no real client answered". A run whose trace contains one of these
# must never reach a results file; `PipelineState.publishable()` is the single gate.
PLACEHOLDER_MODEL_NAMES = frozenset({"stub"})

# How much of an error message `results_row` keeps. The full text stays on the state for whoever is
# debugging the run; the row is a line in a JSONL someone greps, and a model client's exception
# repr can carry an entire HTTP response body into it.
ERROR_MESSAGE_LIMIT = 500

ReviewVerdict = Literal["pending", "pass", "block", "exhausted"]

# Which reviewer system prompt the run used. A run condition for the same reason `naming` is:
# the reviewer's live miss is a confound between "cannot see the leak" and "was never asked which
# column produced the score", and a reviewer number that does not say which prompt produced it
# cannot separate them. `base` is the prompt every run before 2026-08-28 used, byte-identical.
ReviewerPrompt = Literal["base", "which_column"]
REVIEWER_PROMPTS: tuple[ReviewerPrompt, ...] = ("base", "which_column")

# How the graph decides WHICH NODE ACTS on an objection. Not what the reviewer said: that is
# `Objection.target_node`, which is recorded verbatim and never rewritten. `as_addressed` obeys it
# and is what every run committed before 2026-08-28 did. `by_category` gives any objection in
# COLUMN_SCOPED_CATEGORIES an effective target of `feature_eng` whatever the reviewer chose,
# because a column is the only thing feature_eng can act on and the modeler has no column lever at
# all -- every candidate is fit on the one transform feature_eng already froze. The evidence for
# needing the condition at all: across the 27 rows carrying `route_sequence`, 0 of the 21 runs that
# never routed to `feature_eng` remediated, against 3 of the 6 that did.
ObjectionRouting = Literal["as_addressed", "by_category"]
OBJECTION_ROUTINGS: tuple[ObjectionRouting, ...] = ("as_addressed", "by_category")

# Whether the reviewer was told what "done" looks like. The observed failure, live on 2026-08-28:
# across three diagnostic runs the reviewer dispositioned nothing `resolved`, and in the clearest
# one the pipeline dropped both planted columns, the claimed roc_auc fell 0.986 -> 0.823, the
# reviewer WROTE that the fall was consistent with removing leakage -- and held the objection open
# anyway, because the columns "were never validated as non-leaking, only removed". That is an
# unfalsifiable standard: a reviewer holding one can never let a run pass, so every run grinds to
# the cap and `exhausted` stops being evidence that the fix did not land. `on` appends one rule
# pointing at `final_features`, a field the reviewer is already shown.
#
# Its own axis rather than a third `ReviewerPrompt` value, because the two rules are independent
# conditions: bundling them would mean closure could never be measured without `which_column`
# attached, and the two effects could never be attributed separately.
ObjectionClosure = Literal["off", "on"]
OBJECTION_CLOSURES: tuple[ObjectionClosure, ...] = ("off", "on")

# Which disposition RELEASES a column that an objection forced out of the matrix. Under
# `resolved_or_withdrawn`, `binding_objections` collapses into `open_objections` exactly -- which is
# what `feature_eng._forced_drops` read before 2026-08-28: a `resolved` objection stopped forcing
# its drop, and the next return to that node, for any unrelated reason, put the leaked column back.
#
# THE DEFAULT IS DELIBERATELY NOT THE OLD BEHAVIOUR, and this is the only axis on RunConfig of which
# that is true. `naming`, `reviewer_prompt`, `objection_routing` and `objection_closure` all default
# to the arm that reproduces every committed row byte for byte, because each of those is a real
# design question with two defensible answers. This one is not: `resolved_or_withdrawn` is a defect,
# and it is here only because the fix for it is the largest single effect measured in this project
# (`leakage_remediated` 5/10 -> 9/10 across the code boundary at d5a9a28) and a same-commit control
# is the only way to confirm that. It is a defect-reproduction switch for one pre-registered cell
# and must never be the baseline of another arm. See DECISIONS.md 2026-08-28 (fifth entry).
ForcedDropRelease = Literal["withdrawn_only", "resolved_or_withdrawn"]
FORCED_DROP_RELEASES: tuple[ForcedDropRelease, ...] = ("withdrawn_only", "resolved_or_withdrawn")

Disposition = Literal["still_open", "resolved", "withdrawn", "not_reviewed"]


class Contract(BaseModel):
    """Base for every state object. Unknown fields are an error, not a silent pass."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def from_dump(cls, data: dict[str, Any]) -> Self:
        """Rebuild from `model_dump()` output. Use this to read a committed run back."""
        return cls.model_validate(_strip_computed(cls, data))


# The one data seed in the project. Named rather than inlined because two things now read it and
# they must agree: `RunConfig.random_seed`, which every node's snippet is handed, and
# `holdout.prepare`, which carves the withheld rows before any node runs. A carve at one seed and a
# split at another would not be wrong, but it would be two numbers where the file claims one.
DatasetSource = Literal["fixture", "benchmark"]
"""Which registry a run's dataset came from.

Declared here rather than in `runnable.py` because it is a field on the frozen `RunConfig`, and
this module must not import a dataset registry -- every node imports this module, and a node that
could reach a registry could read the answer key.
"""

RescoreStatus = Literal[
    "not_attempted",
    "no_withheld_holdout",
    "no_spec",
    "no_split",
    "no_feature_code",
    "empty_matrix",
    "no_model",
    "unknown_model_spec",
    "single_class_holdout",
    "snippet_failed",
    "sandbox_error",
    "refit_mismatch",
    "ok",
]
"""Why `verified_holdout_score` is or is not on this row.

An enum rather than a bool, and a column rather than a bare null, for the reason `leakage_graded`
exists: nine leakage columns used to read `None` for two unrelated reasons and a reader had to
guess which. A null score can mean the dataset had no withheld holdout at all (every fixture row),
that the graph produced no model to refit, or that the grader's sandbox died -- three facts with
completely different consequences for a table. `refit_mismatch` is the one value that carries a
score anyway: the number is kept because deleting it would hide the finding, and flagged because
pooling it would launder a bug into a result.

Nothing here ever appends a `PipelineError`. `errored` means the RUN went wrong; overloading it
with "the grader went wrong" is the defect NEXT.md already records against it.
"""

BaselineStatus = Literal[
    "not_attempted",
    "no_withheld_holdout",
    "rescore_unavailable",
    "empty_source_matrix",
    "single_class_train",
    "zero_point_failed",
    "unit_point_failed",
    "snippet_failed",
    "sandbox_error",
    "ok",
]
"""Why `baseline_zero_score` and `baseline_unit_score` are or are not on this row.

Its own enum rather than a widening of `RescoreStatus`, and this is the whole reason the baseline
runs in its own process: the yardstick can fail without the measurement failing. A RandomForest
that dies on a wide frame must not take `verified_holdout_score` with it, and if the two shared a
status column there would be no way to say so. `unit_point_failed` is the value that exists for
that case -- `baseline_zero_score` is KEPT on such a row, because the zero point was measured and
throwing it away would hide the fact that the scale has a floor but no ceiling.

`rescore_unavailable` is the coupling that does exist and is stated rather than hidden: the
baseline is fit on the agents' `split["train"]` and scored on the same withheld rows, so almost
every reason the re-scorer could not run is also a reason this could not. `baseline_detail` carries
which `rescore_status` it was.

Three things are deliberately NOT statuses. A degenerate scale -- the unit point level with or
below the zero point -- stays `ok`, because both points really were measured and that is a finding
about the dataset rather than a failure to measure; `baseline_normalised_score` returns `None` and
the two raw columns show why. A planted leak leaves both raw scores on the row untouched: only the
normalised column is suppressed, by the gate in `baseline_normalised_score`. And a scale that is
real but NARROW is not a status either -- `numerai28_6` measured both points honestly 0.0101 apart
and read 2.089 normalised, which is a fact about a near-chance dataset rather than a failure. That
one is why `baseline_separation` exists: the span goes on the row so a reader can see what the
normalised column was divided by, instead of a status flag asserting that they should not trust it.
"""

DEFAULT_RANDOM_SEED = 20260822

DEFAULT_LOOP_CAP = 3
"""How many review passes a run gets before the router gives up and reports.

Here rather than in FOUR places: `RunConfig`, `cli._run_once`, `harness.EvalCell`, and
`--loop-cap`'s own argparse default, which also wrote the literal into its help text. NEXT.md
recorded three; the argparse one was found while consolidating the other three. A sweep that
changed one silently left the rest on the old value -- and `loop_cap` is a recorded condition on
every results row, so a disagreement between them would not show up as a crash but as two cells
that claim the same condition and did not run under it.
"""

BASELINE_MIN_SEPARATION = 1e-9
"""The smallest `unit - zero` that `baseline_normalised_score` will divide by.

A divide-by-zero guard and nothing more. It is deliberately NOT a power criterion: deciding whether
a separation is large enough to be meaningful needs `n_withheld_rows` and belongs in `evaldiff`,
and a computed field that silently withholds numbers on a statistical test is worse than one that
publishes an unstable number next to the two inputs a reader can check it against.
"""


class RunConfig(Contract):
    """What this run IS. Frozen at construction, so no node can rewrite its own conditions.

    Every results row must be self-describing from the state object alone. Without this, the
    reviewer-off arm is byte-identical to a run where the reviewer crashed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    arm: Literal["team", "single_agent"] = "team"
    reviewer_enabled: bool = True
    reviewer_model: str = "haiku"
    default_model: str = "haiku"
    loop_cap: int = Field(default=DEFAULT_LOOP_CAP, ge=0)
    reviewer_sees_code: bool = Field(
        default=True,
        description="Whether the reviewer may read the feature engineering code, not just its "
        "outputs. Starts on; becomes an ablation later.",
    )
    naming: Literal["descriptive", "opaque"] = Field(
        default="descriptive",
        description="Whether the agents saw the fixture's real column names or `var_NN`. The "
        "profiler nominates leakage largely off column names, so this sets trap difficulty more "
        "than any statistical property of the data does. It lives here, on the frozen config, "
        "because a leakage number without it is not interpretable and the two arms are otherwise "
        "byte-identical -- see `ds_agents.naming`.",
    )
    reviewer_prompt: ReviewerPrompt = Field(
        default="base",
        description="Which system prompt the reviewer ran under: `base`, byte-identical to every "
        "run before 2026-08-28, or `which_column`, which appends one rule asking the reviewer to "
        "name the column that explains an implausible score. On the frozen config because a "
        "reviewer-model comparison that does not record the prompt is confounded by it.",
    )
    objection_routing: ObjectionRouting = Field(
        default="as_addressed",
        description="Who the graph asks to act on an objection: `as_addressed`, obeying the "
        "reviewer's own `target_node` exactly as every run before 2026-08-28 did, or "
        "`by_category`, which routes a column-scoped objection to `feature_eng` regardless of "
        "what the reviewer wrote. On the frozen config for the same reason `naming` and "
        "`reviewer_prompt` are: the two arms remediate at different rates and a row that did not "
        "carry this would be averaged with rows from the other. Note it changes what `feature_eng` "
        "and `modeler` are SHOWN as well as where the run goes -- both read "
        "`open_objections(target)` -- so it is not a pure edge change.",
    )
    objection_closure: ObjectionClosure = Field(
        default="off",
        description="Whether the reviewer was given a termination condition it can check: `off`, "
        "byte-identical to every prompt before 2026-08-28, or `on`, which appends one rule saying "
        "an objection about a column is answered when that column is absent from "
        "`final_features`. On the frozen config for the same reason every other condition is, and "
        "with a specific hazard of its own: a prompt that buys termination by teaching the "
        "reviewer to say 'fixed' is worse than no prompt, so `objections_falsely_resolved` is on "
        "the results row beside it.",
    )
    forced_drop_release: ForcedDropRelease = Field(
        default="withdrawn_only",
        description="Which disposition releases a column that an objection forced out of the "
        "matrix: `withdrawn_only`, the correct rule and the default, or `resolved_or_withdrawn`, "
        "which reproduces the pre-2026-08-28 defect where a `resolved` objection stopped forcing "
        "its drop and the next return to feature_eng put the leaked column back. THE ONLY "
        "condition here whose default is not the old behaviour, because the old behaviour is a bug "
        "and not a design alternative. Defect reproduction only: it exists so the largest effect "
        "in the project has a same-commit control, it is not a general ablation lever, and no "
        "other arm may use it.",
    )
    holdout_fraction: float = Field(
        default=0.0,
        ge=0.0,
        lt=0.5,
        description="How much of the dataset was withheld from the agents before the graph "
        "started. "
        "0.0 for every fixture, by decision rather than omission -- carving rows out of a 200-row "
        "toy would change what the agents see and make all 145 committed rows incomparable, for no "
        "gain, because planted leakage is a column and a random holdout still contains it.",
    )
    dataset_source: DatasetSource = Field(
        default="fixture",
        description="Which registry the dataset came from. A fixture has a complete planted answer "
        "key; a benchmark dataset has none and is graded on a withheld holdout instead. The two "
        "are graded by different columns, so a row that did not say which it was would invite "
        "pooling a leakage rate with a score gap.",
    )
    random_seed: int = DEFAULT_RANDOM_SEED
    dataset_hash: str | None = Field(
        default=None,
        description="sha256 of the CSV the agents were actually mounted -- the materialised, "
        "post-carve file, not the source on disk. Two rows under the same `dataset_id` and "
        "`naming` that disagree here saw different bytes, which is the one difference no other "
        "field on this config can express.",
    )
    commit: str | None = Field(
        default=None,
        description="Short git hash of the tree that produced this run, `-dirty` suffixed when the "
        "working tree was not clean. Null when nothing recorded it. On the frozen config for the "
        "same reason `naming` and `loop_cap` are: which code ran is a run condition no other field "
        "carries, and every code boundary this project has had to reason about so far -- the "
        "sticky-drop fix, the naming ablation's schema change, the block-retry -- had to be "
        "reconstructed from commit messages after the fact. A dirty tree groups as its own cell in "
        "`eval-diff`, which is correct: a dirty run is not reproducible.",
    )


class TaskSpec(Contract):
    """What the run is trying to do. Written by intake, read by everyone after it."""

    target: str
    task_type: TaskType
    metric: Metric
    split_strategy: Literal["random", "stratified", "temporal", "grouped"] = "stratified"
    split_key: str | None = Field(
        default=None,
        description="Column used by temporal or grouped splits. Required for those strategies.",
    )
    positive_class: str | None = None

    @computed_field
    @property
    def greater_is_better(self) -> bool:
        return GREATER_IS_BETTER[self.metric]

    @model_validator(mode="after")
    def _split_key_required_for_keyed_strategies(self) -> "TaskSpec":
        if self.split_strategy in {"temporal", "grouped"} and not self.split_key:
            raise ValueError(f"split_strategy={self.split_strategy!r} requires split_key")
        return self


class LeakageCandidate(Contract):
    """A column the profiler thinks may encode the target.

    The profiler only flags. Deciding what to do about it is the reviewer's job, and whether the
    reviewer agrees is exactly what the eval measures.
    """

    column: str
    reason: str
    evidence: str = Field(description="The number or observation behind the flag, not a vibe.")
    suspicion: Severity = Field(
        description="Ordinal on purpose. A float from an uncalibrated model is fake precision."
    )


class ColumnProfile(Contract):
    name: str
    dtype: str
    missing_fraction: float = Field(ge=0.0, le=1.0)
    n_unique: int = Field(ge=0)
    sample_values: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Capped: this renders into every prompt that shows the profile.",
    )


class ProfileReport(Contract):
    """Written by profiler. The reviewer reads this to argue with the modeler."""

    n_rows: int = Field(ge=0)
    n_columns: int = Field(ge=0)
    columns: list[ColumnProfile] = Field(default_factory=list)
    leakage_candidates: list[LeakageCandidate] = Field(default_factory=list)
    target_balance: dict[str, float] = Field(default_factory=dict)


class ModelResult(Contract):
    name: str
    # Deliberately untyped values: real estimator params include lists and nested dicts, and with
    # extra="forbid" a ValidationError mid-run costs the whole eval row. Nothing reads this.
    params: dict[str, Any] = Field(default_factory=dict)
    cv_scores: list[float] = Field(default_factory=list)
    claimed_holdout_score: float | None = Field(
        default=None,
        description="What the modeler says it scored. A claim, not a measurement. The harness "
        "writes the independent number to PipelineState.verified_holdout_score.",
    )
    model_artifact: ArtifactId | None = None

    @computed_field
    @property
    def cv_mean(self) -> float | None:
        return sum(self.cv_scores) / len(self.cv_scores) if self.cv_scores else None


class Objection(Contract):
    """A blocking complaint from the reviewer, aimed at one node.

    Strictly append-only: once raised, an Objection is never edited or removed. Whether it was
    later accepted, dropped, or silently forgotten is recorded in `ReviewPass.dispositions`, so
    that "the reviewer withdrew it" stays distinguishable from "the reviewer never looked again."
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    category: ObjectionCategory
    subcategory: str = Field(
        description="Free text, so a reviewer catching something unanticipated is not forced to "
        "mislabel it as spec_violation. Counted by category; read by humans."
    )
    target_node: RoutableNode
    columns: list[str] = Field(
        default_factory=list,
        description="The columns at issue. Required for column-scoped categories, because "
        "leakage_caught is a set comparison against ground truth, not a substring search.",
    )
    evidence: str
    severity: Severity
    raised_at_iteration: int = Field(ge=0)

    @model_validator(mode="after")
    def _column_scoped_categories_need_columns(self) -> "Objection":
        if self.category in COLUMN_SCOPED_CATEGORIES and not self.columns:
            raise ValueError(f"category={self.category!r} requires at least one column")
        return self


class ReviewPass(Contract):
    """One completed reviewer invocation.

    This is what makes silence legible. Without a per-pass disposition for every open objection,
    "feature_eng fixed it" and "the reviewer forgot about it" are the same absence.
    """

    iteration: int = Field(ge=0)
    claim: Literal["pass", "block"] = Field(
        description="What the reviewer asked for. Not the outcome: the router decides that."
    )
    routed_to: RoutableNode | Literal["reporter"]
    dispositions: dict[str, Disposition] = Field(
        default_factory=dict,
        description="Keyed by Objection.id, for every objection open on entry.",
    )
    new_objection_ids: list[str] = Field(default_factory=list)


class PipelineError(Contract):
    node: NodeName
    message: str
    recoverable: bool = True
    occurred_at: datetime = Field(default_factory=utc_now)


class NodeEvent(Contract):
    """One node execution.

    Appended by every node so results files do not depend on the tracing backend.
    """

    node: NodeName
    started: datetime
    ended: datetime | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    model: str | None = None

    @computed_field
    @property
    def wall_seconds(self) -> float | None:
        if self.ended is None:
            return None
        return (self.ended - self.started).total_seconds()


class PipelineState(Contract):
    """Every node reads and writes this. Nothing else.

    The four list fields carrying history are annotated with `operator.add` so LangGraph appends
    partial updates instead of replacing them. Without that, a node returning `{"node_trace": [e]}`
    silently discards every earlier event and the cost table becomes fiction.
    """

    # run identity and conditions
    config: RunConfig = Field(default_factory=RunConfig)
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None

    # intake
    dataset_id: str
    task_description: str
    spec: TaskSpec | None = None

    # profiler
    profile: ProfileReport | None = None
    split_artifact: ArtifactId | None = Field(
        default=None,
        description="Row-id manifest for train/holdout/folds. Pinned before feature_eng runs and "
        "never rewritten, or contamination objections are unfalsifiable and runs are not "
        "reproducible.",
    )

    # feature_eng
    feature_code_artifact: ArtifactId | None = None
    feature_summary: str | None = None
    final_features: list[str] | None = Field(
        default=None,
        description="Columns that survived into the matrix the model was fit on. This is what "
        "makes 'the reviewer caught it' and 'the leak was removed' two different numbers. SOURCE "
        "column names, never one-hot expansions: `results_row()` intersects this with "
        "`planted_leakage_columns`, which are source names, so storing 'colour=red' here would "
        "empty the intersection and score every run as remediated.",
    )
    dropped_features: list[str] = Field(default_factory=list)
    skipped_high_cardinality: list[str] = Field(
        default_factory=list,
        description="Columns dropped because they have more distinct values than the one-hot "
        "encoder will expand. A recorded DECISION, not a `PipelineError`: it is what the node is "
        "supposed to do at that cardinality, nothing downstream is degraded by it, and recording "
        "it in `errors` made `errored` -- which is `bool(self.errors)` -- read true on four "
        "completely healthy `adult` runs, so any table using `errored` as a rate scored that "
        "cell as a 100% failure. SOURCE column names, and a subset of `dropped_features`.",
    )

    # modeler
    candidates: list[ModelResult] = Field(default_factory=list)
    chosen_model: ModelResult | None = None
    importance_artifact: ArtifactId | None = Field(
        default=None,
        description="Permutation importances for every candidate, computed on the agents' holdout "
        "with the feature transform inside the estimator pipeline so the names are source columns. "
        "Named for what it holds: `shap` is not a dependency and this has never held SHAP values.",
    )
    top_importances: list[tuple[str, float]] = Field(
        default_factory=list,
        description="The chosen model's permutation importances, source column names, highest "
        "first. Source names because `planted_leakage_columns` and `Objection.columns` speak that "
        "vocabulary, and a set comparison across two vocabularies is not a comparison.",
    )

    # reviewer
    review_iterations: int = Field(
        default=0,
        ge=0,
        description="Completed reviewer invocations. Incremented by the router, never by the "
        "reviewer node, so the model under test cannot exceed its own cap.",
    )
    objections: Annotated[list[Objection], operator.add] = Field(default_factory=list)
    review_passes: Annotated[list[ReviewPass], operator.add] = Field(default_factory=list)
    reviewer_claim: Literal["pass", "block"] | None = Field(
        default=None,
        description="The reviewer's own last word. Input to the verdict, not the verdict.",
    )
    reviewer_dispositions: dict[str, Disposition] = Field(
        default_factory=dict,
        description="Handoff, not history: the reviewer's raw disposition-per-objection-id from "
        "its most recent pass, overwritten wholesale on every pass (deliberately not "
        "`operator.add`). The router is the sole reader -- it folds this into the `ReviewPass` it "
        "mints and never clears it. `ReviewPass.dispositions` is the durable record; this field is "
        "just how the disposition gets from the reviewer to the router within one invocation.",
    )
    review_verdict: ReviewVerdict = Field(
        default="pending",
        description="Derived by the router. 'exhausted' means the reviewer still wanted to block "
        "when it ran out of loops, which no node is permitted to self-certify.",
    )

    # reporter
    report_artifact: ArtifactId | None = None

    # harness-written ground truth. Nodes never set these; they are how the run gets graded.
    planted_leakage_columns: list[str] = Field(default_factory=list)
    verified_holdout_score: float | None = Field(
        default=None,
        description="Scored by the harness on a holdout the agents never see. The gap against "
        "chosen_model.claimed_holdout_score is itself a finding.",
    )
    rescore_status: RescoreStatus = Field(
        default="not_attempted",
        description="Why verified_holdout_score is or is not present. Never an error on the run.",
    )
    rescore_detail: str = Field(
        default="",
        description="Free text for the statuses that have something to say -- a snippet's last "
        "lines, or the two numbers behind a refit_mismatch. Never parsed.",
    )
    refit_claim_gap: float | None = Field(
        default=None,
        description="The self-check that earns the withheld number: the harness refit scored on "
        "the AGENTS' own holdout, minus what the modeler claimed on it. Anything but ~0 means the "
        "refit is not the model that produced the claim, and the withheld score is measuring "
        "something else.",
    )
    n_withheld_rows: int | None = Field(
        default=None,
        description="How many rows verified_holdout_score was measured on. Without it a gap of "
        "0.05 on 200 rows is indistinguishable from one on 20000.",
    )
    baseline_zero_score: float | None = Field(
        default=None,
        description="AMLB's zero point -- a constant class-prior predictor, fit on the agents' "
        "train split and scored on the same withheld rows as the run. For roc_auc this is exactly "
        "0.5 by construction, which makes it a correctness assertion on the grader as well as a "
        "column.",
    )
    baseline_unit_score: float | None = Field(
        default=None,
        description="AMLB's unit point in convention only: a RandomForest on the raw columns, "
        "fit and scored on the same rows as the zero point. The recipe is OURS -- see "
        "rescore.BASELINE_SPECS and baseline_recipe -- because AMLB's own grid cannot be "
        "re-fetched from this repo.",
    )
    baseline_status: BaselineStatus = Field(
        default="not_attempted",
        description="Why the two baseline points are or are not present. Independent of "
        "rescore_status on purpose: the yardstick can fail without the measurement failing.",
    )
    baseline_detail: str = Field(
        default="",
        description="Free text for the statuses that have something to say -- a snippet's last "
        "lines, or which rescore_status blocked the baseline. Never parsed.",
    )
    baseline_recipe: str = Field(
        default="",
        description="Version string for the unit point's recipe, on the ROW so that a change to "
        "it is visible in the data and not only in git. Two rows carrying different values must "
        "never be pooled into one normalised distribution.",
    )
    rescore_seconds: float | None = Field(
        default=None,
        description="Wall seconds the re-scorer spent. NOT part of wall_seconds, which stops when "
        "the graph returns -- the grader runs after it. None means the re-scorer never ran, which "
        "is different from 0.0.",
    )
    baseline_seconds: float | None = Field(
        default=None,
        description="Wall seconds the two baseline points spent, separate from rescore_seconds "
        "because they are a separate process with a separate timeout and can fail alone. This is "
        "the term the yardstick's wall cost lives in -- 0.2s at credit_g's shape and about a "
        "minute at the manifest's largest -- and that nothing recorded until 2026-09-01.",
    )

    # bookkeeping
    errors: Annotated[list[PipelineError], operator.add] = Field(default_factory=list)
    node_trace: Annotated[list[NodeEvent], operator.add] = Field(default_factory=list)

    @computed_field
    @property
    def total_cost_usd(self) -> float:
        return sum(event.cost_usd for event in self.node_trace)

    @computed_field
    @property
    def loop_exhausted(self) -> bool:
        return self.review_iterations >= self.config.loop_cap

    @computed_field
    @property
    def wall_seconds(self) -> float | None:
        """Real elapsed time, including graph and tool overhead that node events miss."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def node_seconds(self) -> dict[str, float]:
        """Wall seconds per node, summed over repeats within the run.

        A plain property rather than a `computed_field`: this is a results-row column, not part of
        the state contract other nodes read, and every node already carries its own `NodeEvent`.
        Summed rather than listed because the question it answers is "where did the wall time go at
        98k rows", and a node that took the review cycle three times spent all three.

        An event with no `ended` contributes nothing rather than raising -- a run that died mid-node
        still has a row worth writing, and a missing end is already recorded as an error.
        """
        totals: dict[str, float] = {}
        for event in self.node_trace:
            seconds = event.wall_seconds
            if seconds is not None:
                totals[event.node] = round(totals.get(event.node, 0.0) + seconds, 3)
        return totals

    @computed_field
    @property
    def baseline_separation(self) -> float | None:
        """How long the baseline scale is: `unit - zero`, sign-corrected so bigger is always wider.

        The denominator `baseline_normalised_score` divides by, published as its own column so a
        reader can see it. Without it a normalised score is uncheckable: `numerai28_6` returns
        2.089 not because the run was extraordinary but because the two reference points are
        0.0101 apart on a near-chance dataset, and nothing on the row said so.

        Deliberately NOT a suppression and NOT a `baseline_status` value. `BASELINE_MIN_SEPARATION`
        already records why -- withholding a number on a statistical test is worse than publishing
        an unstable one beside the inputs a reader can check it against -- and a status value would
        overload a column whose job is why the two raw scores ARE or ARE NOT here, when on a narrow
        scale both were measured perfectly well.

        Not suppressed on a planted leak either, unlike the normalised score. The two raw points
        stay on such a row because they are honest measurements, and the distance between them is a
        property of the dataset and the recipe rather than of the run's grade.
        """
        if self.spec is None:
            return None
        if self.baseline_zero_score is None or self.baseline_unit_score is None:
            return None
        raw = self.baseline_unit_score - self.baseline_zero_score
        return raw if self.spec.greater_is_better else -raw

    @computed_field
    @property
    def baseline_normalised_score(self) -> float | None:
        """Where this run sits on the scale from a constant predictor to a RandomForest.

        `(verified - zero) / (unit - zero)`. 0.0 means the run did no better than predicting the
        class prior; 1.0 means it matched the RandomForest; above 1.0 means it beat it. This is
        AMLB's normalisation and NOT a ratio -- `verified / baseline` disagrees with it about what
        1.0 means, and on a metric whose floor is not zero it is meaningless. r2 is the example
        that settles it: a predict-the-train-mean baseline scores slightly NEGATIVE on a holdout,
        not 0.0, because the r2 denominator is the holdout's variance about its own mean.

        Written as a difference the formula is already direction-invariant -- for a lower-is-better
        metric both differences flip sign together and the quotient is unchanged -- so the explicit
        branch below buys only the sign guard, and `tests/test_state.py` pins that the two forms
        agree so a future simplification cannot reintroduce the bug the ratio had.

        Never raises, for the reason the retired `score_ratio` gave: a computed field that raises
        takes `model_dump_json()` and `results_row()` down with it, and ARCHITECTURE.md requires a
        row for every dataset even on hard failure.
        """
        if self.verified_holdout_score is None or self.spec is None:
            return None
        # Implied by `baseline_separation` being None below, and restated because the numerator
        # subtracts `baseline_zero_score` directly and a reader should not have to follow a
        # property into another property to see that it cannot be None there.
        if self.baseline_zero_score is None or self.baseline_unit_score is None:
            return None
        # SUPPRESSED ON A PLANTED LEAK, and this is the pooling hazard's whole resolution. The
        # baseline is fit on every raw column, including the trap the pipeline was supposed to
        # drop. On a labelled dataset a pipeline that correctly drops it therefore scores BELOW a
        # baseline that kept it -- so a value under 1 would be evidence of GOOD behaviour there and
        # of BAD behaviour on an unlabelled dataset, the same number meaning opposite things with
        # nothing on the row to separate them. Gated exactly as the nine leakage columns are gated
        # on `graded_for_leakage`, for the mirror-image reason. Both raw points STAY on the row:
        # they are honest measurements, and a reader who knows about the trap can use them.
        if self.planted_leakage_columns:
            return None
        separation = self.baseline_separation
        # `<=`, not `abs(...) < eps`. A NEGATIVE separation means the RandomForest did worse than
        # the class prior, which inverts the scale: a run that beat the prior would come out
        # negative and a reader would take that for "worse than the prior". Reachable rather than
        # hypothetical -- f1 with a minority positive class has a zero point of exactly 0.0, and a
        # dataset with no signal at all separates the two points by noise in either direction.
        if separation is None or separation <= BASELINE_MIN_SEPARATION:
            return None
        numerator = self.verified_holdout_score - self.baseline_zero_score
        return (numerator if self.spec.greater_is_better else -numerator) / separation

    def objected_columns(
        self,
        categories: frozenset[str] = COLUMN_SCOPED_CATEGORIES,
        standing_only: bool = False,
    ) -> set[str]:
        """Columns the reviewer named, for the given categories.

        `standing_only=False` means "ever raised", which is what `leakage_caught` asks: did the
        reviewer surface this column at any point. `standing_only=True` drops columns whose every
        objection was later resolved or withdrawn, which is what a false-alarm count should ask:
        a reviewer that takes back a bad flag is behaving better than one that does not, and
        scoring them identically hides that.
        """
        source = self.open_objections() if standing_only else self.objections
        return {c for o in source if o.category in categories for c in o.columns}

    def results_row(self) -> dict[str, Any]:
        """The flat row `harness.py` writes to results JSONL. One per dataset per run.

        Every eval outcome named in docs/ARCHITECTURE.md is computed here, from fields only. If a
        number in the published tables cannot be traced to this method, it is not a real number.
        """
        leak_categories = frozenset({"leakage", "contamination"})
        planted = set(self.planted_leakage_columns)
        # `planted` is a COMPLETE ground-truth list or it is nothing at all.
        #
        # On a fixture it is complete by construction -- `generate.py` writes the manifest at the
        # moment it writes the CSV. On an external benchmark dataset there is no such list, and
        # nobody has enumerated the leaks in `adult` or `nomao`; `evals/datasets/manifest.yaml`
        # says so per entry with `leakage_labelled: false`. Left ungated, an empty `planted` makes
        # eight columns below assert two things this project has no evidence for at once: that the
        # dataset contains no leak, and that the reviewer failed to find it. `leakage_caught` would
        # read False, and every column the reviewer flagged would be counted a false alarm.
        #
        # So they report `None` -- not measured -- exactly as `leakage_remediated` and the three
        # `*_recall` columns already do on the same reasoning. `evaldiff` excludes `None` metrics
        # from its denominators, so a dataset with no answer key drops out of a rate rather than
        # dragging it down.
        #
        # Deliberately derived rather than stored: a `leakage_ground_truth` field on the state
        # would be a second source of truth for something `planted` already says.
        graded_for_leakage = bool(planted)
        flagged = self.objected_columns(leak_categories)
        standing = self.objected_columns(leak_categories, standing_only=True)
        true_positives = planted & flagged
        false_positives = flagged - planted

        # An empty feature matrix is not remediation. A feature_eng crash leaves
        # `final_features=[]`, which trivially contains no planted column; counting that as a fix
        # inflates the headline remediation rate with runs that produced nothing.
        remediated: bool | None = None
        if planted and self.final_features:
            remediated = not (planted & set(self.final_features))

        # The profiler's nominations, scored separately from the reviewer's objections. They are
        # different questions with different answers: on the trap fixtures the profiler nominates
        # the planted column and the reviewer, shown the same run, says nothing. Folding them into
        # one `leakage_caught` would report a team that catches leaks while hiding which member
        # caught it, and the name-transparency arm moves this number and not the reviewer's.
        nominated: set[str] | None = None
        if self.profile is not None:
            nominated = {c.column for c in self.profile.leakage_candidates}

        # The reviewer's columns over ALL column-scoped categories, `implausible_importance`
        # included. `flagged` above stays a two-category number on purpose: the committed
        # naming-ablation rows were written under that definition, and widening it would silently
        # redefine the only published results file. `None` when no pass completed: the
        # reviewer-off arm still runs the node as a no-op, and a 0.0 recall from a reviewer that
        # never looked would average in with one that looked and declined.
        objected: set[str] | None = None
        if self.config.reviewer_enabled and self.review_passes:
            objected = self.objected_columns()
        by_category = dict.fromkeys(get_args(ObjectionCategory), 0)
        for objection in self.objections:
            by_category[objection.category] += 1

        # Why a run that caught the trap still shipped it. `objections_by_category` says what the
        # reviewer objected to; these four say whether anything could act on it. They are derived
        # here rather than recorded by the nodes for the same reason every other outcome is: a node
        # that wrote down its own remediation would be a node reporting its own score.
        # `target_node` is read RAW here, deliberately, and `effective_target` is only used to
        # count the disagreements. This is what makes `objection_routing="by_category"` a recorded
        # condition rather than a thumb on the scale: the reviewer's own dispatch judgement stays
        # measurable in the arm that overrides it, and `objections_rerouted` says how often the
        # graph disagreed with it. Reading the effective target into this counter instead would
        # delete the only evidence that the override was ever needed.
        by_target_node = dict.fromkeys(get_args(RoutableNode), 0)
        rerouted = 0
        for objection in self.objections:
            by_target_node[objection.target_node] += 1
            if self.effective_target(objection) != objection.target_node:
                rerouted += 1

        # The caught-versus-remediated gap as a list of names. A column-scoped objection whose
        # column is still in the matrix at the end was raised and not acted on, whatever the
        # verdict says. `None` in two cases, and both are the same distinction the fields above
        # draw: when feature_eng produced nothing (an empty matrix is not a clean one, matching
        # `leakage_remediated`), and when no reviewer pass completed (the reviewer-off arm runs
        # the node as a no-op, and an empty list there would read as "objected and remediated"
        # rather than "never objected", which is the opposite finding).
        unremediated: list[str] | None = None
        if objected is not None and self.final_features:
            unremediated = sorted(objected & set(self.final_features))

        # Closure, as something measured rather than hoped for. `objections_open_at_end` says how
        # many were never closed; these say HOW the closed ones closed and whether the closure was
        # honest. `resolved` and `withdrawn` are split because they are opposite claims about the
        # reviewer -- one says the fix landed, the other says the objection was wrong -- and
        # because `binding_objections` now acts on that difference, so a row that conflated them
        # could not be used to reason about what feature_eng actually dropped. No committed row
        # before 2026-08-28 splits them, which is why the sticky-drop screen of those rows could
        # not be resolved past "at risk".
        #
        # Gated exactly as `reviewer_nominated` and friends are. A reviewer that ran and closed
        # nothing is a REAL 0, not a `None`: that 0 is the entire pre-closure finding, and
        # collapsing it into "not measured" would delete the control's result.
        latest = self.latest_dispositions()
        by_id = {o.id: o for o in self.objections}
        n_resolved: int | None = None
        n_withdrawn: int | None = None
        falsely_resolved: int | None = None
        if self.config.reviewer_enabled and self.review_passes:
            n_resolved = sum(1 for d in latest.values() if d == "resolved")
            n_withdrawn = sum(1 for d in latest.values() if d == "withdrawn")
            # The failure mode `objection_closure="on"` creates and nothing before it could: an
            # objection marked `resolved` whose column is still in the matrix. A prompt that buys
            # termination by teaching the reviewer to say "fixed" is worse than no prompt, and
            # this is the only column that would catch it. Counted over objections rather than
            # columns because the unit being scored is the reviewer's judgement act, and over
            # column-scoped categories only -- a `resolved` `overfit` objection names no column
            # and cannot be checked this way.
            #
            # `None` on an empty matrix for the same reason as `leakage_remediated`: an empty
            # matrix contains no column, so every resolution would score honest, and the
            # inflation would flatter the arm under test.
            if self.final_features:
                final = set(self.final_features)
                falsely_resolved = sum(
                    1
                    for oid, disposition in latest.items()
                    if disposition == "resolved"
                    and (objection := by_id.get(oid)) is not None
                    and objection.category in COLUMN_SCOPED_CATEGORIES
                    and set(objection.columns) & final
                )

        passes = sorted(self.review_passes, key=lambda rp: rp.iteration)

        claimed = self.chosen_model.claimed_holdout_score if self.chosen_model else None
        gap: float | None = None
        if claimed is not None and self.verified_holdout_score is not None:
            # Signed so that positive always means "the agent overstated itself", whichever
            # direction the metric runs. Unsigned, an overclaim on rmse and one on roc_auc have
            # opposite signs and cancel to nothing when averaged over a mixed benchmark.
            raw = claimed - self.verified_holdout_score
            gap = raw if (self.spec is None or self.spec.greater_is_better) else -raw

        return {
            # what this run was
            "run_id": self.config.run_id,
            "dataset_id": self.dataset_id,
            "arm": self.config.arm,
            "reviewer_enabled": self.config.reviewer_enabled,
            "default_model": self.config.default_model,
            "reviewer_model": self.config.reviewer_model,
            "reviewer_prompt": self.config.reviewer_prompt,
            "reviewer_sees_code": self.config.reviewer_sees_code,
            "naming": self.config.naming,
            "loop_cap": self.config.loop_cap,
            "objection_routing": self.config.objection_routing,
            "objection_closure": self.config.objection_closure,
            "forced_drop_release": self.config.forced_drop_release,
            "random_seed": self.config.random_seed,
            "commit": self.config.commit,
            "dataset_source": self.config.dataset_source,
            "holdout_fraction": self.config.holdout_fraction,
            "dataset_hash": self.config.dataset_hash,
            # scores. `claimed` is what the agent said; `verified` is what we measured.
            "claimed_holdout_score": claimed,
            "verified_holdout_score": self.verified_holdout_score,
            "holdout_claim_gap": gap,
            "metric": self.spec.metric if self.spec else None,
            # Why the two columns above do or do not carry a number, and what the number was
            # measured on. `rescore_status` is to the score what `leakage_graded` is to the nine
            # leakage columns: the reason a null is null, said outright instead of inferred.
            "rescore_status": self.rescore_status,
            "rescore_detail": self.rescore_detail[:ERROR_MESSAGE_LIMIT],
            "refit_claim_gap": self.refit_claim_gap,
            "n_withheld_rows": self.n_withheld_rows,
            # The scale `verified_holdout_score` is read against: a constant class-prior
            # predictor at 0 and a RandomForest at 1, both fit on the agents' train split and
            # scored on the same withheld rows. Published as two raw points plus the
            # normalisation, rather than as one `baseline_score`, because a single number cannot
            # say which end of the scale it is -- and because the normalised column is suppressed
            # on a dataset with a planted leak while the two raw points are not.
            "baseline_zero_score": self.baseline_zero_score,
            "baseline_unit_score": self.baseline_unit_score,
            "baseline_normalised_score": self.baseline_normalised_score,
            "baseline_separation": self.baseline_separation,
            "baseline_status": self.baseline_status,
            "baseline_detail": self.baseline_detail[:ERROR_MESSAGE_LIMIT],
            "baseline_recipe": self.baseline_recipe,
            # leakage, as a set comparison against ground truth
            #
            # Stated outright rather than left to be inferred from nine separate `None`s. This is
            # the same defect docs/NEXT.md already records against `errored`, whose two meanings
            # can only be separated by string-matching an error prefix -- a column whose absence
            # of a value carries information needs a companion that says so. A reader pooling
            # `leakage_caught` across a results file can filter on this instead of guessing why a
            # null is null.
            "leakage_graded": graded_for_leakage,
            "leakage_planted": sorted(planted),
            "leakage_flagged": sorted(flagged),
            "leakage_caught": bool(true_positives) if graded_for_leakage else None,
            "leakage_remediated": remediated,
            "leakage_recall": len(true_positives) / len(planted) if planted else None,
            "leakage_precision": (
                len(true_positives) / len(flagged) if graded_for_leakage and flagged else None
            ),
            "false_alarm_columns": sorted(false_positives) if graded_for_leakage else None,
            "false_alarm": len(false_positives) if graded_for_leakage else None,
            "leakage_flagged_standing": sorted(standing),
            "false_alarm_standing": len(standing - planted) if graded_for_leakage else None,
            # How wide the matrix the run actually shipped is. `leakage_remediated` is None on an
            # EMPTY matrix but True on a one-column one, so without this a run that remediated by
            # force-dropping most of the fixture is indistinguishable from one that dropped only
            # the trap. `objection_routing="by_category"` turns a reviewer false positive into a
            # real dropped column, so this is the price tag on that arm.
            "n_final_features": (
                len(self.final_features) if self.final_features is not None else None
            ),
            # A routine decision, on the row as a count rather than as an error. `feature_summary`
            # names the columns but never reaches a results row, so before this column the only
            # trace a skip left in `evals/results/` was the `errored` flag it wrongly set.
            "n_skipped_high_cardinality": len(self.skipped_high_cardinality),
            # the same comparison one node upstream. `None` rather than empty when the profiler
            # never ran: a node that crashed nominated nothing in a different sense than a node
            # that looked and declined, and averaging those together would be a lie.
            "profiler_nominated": sorted(nominated) if nominated is not None else None,
            "profiler_caught": (
                bool(nominated & planted) if nominated is not None and graded_for_leakage else None
            ),
            "profiler_recall": (
                len(nominated & planted) / len(planted)
                if nominated is not None and planted
                else None
            ),
            "profiler_false_alarm": (
                len(nominated - planted) if nominated is not None and graded_for_leakage else None
            ),
            # the same comparison at the reviewer, all column-scoped categories
            "reviewer_nominated": sorted(objected) if objected is not None else None,
            "reviewer_caught": (
                bool(objected & planted) if objected is not None and graded_for_leakage else None
            ),
            "reviewer_recall": (
                len(objected & planted) / len(planted) if objected is not None and planted else None
            ),
            "reviewer_false_alarm": (
                len(objected - planted) if objected is not None and graded_for_leakage else None
            ),
            # the loop
            "review_verdict": self.review_verdict,
            "review_loops": self.review_iterations,
            "objections_raised": len(self.objections),
            "objections_by_category": by_category,
            "objections_open_at_end": len(self.open_objections()),
            "objections_resolved": n_resolved,
            "objections_withdrawn": n_withdrawn,
            "objections_falsely_resolved": falsely_resolved,
            # who the reviewer asked to fix it, whether anything was fixed, and where the loop
            # actually went. Between them these separate "the reviewer was wrong" from "the
            # reviewer was right and told a node that has no lever".
            "objections_by_target_node": by_target_node,
            "objections_rerouted": rerouted,
            "objected_columns_unremediated": unremediated,
            "route_sequence": [rp.routed_to for rp in passes],
            "new_objections_per_pass": [len(rp.new_objection_ids) for rp in passes],
            # cost and reliability
            "wall_seconds": self.wall_seconds,
            # The grader's own wall cost, which `wall_seconds` cannot see: `ended_at` is stamped
            # when the graph returns and the re-scorer and baseline run after it. Two columns and
            # not one, because they are two processes with two timeouts that fail independently.
            "rescore_seconds": self.rescore_seconds,
            "baseline_seconds": self.baseline_seconds,
            # Which node the wall time actually went to. `NodeEvent` has carried this since Phase 1
            # but only `ds-agents run` ever printed it -- the harness calls `_run_once` directly, so
            # every committed row discarded it. Summed over repeats, because a node that ran three
            # times cost three times.
            "node_seconds": self.node_seconds,
            "cost_usd": self.total_cost_usd,
            "errored": bool(self.errors),
            # WHICH node's refusal ended the run, or None if none did. `errored` is one bit and
            # cannot separate "a column name could not be associated with the target" from "this
            # dataset cannot be run at all" -- and until the graph learned to halt, the second case
            # produced a row that read like a measurement. Non-null is the greppable "this cell did
            # not run" flag. Fourth instance of the same fix, after `leakage_graded`,
            # `rescore_status` and `baseline_status`: a status column beside the number rather than
            # a row silently withheld, because the reporter exists precisely so the hardest
            # datasets still land in the results file.
            "halted_at": self.halted_at(),
            # What went wrong, not just that something did. `errored` is one bit for every failure
            # mode a run has, which is why the zero-objection `block` bug was invisible in the
            # committed results files and had to be counted by hand off `route_sequence`. Always a
            # list: zero errors is a real 0, and `None` would be indistinguishable from a row
            # written before this column existed. Truncated here and not in the state, because the
            # row is a line someone greps and a client's exception repr can carry an HTTP body.
            "errors": [
                {
                    "node": e.node,
                    "message": e.message[:ERROR_MESSAGE_LIMIT],
                    "recoverable": e.recoverable,
                }
                for e in self.errors
            ],
        }

    def effective_target(self, objection: Objection) -> RoutableNode:
        """Which node this run will actually route the objection to.

        The one place `config.objection_routing` is applied. `Objection.target_node` is never
        rewritten -- the reviewer's own dispatch choice stays on the record, so
        `objections_by_target_node` keeps measuring its judgement even in the arm that overrides
        it. That is what makes `by_category` a recorded condition rather than a thumb on the
        scale, and it means the raw field and this method answer different questions: callers
        asking who ACTS come through `open_objections(target)`, which is this method's only
        caller inside the graph. `results_row` asks it directly, once, to count the disagreements.
        """
        if (
            self.config.objection_routing == "by_category"
            and objection.category in COLUMN_SCOPED_CATEGORIES
        ):
            return "feature_eng"
        return objection.target_node

    def open_objections(self, target_node: RoutableNode | None = None) -> list[Objection]:
        """Objections whose LAST disposition is not resolved or withdrawn.

        Order matters and a set union loses it. The router can send a run back to feature_eng, so
        an objection resolved in pass 2 and marked `still_open` again in pass 3 is reachable; a
        union over all passes would report it closed and the run would end with
        `objections_open_at_end: 0` while the problem is still there.

        `target_node` here means who ACTS, not who the reviewer addressed. Under
        `config.objection_routing="by_category"` a column-scoped objection answers to
        `feature_eng` whatever the reviewer wrote; see `effective_target`.
        """
        closed = {
            oid
            for oid, disposition in self.latest_dispositions().items()
            if disposition in {"resolved", "withdrawn"}
        }
        pending = [o for o in self.objections if o.id not in closed]
        if target_node is not None:
            pending = [o for o in pending if self.effective_target(o) == target_node]
        return pending

    def would_be_open(
        self,
        *,
        adding: Sequence[Objection] = (),
        dispositions: dict[str, Disposition],
        target_node: RoutableNode | None = None,
    ) -> list[Objection]:
        """`open_objections`, asked against the state as it is ABOUT to be.

        `open_objections` folds only the dispositions already recorded in `review_passes`. Two
        callers need the question asked one step earlier, against objections this pass is adding
        and dispositions this pass has not handed to the router yet: the router, deciding where a
        `block` goes, and the reviewer, deciding whether its own `block` has anything to act on at
        all. One implementation, because two would eventually disagree about whether a `block` is
        actionable -- and a reviewer and a router disagreeing about that is the zero-objection
        `block` bug's whole shape.

        `dispositions` can only CLOSE here, never reopen: it is applied by removing the ids it
        resolves or withdraws from the already-open set, rather than by overwriting
        `latest_dispositions`. That mirrors `_route_for_block`'s pre-2026-08-29 arithmetic exactly,
        and it is unreachable to do otherwise through the graph -- the reviewer keys its
        dispositions to ids that were open when the pass began.
        """
        closing_now = {oid for oid, d in dispositions.items() if d in {"resolved", "withdrawn"}}
        pending = [o for o in self.open_objections(target_node) if o.id not in closing_now]
        pending += [
            o
            for o in adding
            if o.id not in closing_now
            and (target_node is None or self.effective_target(o) == target_node)
        ]
        return pending

    def latest_dispositions(self) -> dict[str, Disposition]:
        """Last-write-wins disposition per objection id, folded in iteration order.

        One implementation of the fold, because there were three: this, `open_objections`, and
        `reporter._objection_status`. The ordering is load-bearing rather than incidental -- see
        `open_objections` for the pass-2-resolved, pass-3-reopened case that a set union gets
        wrong -- so three copies were three chances to get it wrong differently.
        """
        latest: dict[str, Disposition] = {}
        for review in sorted(self.review_passes, key=lambda r: r.iteration):
            latest.update(review.dispositions)
        return latest

    def binding_objections(self, target_node: RoutableNode | None = None) -> list[Objection]:
        """Objections whose named columns must stay OUT of the feature matrix.

        NOT the same question as `open_objections`, and the difference is the whole point of the
        method existing. `open_objections` answers "what is still being complained about" and
        closes on `resolved` OR `withdrawn`. This answers "what must stay dropped" and releases
        only on `withdrawn`.

        The reason is that on this pipeline the fix for a column objection IS the drop.
        `feature_eng._forced_drops` recomputes from scratch on every entry, so while it read
        `open_objections`, an objection the reviewer marked `resolved` stopped forcing its drop and
        the next return to that node -- for any unrelated reason -- silently put the leaked column
        back in the matrix. `resolved` would have un-fixed itself. `withdrawn` is the reviewer
        saying it was never a problem, which is the opposite claim, and it is the only route back
        for a false positive: `objection_routing="by_category"` dropped a legitimate strong feature
        in 2 of 10 runs, and without a release that mistake would be permanent for the rest of the
        run. `not_reviewed` binds, for the same reason REVIEWER_SYSTEM forbids silence-as-approval.

        This has exactly ONE caller in the graph, `feature_eng._forced_drops`, and that is the
        invariant to check before adding a second. The router's `_route_for_block` and the
        reviewer's own `_user_message` must keep asking `open_objections`: a resolved objection
        that still routed the run upstream would loop to the cap on every run and make `exhausted`
        structurally guaranteed, and one still shown to the reviewer would be re-adjudicated
        forever. Either would destroy the signal the closure arm exists to make honest.

        `target_node` means who ACTS, resolved through `effective_target`, so this inherits
        `config.objection_routing` rather than becoming a second answer to that question.

        `config.forced_drop_release` is read HERE AND NOWHERE ELSE. It is not a design fork: the
        default `withdrawn_only` is the rule described above, and `resolved_or_withdrawn` makes this
        method exactly `open_objections` again -- the pre-2026-08-28 predicate, reproduced so the
        effect of fixing it has a same-commit control. That equality is pinned by
        `test_the_unsticky_arm_is_exactly_open_objections_again` rather than asserted here, and the
        single-caller invariant above is enforced by
        `test_binding_objections_has_exactly_one_caller_in_the_graph` rather than left to this
        docstring. See DECISIONS.md 2026-08-28 (fifth entry).
        """
        releasing: set[Disposition] = (
            {"withdrawn"}
            if self.config.forced_drop_release == "withdrawn_only"
            else {"resolved", "withdrawn"}
        )
        released = {
            oid
            for oid, disposition in self.latest_dispositions().items()
            if disposition in releasing
        }
        binding = [o for o in self.objections if o.id not in released]
        if target_node is not None:
            binding = [o for o in binding if self.effective_target(o) == target_node]
        return binding

    def placeholder_models(self) -> list[str]:
        """Model names in the trace that are placeholders rather than a real client.

        `StubModel` answers intake from column-name convention and nominates zero leakage
        candidates by design. A results row built from those events would look like a working
        pipeline that found nothing, which is indistinguishable in a table from a real reviewer
        that missed the leak. The harness calls this and refuses the row; see
        `publishable()`.
        """
        return sorted({e.model for e in self.node_trace if e.model in PLACEHOLDER_MODEL_NAMES})

    def fatal_errors(self) -> list[PipelineError]:
        """Errors a node marked `recoverable=False`, in the order they were raised.

        `errors` uses an `operator.add` reducer, so the first fatal error is on the state for the
        rest of the run and a halted run can never un-halt. That is correct by the definition of
        `recoverable=False` -- the node said nothing downstream can produce a trustworthy result --
        but it is a property of the reducer rather than of this method, so it is written down here
        rather than left to be inferred.
        """
        return [e for e in self.errors if not e.recoverable]

    def halted(self) -> bool:
        """Whether the graph should stop running nodes and go straight to the reporter."""
        return bool(self.fatal_errors())

    def halted_at(self) -> NodeName | None:
        """The node whose unrecoverable refusal ended the run, for `results_row()`."""
        fatal = self.fatal_errors()
        return fatal[0].node if fatal else None

    def publishable(self) -> tuple[bool, str]:
        """Whether this run may be written to a results file, and why not if it may not.

        A tuple rather than a bool because the caller has to print the reason. Returning False
        with no explanation produces a harness that silently drops rows, which is the same
        failure as writing fake ones.
        """
        placeholders = self.placeholder_models()
        if placeholders:
            return False, (
                f"node_trace contains placeholder model(s) {', '.join(placeholders)}; "
                "no number from this run is real"
            )
        if not self.node_trace:
            return False, "node_trace is empty; nothing ran"
        # A halted run IS publishable, deliberately. Its cost, node trace, timings and errors are
        # all real; only its score columns are None, and `halted_at` says why. Refusing it would
        # delete exactly the hardest datasets from the results file, which is the bias
        # `nodes/reporter.py` exists to prevent -- and would leave the reason in stdout scrollback,
        # which is the same shape as the defect this column was added to fix.
        return True, ""
