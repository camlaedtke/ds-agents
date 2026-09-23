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

import itertools
import operator
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

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

# Which reviewer system prompt the run used. A run condition: "cannot see the leak" and "was
# never asked which column produced the score" are different failures, and a reviewer number that
# doesn't record the prompt can't separate them.
ReviewerPrompt = Literal["base", "which_column"]
REVIEWER_PROMPTS: tuple[ReviewerPrompt, ...] = ("base", "which_column")

# How the graph decides WHICH NODE ACTS on an objection, not what the reviewer said (that is
# `Objection.target_node`, recorded verbatim and never rewritten). `as_addressed` obeys it.
# `by_category` routes any COLUMN_SCOPED_CATEGORIES objection to `feature_eng` regardless of what
# the reviewer chose, since a column is the only thing feature_eng can act on.
ObjectionRouting = Literal["as_addressed", "by_category"]
OBJECTION_ROUTINGS: tuple[ObjectionRouting, ...] = ("as_addressed", "by_category")

# Whether the reviewer was told what "done" looks like. Without it, a reviewer can hold an
# objection open on an unfalsifiable standard (a column "never validated as non-leaking, only
# removed") forever, so every run grinds to the cap. `on` appends one rule pointing at
# `final_features`, a field the reviewer is already shown.
#
# Its own axis rather than a third `ReviewerPrompt` value: bundling the two would mean closure
# could never be measured independent of `which_column`.
ObjectionClosure = Literal["off", "on"]
OBJECTION_CLOSURES: tuple[ObjectionClosure, ...] = ("off", "on")

# Which disposition RELEASES a column an objection forced out of the matrix. `withdrawn_only` is
# the correct rule; `resolved_or_withdrawn` reproduces a defect where a `resolved` objection
# stopped forcing its drop and the leaked column came back on the next unrelated return to
# feature_eng.
#
# The only RunConfig axis whose default is NOT the pre-fix behaviour: it exists only as a
# same-commit control for one pre-registered cell, never as a general ablation lever. See
# DECISIONS.md (2026-08-28, fifth entry).
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

An enum rather than a bare null, so a reader doesn't have to guess whether a null score means no
withheld holdout existed, no model was produced, or the sandbox died. `refit_mismatch` keeps the
score but flags it, since pooling it would launder a bug into a result.

Never appends a `PipelineError`: `errored` means the RUN went wrong, not the grader.
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

Its own enum rather than a widening of `RescoreStatus`: the baseline runs in its own process so
the yardstick can fail without the measurement failing. `unit_point_failed` keeps
`baseline_zero_score` on the row, since the zero point was measured even when the unit point fit
did not survive.

`rescore_unavailable` states the one real coupling: the baseline is fit on the agents'
`split["train"]` and scored on the same withheld rows, so most reasons the re-scorer can't run
apply here too.

A degenerate or merely narrow scale is not a status -- both points were honestly measured either
way. `baseline_separation` publishes the span so a reader can judge a narrow scale directly instead
of trusting a flag.
"""

TOP_IMPORTANCE_N80_THRESHOLD = 0.8
"""The fraction of positive importance mass `top_importance_n80` walks `top_importances` to reach.

Named for the field it feeds so the two cannot drift apart silently -- `n80` in the field name and
`0.8` here must always mean the same number.
"""

TopImportanceStatus = Literal["ok", "no_importances", "no_positive_importance"]
"""Why `top_importance_share` and `top_importance_n80` are or are not on this row.

Two different reasons a summary can be missing. `no_importances` means `top_importances` itself
is empty -- no candidate chosen, or modeling produced nothing. `no_positive_importance` means
every mean importance is zero or negative -- permutation importance can go negative when
shuffling a column improves the score -- leaving no positive mass to divide or sum toward. Both
derived columns are `None` under either status.

Computed over the same top `TOP_IMPORTANCES` (15) entries the reviewer is actually shown, since
these two columns describe the shape of what the reviewer's prompt contains, not the full ranking.
"""

# The one data seed in the project. `RunConfig.random_seed` and `holdout.prepare` (which carves
# the withheld rows before any node runs) must both read this, or the file claims one seed while
# two different numbers were actually used.
DEFAULT_RANDOM_SEED = 20260822

DEFAULT_LOOP_CAP = 3
"""How many review passes a run gets before the router gives up and reports.

Named once and shared by `RunConfig`, `cli._run_once`, `harness.EvalCell` and the `--loop-cap`
argparse default, since `loop_cap` is a recorded condition on every results row and a duplicated
literal is a chance for a cell to silently claim a condition it did not run under.
"""

BASELINE_MIN_SEPARATION = 1e-9
"""The smallest `unit - zero` that `baseline_normalised_score` will divide by.

A divide-by-zero guard, not a power criterion -- deciding whether a separation is large enough to
be meaningful needs `n_withheld_rows` and belongs in `evaldiff`. Publishing an unstable number
beside its two raw inputs is preferred to silently withholding it.
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
        description="Whether the agents saw the fixture's real column names or `var_NN`. Sets "
        "trap difficulty, since the profiler nominates leakage largely off column names. On the "
        "frozen config because a leakage number without it is not interpretable. See "
        "`ds_agents.naming`.",
    )
    reviewer_prompt: ReviewerPrompt = Field(
        default="base",
        description="Which system prompt the reviewer ran under: `base`, or `which_column`, which "
        "appends one rule asking the reviewer to name the column that explains an implausible "
        "score. On the frozen config because a reviewer comparison that doesn't record the prompt "
        "is confounded by it.",
    )
    objection_routing: ObjectionRouting = Field(
        default="as_addressed",
        description="Who the graph asks to act on an objection: `as_addressed`, obeying the "
        "reviewer's own `target_node`, or `by_category`, which routes a column-scoped objection "
        "to `feature_eng` regardless of what the reviewer wrote. On the frozen config since the "
        "two arms remediate at different rates. Changes what `feature_eng` and `modeler` are "
        "SHOWN, not just where the run goes.",
    )
    objection_closure: ObjectionClosure = Field(
        default="off",
        description="Whether the reviewer was given a termination condition it can check: `off`, "
        "or `on`, which appends one rule saying an objection about a column is answered when that "
        "column is absent from `final_features`. Hazard: a prompt that buys termination by "
        "teaching the reviewer to say 'fixed' is worse than none, so `objections_falsely_resolved` "
        "is tracked beside it.",
    )
    forced_drop_release: ForcedDropRelease = Field(
        default="withdrawn_only",
        description="Which disposition releases a column an objection forced out of the matrix: "
        "`withdrawn_only`, the correct rule and the default, or `resolved_or_withdrawn`, which "
        "reproduces a defect where a `resolved` objection stopped forcing its drop and the leaked "
        "column came back on the next unrelated return to feature_eng. Defect reproduction only, "
        "for a same-commit control -- not a general ablation lever.",
    )
    holdout_fraction: float = Field(
        default=0.0,
        ge=0.0,
        lt=0.5,
        description="How much of the dataset was withheld from the agents before the graph "
        "started. 0.0 for every fixture by decision: planted leakage is a column, so a random "
        "holdout still contains it, and carving rows out would only make committed rows "
        "incomparable.",
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
        description="Short git hash of the tree that produced this run, `-dirty` suffixed when "
        "the working tree was not clean. Null when nothing recorded it. Which code ran is a run "
        "condition no other field carries. A dirty tree groups as its own cell in `eval-diff`, "
        "correctly: a dirty run is not reproducible.",
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
    fit_error: str | None = Field(
        default=None,
        description="Why this candidate could not be fit at all, as 'Type: message'. NOT "
        "derivable from cv_scores: an empty list also means fit-but-scored-nothing, a different "
        "event with a different cause. A companion to the node's PipelineError, not a replacement.",
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
        description="Columns that survived into the matrix the model was fit on. SOURCE column "
        "names, never one-hot expansions: `results_row()` intersects this with "
        "`planted_leakage_columns`, which are source names, so storing 'colour=red' here would "
        "empty the intersection and score every run as remediated.",
    )
    dropped_features: list[str] = Field(default_factory=list)
    skipped_high_cardinality: list[str] = Field(
        default_factory=list,
        description="Columns dropped because they have more distinct values than the one-hot "
        "encoder will expand. A recorded DECISION, not a `PipelineError`: counting it in `errors` "
        "made `errored` read true on completely healthy runs. SOURCE column names, and a subset "
        "of `dropped_features`.",
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
        "vocabulary, and a set comparison across two vocabularies is not a comparison. Enforced "
        "sorted descending by mean importance at construction -- see the field_validator below.",
    )

    @field_validator("top_importances")
    @classmethod
    def _top_importances_sorted_descending(
        cls, value: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Reject a `top_importances` not sorted highest-mean-first.

        `top_importance_share` and `top_importance_n80` both read rank order off this list without
        re-sorting -- see their own docstrings -- and the reviewer's prompt is told to "read
        `top_importances` from the top" on the same assumption. An out-of-order list would make
        both computed fields silently wrong and the reviewer's prompt silently misleading, so this
        fails loudly at the field boundary instead of being repaired downstream. Ties (equal means)
        are allowed -- only a STRICT increase from one entry to the next is a violation.
        """
        for (column_a, mean_a), (column_b, mean_b) in itertools.pairwise(value):
            if mean_a < mean_b:
                raise ValueError(
                    "top_importances must be sorted descending by mean importance: "
                    f"{column_a!r} ({mean_a}) precedes {column_b!r} ({mean_b}), which is greater"
                )
        return value

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
        "`operator.add`). The router folds this into the `ReviewPass` it mints; "
        "`ReviewPass.dispositions` is the durable record.",
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
        "because they are a separate process with a separate timeout and can fail alone.",
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
        reader can see it -- a normalised score is otherwise uncheckable when the scale itself is
        narrow (see `BASELINE_MIN_SEPARATION`).

        Not suppressed on a planted leak, unlike the normalised score: the two raw points are
        honest measurements and the distance between them is a property of the dataset, not the
        run's grade.
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
        AMLB's normalisation and NOT a ratio -- `verified / baseline` disagrees about what 1.0
        means, and is meaningless on a metric whose floor isn't zero (e.g. r2).

        Written as a difference so the formula is already direction-invariant; `tests/test_state.py`
        pins that the ratio form would agree, so a future simplification can't reintroduce that bug.

        Never raises: a computed field that raises takes `model_dump_json()` and `results_row()`
        down with it, and every dataset needs a row even on hard failure.
        """
        if self.verified_holdout_score is None or self.spec is None:
            return None
        # Restated rather than left implicit in `baseline_separation` below, so a reader doesn't
        # have to follow a property into another property to see the numerator can't be None here.
        if self.baseline_zero_score is None or self.baseline_unit_score is None:
            return None
        # SUPPRESSED ON A PLANTED LEAK. The baseline is fit on every raw column, including the
        # trap the pipeline was supposed to drop, so a pipeline that correctly drops it scores
        # BELOW a baseline that kept it -- the same value under 1 means opposite things on a
        # labelled vs. unlabelled dataset. Gated like the leakage columns; raw points stay on the
        # row since they are honest measurements.
        if self.planted_leakage_columns:
            return None
        separation = self.baseline_separation
        # `<=`, not `abs(...) < eps`: a NEGATIVE separation means the RandomForest did worse than
        # the class prior, which inverts the scale, and this is reachable (f1 with a minority
        # positive class has a zero point of exactly 0.0).
        if separation is None or separation <= BASELINE_MIN_SEPARATION:
            return None
        numerator = self.verified_holdout_score - self.baseline_zero_score
        return (numerator if self.spec.greater_is_better else -numerator) / separation

    @computed_field
    @property
    def top_importance_status(self) -> TopImportanceStatus:
        """Why `top_importance_share` and `top_importance_n80` are or are not on this row.

        See the module-level `TopImportanceStatus` docstring for what the two null values mean.
        """
        if not self.top_importances:
            return "no_importances"
        if not any(mean > 0.0 for _, mean in self.top_importances):
            return "no_positive_importance"
        return "ok"

    @computed_field
    @property
    def top_importance_share(self) -> float | None:
        """Rank-1 mean importance / sum of positive mean importances, among the shown columns.

        A float in (0, 1]: `top_importances` is sorted highest-first (see its own docstring), so
        the rank-1 entry is the maximum of the list, and under `top_importance_status == "ok"`
        that maximum is itself positive and no larger than the sum of every positive value in the
        list. Close to 1.0 when one column dominates the distribution the reviewer is shown; close
        to 1/n when it is flat across n roughly-equal columns.

        `None` exactly when `top_importance_status != "ok"` -- see that field for the two reasons.
        """
        if self.top_importance_status != "ok":
            return None
        positive_sum = sum(mean for _, mean in self.top_importances if mean > 0.0)
        return self.top_importances[0][1] / positive_sum

    @computed_field
    @property
    def top_importance_n80(self) -> int | None:
        """How many shown columns it takes to reach `TOP_IMPORTANCE_N80_THRESHOLD` of their total
        positive importance.

        Walks `top_importances` in its existing highest-first order, summing only positive mean
        importances -- zero and negative entries contribute no mass, and because the list is
        sorted descending they only ever appear after every positive one -- and returns the rank
        at which the running sum first reaches `TOP_IMPORTANCE_N80_THRESHOLD` of the positive
        total. 1 when a single column carries the distribution the reviewer is shown; close to the
        count of positive columns when it is flat.

        An integer >= 1 exactly when `top_importance_status == "ok"`, `None` otherwise -- see that
        field for the two reasons a shape cannot be summarised.
        """
        if self.top_importance_status != "ok":
            return None
        positive = [mean for _, mean in self.top_importances if mean > 0.0]
        threshold = TOP_IMPORTANCE_N80_THRESHOLD * sum(positive)
        cumulative = 0.0
        for rank, mean in enumerate(positive, start=1):
            cumulative += mean
            if cumulative >= threshold:
                return rank
        # Unreachable in exact arithmetic: the full positive prefix sums to exactly the total,
        # which is always >= threshold. Kept only as a floating-point guard so this can never
        # return None once `top_importance_status` says "ok".
        return len(positive)

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
        # `planted` is a COMPLETE ground-truth list or it is nothing at all. A fixture has one by
        # construction; an external benchmark dataset has none (`manifest.yaml` marks it
        # `leakage_labelled: false`). Left ungated, an empty `planted` would make the leakage
        # columns below assert "no leak" and "reviewer missed it" with no evidence for either, so
        # they report `None` instead and `evaldiff` excludes `None` from its denominators.
        #
        # Derived rather than stored, so there is no second source of truth for what `planted` says.
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

        # The profiler's nominations, scored separately from the reviewer's objections: they can
        # disagree (profiler flags a column, reviewer says nothing), and folding them into one
        # `leakage_caught` would hide which team member actually caught it.
        nominated: set[str] | None = None
        if self.profile is not None:
            nominated = {c.column for c in self.profile.leakage_candidates}

        # The reviewer's columns over ALL column-scoped categories, `implausible_importance`
        # included -- `flagged` above stays a two-category number so it doesn't redefine the
        # published results file. `None` when no pass completed, so a reviewer-off no-op run
        # doesn't average in as a 0.0 recall against one that looked and declined.
        objected: set[str] | None = None
        if self.config.reviewer_enabled and self.review_passes:
            objected = self.objected_columns()
        by_category = dict.fromkeys(get_args(ObjectionCategory), 0)
        for objection in self.objections:
            by_category[objection.category] += 1

        # Why a run that caught the trap still shipped it. `objections_by_category` says what the
        # reviewer objected to; these four say whether anything could act on it. `target_node` is
        # read RAW here, deliberately -- `effective_target` is only used to count disagreements --
        # so the reviewer's own dispatch judgement stays measurable even in the arm that overrides
        # it, and `objections_rerouted` says how often the graph disagreed with it.
        by_target_node = dict.fromkeys(get_args(RoutableNode), 0)
        rerouted = 0
        for objection in self.objections:
            by_target_node[objection.target_node] += 1
            if self.effective_target(objection) != objection.target_node:
                rerouted += 1

        # The caught-versus-remediated gap as a list of names: a column-scoped objection whose
        # column is still in the matrix at the end was raised and not acted on. `None` when
        # feature_eng produced nothing, or when no reviewer pass completed -- same reasoning as
        # `leakage_remediated` and `objected` above.
        unremediated: list[str] | None = None
        if objected is not None and self.final_features:
            unremediated = sorted(objected & set(self.final_features))

        # Closure, as something measured rather than hoped for. `objections_open_at_end` says how
        # many were never closed; these say HOW the closed ones closed. `resolved` and `withdrawn`
        # are split because they are opposite claims about the reviewer -- one says the fix landed,
        # the other says the objection was wrong -- and `binding_objections` acts on that
        # difference.
        #
        # Gated exactly as `reviewer_nominated` and friends are. A reviewer that ran and closed
        # nothing is a REAL 0, not a `None`.
        latest = self.latest_dispositions()
        by_id = {o.id: o for o in self.objections}
        n_resolved: int | None = None
        n_withdrawn: int | None = None
        falsely_resolved: int | None = None
        if self.config.reviewer_enabled and self.review_passes:
            n_resolved = sum(1 for d in latest.values() if d == "resolved")
            n_withdrawn = sum(1 for d in latest.values() if d == "withdrawn")
            # Catches the failure mode `objection_closure="on"` creates: an objection marked
            # `resolved` whose column is still in the matrix. Counted over objections, not
            # columns, since the unit being scored is the reviewer's judgement act, and only over
            # column-scoped categories -- a `resolved` `overfit` objection names no column.
            #
            # `None` on an empty matrix, same reasoning as `leakage_remediated`: every resolution
            # would score honest there, flattering the arm under test.
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
            # Why the two columns above do or do not carry a number: the reason a null is null,
            # said outright instead of inferred.
            "rescore_status": self.rescore_status,
            "rescore_detail": self.rescore_detail[:ERROR_MESSAGE_LIMIT],
            "refit_claim_gap": self.refit_claim_gap,
            "n_withheld_rows": self.n_withheld_rows,
            # The scale `verified_holdout_score` is read against: a constant class-prior
            # predictor at 0 and a RandomForest at 1. Published as two raw points plus the
            # normalisation, not one `baseline_score`, since the normalised column is suppressed
            # on a planted leak while the two raw points are not.
            "baseline_zero_score": self.baseline_zero_score,
            "baseline_unit_score": self.baseline_unit_score,
            "baseline_normalised_score": self.baseline_normalised_score,
            "baseline_separation": self.baseline_separation,
            "baseline_status": self.baseline_status,
            "baseline_detail": self.baseline_detail[:ERROR_MESSAGE_LIMIT],
            "baseline_recipe": self.baseline_recipe,
            # leakage, as a set comparison against ground truth. `leakage_graded` states outright
            # why a null is null, so a reader pooling `leakage_caught` can filter on it instead of
            # guessing.
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
            # How wide the matrix the run actually shipped is. Without it, a run that remediated
            # by force-dropping most of the fixture is indistinguishable from one that dropped
            # only the trap.
            "n_final_features": (
                len(self.final_features) if self.final_features is not None else None
            ),
            # A routine decision, on the row as a count rather than as an error.
            "n_skipped_high_cardinality": len(self.skipped_high_cardinality),
            # Which candidates could not be fit at all, as a count and names beside the
            # denominator that makes them readable. `n_candidates` is not decoration: without it a
            # run halted before any candidate was attempted reads as "every candidate fit fine".
            "n_candidates": len(self.candidates),
            "n_candidates_failed_to_fit": sum(1 for c in self.candidates if c.fit_error),
            "candidates_failed_to_fit": sorted(c.name for c in self.candidates if c.fit_error),
            # The shape of the permutation-importance distribution the REVIEWER is shown --
            # `top_importances`, truncated to `TOP_IMPORTANCES`, is the only per-column evidence
            # in its prompt. `top_importance_status` says why the other two are null when they are.
            "top_importance_share": self.top_importance_share,
            "top_importance_n80": self.top_importance_n80,
            "top_importance_status": self.top_importance_status,
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
            # The grader's own wall cost, which `wall_seconds` cannot see: the re-scorer and
            # baseline run after the graph returns, as two processes with independent timeouts.
            "rescore_seconds": self.rescore_seconds,
            "baseline_seconds": self.baseline_seconds,
            # Which node the wall time actually went to, summed over repeats, since a node that
            # ran three times cost three times.
            "node_seconds": self.node_seconds,
            "cost_usd": self.total_cost_usd,
            "errored": bool(self.errors),
            # WHICH node's refusal ended the run, or None if none did. `errored` is one bit and
            # can't separate a recoverable node error from "this dataset cannot be run at all";
            # non-null here is the greppable "this cell did not run" flag.
            "halted_at": self.halted_at(),
            # What went wrong, not just that something did. Always a list: zero errors is a real
            # 0, and `None` would be indistinguishable from a row written before this column
            # existed. Truncated here, not in the state, since a client's exception repr can carry
            # an HTTP body.
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
        rewritten, so the reviewer's own dispatch choice stays on the record and
        `objections_by_target_node` keeps measuring it even in the arm that overrides it. Callers
        asking who ACTS come through `open_objections(target)`.
        """
        if (
            self.config.objection_routing == "by_category"
            and objection.category in COLUMN_SCOPED_CATEGORIES
        ):
            return "feature_eng"
        return objection.target_node

    def open_objections(self, target_node: RoutableNode | None = None) -> list[Objection]:
        """Objections whose LAST disposition is not resolved or withdrawn.

        Order matters and a set union loses it: an objection resolved in pass 2 and marked
        `still_open` again in pass 3 is reachable, and a union over all passes would report it
        closed when the problem is still there.

        `target_node` here means who ACTS, not who the reviewer addressed -- see `effective_target`.
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

        `open_objections` folds only dispositions already recorded in `review_passes`. Two callers
        need the question asked one step earlier, against objections this pass is adding and
        dispositions not yet handed to the router: the router, deciding where a `block` goes, and
        the reviewer, deciding whether its own `block` has anything to act on. One implementation,
        so the two can never disagree about whether a `block` is actionable.

        `dispositions` can only CLOSE here, never reopen: applied by removing the ids it resolves
        or withdraws from the already-open set, never by overwriting `latest_dispositions`.
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

        Shared by `open_objections` and the reporter. The ordering matters: see `open_objections`
        for the pass-2-resolved, pass-3-reopened case a set union gets wrong.
        """
        latest: dict[str, Disposition] = {}
        for review in sorted(self.review_passes, key=lambda r: r.iteration):
            latest.update(review.dispositions)
        return latest

    def binding_objections(self, target_node: RoutableNode | None = None) -> list[Objection]:
        """Objections whose named columns must stay OUT of the feature matrix.

        NOT the same question as `open_objections`. `open_objections` answers "what is still being
        complained about" and closes on `resolved` OR `withdrawn`. This answers "what must stay
        dropped" and releases only on `withdrawn`.

        On this pipeline the fix for a column objection IS the drop, and `feature_eng._forced_drops`
        recomputes from scratch on every entry -- so if a `resolved` objection released its column,
        the next unrelated return to that node would silently put a leaked column back.
        `withdrawn` is the reviewer saying it was never a problem, the opposite claim, and the only
        route back for a false positive. `not_reviewed` binds, for the same reason silence is never
        treated as approval.

        Exactly ONE caller in the graph, `feature_eng._forced_drops` -- check that invariant before
        adding a second. The router and the reviewer's own prompt must keep asking
        `open_objections`: a resolved objection that still routed the run upstream would loop to
        the cap on every run.

        `target_node` means who ACTS, resolved through `effective_target`, inheriting
        `config.objection_routing` rather than becoming a second answer to that question.

        `config.forced_drop_release` is read HERE AND NOWHERE ELSE. `resolved_or_withdrawn` makes
        this method exactly `open_objections` again -- the pre-fix predicate, reproduced as a
        same-commit control (`test_the_unsticky_arm_is_exactly_open_objections_again`). See
        DECISIONS.md (2026-08-28, fifth entry).
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

        `StubModel` nominates zero leakage candidates by design, so a row built from its events
        would look like a real reviewer that missed the leak. The harness calls this and refuses
        the row; see `publishable()`.
        """
        return sorted({e.model for e in self.node_trace if e.model in PLACEHOLDER_MODEL_NAMES})

    def fatal_errors(self) -> list[PipelineError]:
        """Errors a node marked `recoverable=False`, in the order they were raised.

        `errors` uses an `operator.add` reducer, so the first fatal error stays on the state for
        the rest of the run and a halted run can never un-halt -- correct, since
        `recoverable=False` means the node said nothing downstream can be trusted.
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
