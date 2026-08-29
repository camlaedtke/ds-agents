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

# Which direction counts as better. `score_ratio` is meaningless without this: the same ratio
# means "good" for roc_auc and "bad" for rmse.
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
    loop_cap: int = Field(default=3, ge=0)
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
    random_seed: int = 20260822
    dataset_hash: str | None = None
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
    baseline_score: float | None = None

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

    @computed_field
    @property
    def score_ratio(self) -> float | None:
        """Direction-aware, so classification and regression rows mean the same thing.

        `None` on a zero denominator rather than an exception. A computed field that raises takes
        `model_dump_json()` and `results_row()` down with it, and ARCHITECTURE.md requires a row
        for every dataset even on hard failure -- losing the row for the worst outcomes biases
        every table upward. Two zeros are reachable, not hypothetical: rmse 0.0 is what a perfect
        leaked copy scores, and the predict-the-mean baseline for r2 is exactly 0.0.

        The r2 case is a real gap, not just a guard: a ratio against a 0.0 baseline has no
        meaning, so those rows need a difference column instead. See docs/NEXT.md.
        """
        if self.verified_holdout_score is None or self.spec is None:
            return None
        if self.baseline_score is None or self.baseline_score == 0.0:
            return None
        if self.spec.greater_is_better:
            return self.verified_holdout_score / self.baseline_score
        if self.verified_holdout_score == 0.0:
            return None
        return self.baseline_score / self.verified_holdout_score

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
            # scores. `claimed` is what the agent said; `verified` is what we measured.
            "claimed_holdout_score": claimed,
            "verified_holdout_score": self.verified_holdout_score,
            "holdout_claim_gap": gap,
            "baseline_score": self.baseline_score,
            "score_ratio": self.score_ratio,
            "metric": self.spec.metric if self.spec else None,
            # leakage, as a set comparison against ground truth
            "leakage_planted": sorted(planted),
            "leakage_flagged": sorted(flagged),
            "leakage_caught": bool(true_positives),
            "leakage_remediated": remediated,
            "leakage_recall": len(true_positives) / len(planted) if planted else None,
            "leakage_precision": len(true_positives) / len(flagged) if flagged else None,
            "false_alarm_columns": sorted(false_positives),
            "false_alarm": len(false_positives),
            "leakage_flagged_standing": sorted(standing),
            "false_alarm_standing": len(standing - planted),
            # How wide the matrix the run actually shipped is. `leakage_remediated` is None on an
            # EMPTY matrix but True on a one-column one, so without this a run that remediated by
            # force-dropping most of the fixture is indistinguishable from one that dropped only
            # the trap. `objection_routing="by_category"` turns a reviewer false positive into a
            # real dropped column, so this is the price tag on that arm.
            "n_final_features": (
                len(self.final_features) if self.final_features is not None else None
            ),
            # the same comparison one node upstream. `None` rather than empty when the profiler
            # never ran: a node that crashed nominated nothing in a different sense than a node
            # that looked and declined, and averaging those together would be a lie.
            "profiler_nominated": sorted(nominated) if nominated is not None else None,
            "profiler_caught": bool(nominated & planted) if nominated is not None else None,
            "profiler_recall": (
                len(nominated & planted) / len(planted)
                if nominated is not None and planted
                else None
            ),
            "profiler_false_alarm": len(nominated - planted) if nominated is not None else None,
            # the same comparison at the reviewer, all column-scoped categories
            "reviewer_nominated": sorted(objected) if objected is not None else None,
            "reviewer_caught": bool(objected & planted) if objected is not None else None,
            "reviewer_recall": (
                len(objected & planted) / len(planted) if objected is not None and planted else None
            ),
            "reviewer_false_alarm": len(objected - planted) if objected is not None else None,
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
            "cost_usd": self.total_cost_usd,
            "errored": bool(self.errors),
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
        return True, ""
