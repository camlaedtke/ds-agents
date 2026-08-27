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

ReviewVerdict = Literal["pending", "pass", "block", "exhausted"]

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
    random_seed: int = 20260822
    dataset_hash: str | None = None


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
            "reviewer_model": self.config.reviewer_model,
            "reviewer_sees_code": self.config.reviewer_sees_code,
            "naming": self.config.naming,
            "loop_cap": self.config.loop_cap,
            "random_seed": self.config.random_seed,
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
            # the loop
            "review_verdict": self.review_verdict,
            "review_loops": self.review_iterations,
            "objections_raised": len(self.objections),
            "objections_open_at_end": len(self.open_objections()),
            # cost and reliability
            "wall_seconds": self.wall_seconds,
            "cost_usd": self.total_cost_usd,
            "errored": bool(self.errors),
        }

    def open_objections(self, target_node: RoutableNode | None = None) -> list[Objection]:
        """Objections whose LAST disposition is not resolved or withdrawn.

        Order matters and a set union loses it. The router can send a run back to feature_eng, so
        an objection resolved in pass 2 and marked `still_open` again in pass 3 is reachable; a
        union over all passes would report it closed and the run would end with
        `objections_open_at_end: 0` while the problem is still there.
        """
        latest: dict[str, Disposition] = {}
        for review in sorted(self.review_passes, key=lambda r: r.iteration):
            latest.update(review.dispositions)
        closed = {
            oid for oid, disposition in latest.items() if disposition in {"resolved", "withdrawn"}
        }
        pending = [o for o in self.objections if o.id not in closed]
        if target_node is not None:
            pending = [o for o in pending if o.target_node == target_node]
        return pending

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
