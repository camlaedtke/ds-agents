"""reviewer: argue with the run so far and hand the router a claim.

Reads `spec`, `profile`, `feature_summary`, `final_features`, `dropped_features`, `candidates`,
`chosen_model`, `top_importances`, `task_description`, `open_objections()`, `review_iterations`,
`config.*`; `feature_code_artifact` content only when `config.reviewer_sees_code` (via
`read_artifact`). Writes `objections` (append), `reviewer_claim`, `reviewer_dispositions`,
`node_trace`, and `errors` conditionally. Never `review_verdict`, `review_iterations`, or
`review_passes` -- those belong to the router, which is the sole authority that turns a claim into
an outcome. Tools: `read_artifact` only. One `model.generate` call per pass.

The reviewer is a model under test, not a trusted judge, so two things here are deliberately not
the reviewer's call:

1. **A crashed pass reads as no pass.** If the model call fails, the node does not fall back to a
   safe default claim the way `profiler` and `feature_eng` fall back to an empty nomination --
   there is no safe default "claim". Instead both handoff fields (`reviewer_claim`,
   `reviewer_dispositions`) are cleared on every return path, including this one, so the router
   sees exactly what it would see from a disabled reviewer: `pending`, not a stale claim left over
   from whatever the field held before this node ran.
2. **The claim is never repaired.** A `claim: "block"` that ends up with nothing open after
   filtering and dispositions is not silently promoted to `pass` here -- that is the router's
   documented terminal-block path (an error, routed to the reporter), not something this node gets
   to paper over.

Column filtering follows the profiler's own rule: a proposed objection's columns are checked
against the profile's columns minus the target, and a column-scoped category (`leakage`,
`contamination`, `implausible_importance`) left with no surviving column is REJECTED outright, not
downgraded to `other` -- downgrading would produce an un-actionable objection nobody can close,
which guarantees `exhausted` rather than giving the model a real path to `pass`.
"""

import json
from typing import Any, Literal

from pydantic import Field

from ds_agents.nodes._run import NodeRun
from ds_agents.state import (
    COLUMN_SCOPED_CATEGORIES,
    Contract,
    Disposition,
    Objection,
    ObjectionCategory,
    PipelineError,
    PipelineState,
    RoutableNode,
    Severity,
)
from ds_agents.tools.llm import StructuredModel
from ds_agents.tools.protocol import ToolError, Tools

REVIEWER_SYSTEM = """You are the adversarial review step of a tabular data science pipeline.

You see the feature set the pipeline settled on, the models it fit, and (when shown) the code \
that built the features. Decide whether the run may proceed as-is (`claim: "pass"`) or whether \
something needs to change before it does (`claim: "block"`).

Rules:
- Raise an objection only if you can name the column(s) at issue and state the number or \
observation behind it. A vague objection cannot be checked against the profile or acted on by a \
later step.
- `leakage`, `contamination`, and `implausible_importance` objections must name at least one \
column -- their whole meaning is "this column is the problem". Objections in every other category \
may leave `columns` empty.
- `target_node` says who should act on a new objection: `feature_eng` for anything about which \
columns are in the matrix, `modeler` for anything about which candidate was chosen or how it was \
evaluated.
- For every objection listed under `open_objections`, say what happened to it in `dispositions`: \
`resolved` if the problem is actually fixed, `withdrawn` if you no longer think it was a problem, \
`still_open` if it is neither. Leaving one out means "I did not look at it again" -- do not use \
silence to mean resolved.
- `claim: "block"` with nothing left open once your dispositions are applied is a dead end: \
nothing routes back to you empty-handed, so only block when there is something left for \
feature_eng or modeler to act on.
- You are not the modeler or feature_eng. State the problem; do not propose the fix."""


class ProposedObjection(Contract):
    """What the model proposes raising. `Objection` minus `id` and `raised_at_iteration` -- the
    node stamps both after parsing, the latter from `state.review_iterations` before the router
    increments it.

    It deliberately does NOT carry `Objection`'s validator requiring a column for column-scoped
    categories. That rule belongs to the node's filtering loop, one objection at a time. On this
    schema it is a whole-response validator: a single malformed objection fails the entire
    `ReviewFinding`, so the claim, the dispositions and every well-formed objection alongside it
    are lost, and the pass reads as a reviewer crash. Observed live on 2026-08-27 with Haiku."""

    category: ObjectionCategory
    subcategory: str = Field(
        description="Free text, so a reviewer catching something unanticipated is not forced to "
        "mislabel it as spec_violation."
    )
    target_node: RoutableNode
    columns: list[str] = Field(default_factory=list)
    evidence: str
    severity: Severity


class DispositionUpdate(Contract):
    """One objection's fate this pass. A list, not a dict keyed by id, because constrained
    decoding shapes a dict's keys unpredictably. `not_reviewed` is absent from this enum on
    purpose: it is never something the model says, only the node's label for an open objection the
    model's list left silent."""

    objection_id: str
    disposition: Literal["still_open", "resolved", "withdrawn"]


class ReviewFinding(Contract):
    """What the reviewer asks the model for, once per pass."""

    claim: Literal["pass", "block"]
    objections: list[ProposedObjection] = Field(default_factory=list)
    dispositions: list[DispositionUpdate] = Field(default_factory=list)
    summary: str = ""


def _feature_code(
    state: PipelineState, tools: Tools
) -> tuple[str | None, str, PipelineError | None]:
    """Read the feature code iff `config.reviewer_sees_code`, following the same truncation guard
    `modeler` and `feature_eng` apply to artifacts they exec. A partial transform breeds false
    alarms -- a metric this project publishes -- so the recoverable outcome here is "omit and
    continue", never "review whatever bytes came back"."""
    if not state.config.reviewer_sees_code:
        return None, "reviewer_sees_code is False; feature code withheld by config", None
    if state.feature_code_artifact is None:
        return None, "no feature_code_artifact; feature_eng did not produce one", None
    try:
        payload = tools.read_artifact(state.feature_code_artifact)
    except ToolError as exc:
        error = PipelineError(
            node="reviewer",
            message=f"could not read feature code artifact: {exc}; continuing without it",
        )
        return None, f"feature code artifact could not be read: {exc}", error
    if payload.truncated:
        error = PipelineError(
            node="reviewer",
            message="feature code artifact was truncated; omitting rather than reviewing a "
            "partial transform, which breeds false alarms",
        )
        return None, "feature code artifact was truncated; omitted", error
    return payload.content, "shown below", None


def _user_message(state: PipelineState, *, feature_code: str | None, feature_code_note: str) -> str:
    assert state.spec is not None and state.profile is not None
    facts = {
        "task_description": state.task_description,
        "target": state.spec.target,
        "task_type": state.spec.task_type,
        "metric": state.spec.metric,
        "feature_summary": state.feature_summary,
        "final_features": state.final_features or [],
        "dropped_features": state.dropped_features,
        "candidates": [
            {"name": c.name, "cv_mean": c.cv_mean, "claimed_holdout_score": c.claimed_holdout_score}
            for c in state.candidates
        ],
        "chosen_model": state.chosen_model.name if state.chosen_model else None,
        "top_importances": [
            {"column": name, "importance": value} for name, value in state.top_importances
        ],
        "open_objections": [
            {
                "id": o.id,
                "category": o.category,
                "columns": o.columns,
                "evidence": o.evidence,
                "severity": o.severity,
                "target_node": o.target_node,
            }
            for o in state.open_objections()
        ],
        # Feature code lives INSIDE this JSON block as a plain string, never appended after it as
        # a fenced code block. `tools/llm.py:_payload` locates the facts block with
        # find("{")/rfind("}"); a trailing fence would sit outside that span and never reach the
        # stub -- or a real model reading the same block for the same reason.
        "feature_code": feature_code,
        "feature_code_note": feature_code_note,
    }
    return json.dumps(facts, indent=2)


def reviewer(state: PipelineState, *, tools: Tools, model: StructuredModel) -> dict[str, Any]:
    run = NodeRun("reviewer")

    if not state.config.reviewer_enabled:
        # No graph conditional -- graph.py stays logic-free -- so the no-op lives here. Both
        # handoff fields are cleared explicitly rather than left at their PipelineState default,
        # matching every other return path below: the router already treats claim=None with the
        # reviewer disabled as "never adjudicated", not an error.
        return {
            "reviewer_claim": None,
            "reviewer_dispositions": {},
            "node_trace": [run.event()],
        }

    if state.spec is None or state.profile is None:
        return {
            **run.failure(
                "no spec/profile: cannot review a run that has not been profiled",
                recoverable=False,
            ),
            "reviewer_claim": None,
            "reviewer_dispositions": {},
        }

    errors: list[PipelineError] = []
    feature_code, feature_code_note, feature_code_error = _feature_code(state, tools)
    if feature_code_error is not None:
        errors.append(feature_code_error)

    try:
        finding = run.record(
            model.generate(
                system=REVIEWER_SYSTEM,
                user=_user_message(
                    state, feature_code=feature_code, feature_code_note=feature_code_note
                ),
                schema=ReviewFinding,
            )
        )
    except Exception as exc:  # noqa: BLE001 - a real client's timeout/rate-limit/connection error
        # must not crash the node, but there is no safe default "claim" to fall back to the way
        # profiler falls back to an empty nomination. A pass that could not run must read as no
        # pass: both handoff fields are cleared so the router sees `pending`, never a stale claim.
        return {
            **run.failure(f"reviewer model call failed: {exc}"),
            "reviewer_claim": None,
            "reviewer_dispositions": {},
        }

    known_columns = {c.name for c in state.profile.columns} - {state.spec.target}
    kept_objections: list[Objection] = []
    rejected: list[str] = []
    dropped_columns: set[str] = set()

    for proposal in finding.objections:
        filtered = [c for c in proposal.columns if c in known_columns]
        dropped_columns |= set(proposal.columns) - set(filtered)
        if proposal.category in COLUMN_SCOPED_CATEGORIES and not filtered:
            # Rejected outright, not downgraded to "other": a column-scoped objection that named
            # only unknown, dropped, or target columns has nothing left to act on, and relabelling
            # it would just guarantee an unresolvable objection and an eventual "exhausted".
            rejected.append(f"{proposal.category} ({proposal.subcategory})")
            continue
        kept_objections.append(
            Objection(
                category=proposal.category,
                subcategory=proposal.subcategory,
                target_node=proposal.target_node,
                columns=filtered,
                evidence=proposal.evidence,
                severity=proposal.severity,
                raised_at_iteration=state.review_iterations,
            )
        )

    if dropped_columns or rejected:
        # One aggregated error for the whole pass, not one per proposal: a model that names three
        # bad columns in one pass should cost the trace one row, not three.
        parts = []
        if dropped_columns:
            parts.append(
                f"unknown/target columns dropped from proposals: {sorted(dropped_columns)}"
            )
        if rejected:
            parts.append(f"objections rejected for having no surviving column: {rejected}")
        errors.append(PipelineError(node="reviewer", message="; ".join(parts)))

    open_ids = {o.id for o in state.open_objections()}
    dispositions: dict[str, Disposition] = {}
    for update in finding.dispositions:
        # Keyed only to ids that were actually open: a hallucinated id, or one naming an objection
        # already closed, cannot land on the handoff field the router folds straight into the
        # durable ReviewPass record.
        if update.objection_id in open_ids:
            dispositions[update.objection_id] = update.disposition
    for oid in open_ids:
        dispositions.setdefault(oid, "not_reviewed")

    result: dict[str, Any] = {
        "objections": kept_objections,
        "reviewer_claim": finding.claim,
        "reviewer_dispositions": dispositions,
        "node_trace": [run.event()],
    }
    if errors:
        result["errors"] = errors
    return result
