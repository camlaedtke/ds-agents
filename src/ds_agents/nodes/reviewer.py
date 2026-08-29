"""reviewer: argue with the run so far and hand the router a claim.

Reads `spec`, `profile`, `feature_summary`, `final_features`, `dropped_features`, `candidates`,
`chosen_model`, `top_importances`, `task_description`, `open_objections()`, `review_iterations`,
`config.*`; `feature_code_artifact` content only when `config.reviewer_sees_code` (via
`read_artifact`). Writes `objections` (append), `reviewer_claim`, `reviewer_dispositions`,
`node_trace`, and `errors` conditionally. Never `review_verdict`, `review_iterations`, or
`review_passes` -- those belong to the router, which is the sole authority that turns a claim into
an outcome. Tools: `read_artifact` only. One `model.generate` call per pass, and a second one only
on the block-retry path below -- both booked to the same `NodeEvent`, so one pass stays one row in
the cost table.

The reviewer is a model under test, not a trusted judge, so two things here are deliberately not
the reviewer's call:

1. **A crashed pass reads as no pass.** If the FIRST model call fails, the node does not fall back
   to a safe default claim the way `profiler` and `feature_eng` fall back to an empty nomination --
   there is no safe default "claim". Instead both handoff fields (`reviewer_claim`,
   `reviewer_dispositions`) are cleared on every return path, including this one, so the router
   sees exactly what it would see from a disabled reviewer: `pending`, not a stale claim left over
   from whatever the field held before this node ran. A failed *retry* is the one exception and the
   opposite case: the first response is a completed pass, so it stands rather than being discarded.
2. **The claim is never repaired.** A `claim: "block"` that ends up with nothing open after
   filtering and dispositions is not silently promoted to `pass` here -- that is the router's
   documented terminal-block path (an error, routed to the reporter), not something this node gets
   to paper over. What the node does instead, as of 2026-08-29, is ASK AGAIN once: such a block is
   a dead end the router will refuse, so the model is shown the reasons its objections did not
   survive and given one chance to name something real or claim `pass` itself. The distinction is
   the point -- re-asking the model is not the node overruling it, and a second empty answer still
   reaches the router as the `block` the model claimed. See `_nothing_to_act_on`.

Column filtering follows the profiler's own rule: a proposed objection's columns are checked
against the profile's columns minus the target, and a column-scoped category (`leakage`,
`contamination`, `implausible_importance`) left with no surviving column is REJECTED outright, not
downgraded to `other` -- downgrading would produce an un-actionable objection nobody can close,
which guarantees `exhausted` rather than giving the model a real path to `pass`.
"""

import json
from dataclasses import dataclass
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

# Appended to REVIEWER_SYSTEM under `reviewer_prompt="which_column"`, never substituted for it.
# The observed failure this targets, live on 2026-08-27: at a claimed roc_auc of 0.9886 the
# reviewer raised `implausible_importance` naming the two NOISE columns, whose importance was near
# zero, while the planted traps sat at the top of the same ranking. It had the number that says
# the score is too good and searched the wrong end of the list. This asks the question nothing in
# the base prompt asks. It names no fixture, no column and no trap type -- a hint about WHERE the
# answer is would make the ablation arm measure the hint.
WHICH_COLUMN_RULE = """
- If the claimed holdout score is higher than the task plausibly supports, do not stop at saying \
the number is not credible. Say which column produced it: read `top_importances` from the top and \
name the highest-importance column or columns the score depends on, in a `leakage`, \
`contamination` or `implausible_importance` objection. A column with near-zero importance did not \
cause a high score, so it is not an answer to this question."""


# Appended under `objection_closure="on"`, after WHICH_COLUMN_RULE when both are on, never
# substituted for either. The observed failure this targets, live on 2026-08-28: across three
# diagnostic runs the reviewer dispositioned nothing `resolved`. In the clearest of them the
# pipeline dropped both planted columns and the claimed roc_auc fell 0.986 -> 0.823; the reviewer
# wrote that the fall "is consistent with removing leakage" and held the objection open anyway, on
# the grounds that the columns "were never validated as non-leaking, only removed". That is an
# unfalsifiable standard, and a reviewer holding one can never let a run pass: every run grinds to
# the cap, and `exhausted` stops being evidence that the fix did not land. The reviewer had no
# termination condition it could check, so it invented one it could never meet.
#
# The rule points only at `final_features`, a field the reviewer is already shown, and names no
# fixture, no column and no trap type -- the same line WHICH_COLUMN_RULE draws between repairing a
# prompt and injecting the answer. It states the standard, not the answer.
#
# What is deliberately NOT here: a bullet saying `withdrawn` is the only disposition that lets the
# pipeline put a column back. That is true as of the sticky-drop fix, and it is a pipeline
# mechanic, not an observable field; putting it here would break the rule this comment just
# claimed. REVIEWER_SYSTEM already defines `withdrawn` correctly, and `objections_withdrawn` is on
# the results row so that whether the reviewer finds the escape hatch unaided is measured rather
# than assumed.
CLOSURE_RULE = """
- An objection about a column is answered when that column is no longer in the matrix. Before you \
disposition, check each open objection's `columns` against `final_features`. If none of them \
appear there, the column is gone and the objection is `resolved`. Do not hold it open on the \
grounds that the column was never proved harmless: a column absent from `final_features` cannot \
affect the model, and a standard that no evidence could satisfy is not a review.
- Two things are not grounds for `resolved`. Asking for the change is not the change -- resolve on \
what `final_features` contains now, never on the fact that you raised the objection or that a \
later step said it would act. And a column listed in `final_features` is still in the matrix \
whatever any summary says; while it is there, that objection is `still_open`."""


# The stable prefix on every error the block-retry writes. Results rows carry error text as of
# 2026-08-29, so this string is how a cell counts how often the bug fired and how often the retry
# rescued it -- there is deliberately no `block_retries` column, because a counter derived by
# string-matching our own messages is the kind of metric state.py's rule 2 rejects.
BLOCK_RETRY_PREFIX = "block-retry"


def _system_prompt(state: PipelineState) -> str:
    """`base` with nothing appended must stay byte-identical to what every run before 2026-08-28
    used, and `base + WHICH_COLUMN_RULE` byte-identical to the reviewer-ablation and routing cells,
    or rows written under different sessions stop being comparable.

    The two rules are independent conditions on the frozen config, appended in a fixed order.
    One prompt with optional rules appended, never a second code path.
    """
    prompt = REVIEWER_SYSTEM
    if state.config.reviewer_prompt == "which_column":
        prompt += WHICH_COLUMN_RULE
    if state.config.objection_closure == "on":
        prompt += CLOSURE_RULE
    return prompt


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


def _user_message(
    state: PipelineState,
    *,
    feature_code: str | None,
    feature_code_note: str,
    retry_reason: str | None = None,
) -> str:
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
    if retry_reason is not None:
        # Present only on the retry, and absent -- not null -- on the first call, so the first call
        # stays byte-identical to every run this project has published. Inside the facts block for
        # the same reason `feature_code` is: `_payload` reads find("{")..rfind("}"), so a
        # correction appended after the JSON would never reach the model that needs it.
        facts["retry_reason"] = retry_reason
        # Only what the reviewer is already shown. The profile's columns minus the target names no
        # fixture and no trap: it is the same set the filter above checks proposals against, handed
        # over so a second attempt can name something that will survive.
        facts["known_columns"] = sorted(
            {c.name for c in state.profile.columns} - {state.spec.target}
        )
    return json.dumps(facts, indent=2)


@dataclass(frozen=True)
class _Adjudication:
    """One model response, filtered: what the node would return if it stopped here.

    A value rather than four loose locals because the filtering has to run twice -- once on the
    first response and once on the retry's -- and two copies of that loop would be two chances for
    them to disagree about what survives.
    """

    claim: Literal["pass", "block"]
    objections: list[Objection]
    dispositions: dict[str, Disposition]
    filter_error: PipelineError | None
    rejected: list[str]


def _adjudicate(state: PipelineState, finding: ReviewFinding) -> _Adjudication:
    """Filter one `ReviewFinding` against the profile: the column rule, then the disposition keys.

    Pure -- it writes nothing and calls no model -- so a response can be adjudged, judged
    unactionable, and thrown away for a second one without leaving anything behind.
    """
    assert state.spec is not None and state.profile is not None
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

    filter_error = None
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
        filter_error = PipelineError(node="reviewer", message="; ".join(parts))

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

    return _Adjudication(
        claim=finding.claim,
        objections=kept_objections,
        dispositions=dispositions,
        filter_error=filter_error,
        rejected=rejected,
    )


def _nothing_to_act_on(state: PipelineState, adjudged: _Adjudication) -> bool:
    """The router's question, asked one node early: is this `block` a dead end?

    `would_be_open` is the router's own predicate, so this cannot drift from what
    `_route_for_block` will decide about the very same pass. It is deliberately blind to HOW the
    open set came to be empty -- all three causes (every objection filtered away, no objection
    raised at all, this pass's dispositions closing the last one) reach the reporter identically,
    and the teed logs that would have said which one dominates live were never kept.
    """
    return adjudged.claim == "block" and not state.would_be_open(
        adding=adjudged.objections, dispositions=adjudged.dispositions
    )


def _retry_reason(adjudged: _Adjudication) -> str:
    """What the retry tells the model about its own last answer.

    Names the rejections when there were any, because "you named a column that does not exist" and
    "you claimed block and raised nothing" are different mistakes and only the model can tell which
    one it made.
    """
    reason = (
        "Your last response claimed 'block', but once filtering and your dispositions were "
        "applied nothing was left open for feature_eng or modeler to act on, so there is nowhere "
        "to route the run and the block would be discarded."
    )
    if adjudged.rejected:
        reason += (
            f" These objections were rejected for naming no column that exists in this dataset: "
            f"{adjudged.rejected}."
        )
    return (
        f"{reason} Either raise an objection naming at least one column from known_columns below, "
        f"or claim 'pass'."
    )


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
                system=_system_prompt(state),
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

    adjudged = _adjudicate(state, finding)
    # Recorded per adjudication rather than once at the end, because the retry below replaces
    # `adjudged` wholesale: appending only the surviving one would throw away the very diagnostic
    # that says WHY the retry fired, on exactly the passes where the retry worked.
    if adjudged.filter_error is not None:
        errors.append(adjudged.filter_error)

    if _nothing_to_act_on(state, adjudged):
        # The router's own question, asked one node early. Re-ask the model once with what its
        # answer ran into; see BLOCK_RETRY_PREFIX for why this is a retry and not a repair.
        errors.append(
            PipelineError(
                node="reviewer",
                message=f"{BLOCK_RETRY_PREFIX}: claimed 'block' with nothing left for feature_eng "
                f"or modeler to act on; re-asking once",
            )
        )
        try:
            second = run.record(
                model.generate(
                    system=_system_prompt(state),
                    user=_user_message(
                        state,
                        feature_code=feature_code,
                        feature_code_note=feature_code_note,
                        retry_reason=_retry_reason(adjudged),
                    ),
                    schema=ReviewFinding,
                )
            )
        except Exception as exc:  # noqa: BLE001 - same reason as the first call, one step milder:
            # the first response is intact, so a failed repair falls back to it rather than
            # discarding a pass the model did complete.
            errors.append(
                PipelineError(
                    node="reviewer",
                    message=f"{BLOCK_RETRY_PREFIX}: the retry call failed: {exc}; keeping the "
                    f"first response",
                )
            )
        else:
            # Wholesale replacement, not a merge of the two responses: one pass is one adjudication
            # act, and splicing a claim from one call onto dispositions from another would report
            # something the model never said. The cost of that choice is real and deliberate -- a
            # disposition the first call made is gone, so an objection it resolved reverts to
            # `not_reviewed` and stays open, which is conservative and agrees with the block the
            # model is still claiming.
            adjudged = _adjudicate(state, second)
            if adjudged.filter_error is not None:
                errors.append(adjudged.filter_error)
            resolved = not _nothing_to_act_on(state, adjudged)
            errors.append(
                PipelineError(
                    node="reviewer",
                    message=f"{BLOCK_RETRY_PREFIX}: the retry "
                    + (
                        f"produced {len(adjudged.objections)} actionable objection(s)"
                        if resolved
                        else "still produced nothing actionable; the router will refuse this block"
                    ),
                )
            )

    result: dict[str, Any] = {
        "objections": adjudged.objections,
        "reviewer_claim": adjudged.claim,
        "reviewer_dispositions": adjudged.dispositions,
        "node_trace": [run.event()],
    }
    if errors:
        result["errors"] = errors
    return result
