"""Freeze one finished run so the walkthrough viewer can replay it.

Two functions, deliberately split so re-rendering never costs another pipeline run:

- `envelope(state, run_root)` runs right after a pipeline finishes, while the run root still
  exists. It writes the RAW record: the full state dump (nulls present -- a halted run and a
  healthy run must be structurally identical, so panels guard on `null`, never on key-absence),
  the results row, the artifact texts, and the fixture answer key. It costs money only in the
  sense that it needs a live run to have happened.
- `timeline(state)` runs at BUILD time, from a state rebuilt out of a committed envelope. It is
  the subtle piece: there is no checkpointer, so intermediate states were never persisted, and
  the replay works only because every `PipelineState` field has a single writer node and
  `node_trace` is appended in execution order. Anything the replay shows is therefore one of
  three kinds, and every `Step` says which:

  - `recorded`: taken from a field that is genuinely per-event (the trace row itself, objections
    via `raised_at_iteration`, the `ReviewPass` history).
  - `reconstructed`: derived from the final state under the single-writer rule, valid only for a
    node's LAST occurrence -- the final value IS what that occurrence wrote.
  - `not_recorded`: an earlier occurrence of a node that later ran again. Its written values were
    overwritten and are gone. The viewer must say so, never show the final value in their place.

This module is viewer-facing contract, not pipeline: nothing in `nodes/` or `graph.py` may import
it. The dependency points one way -- capture reads the state contract, never the reverse.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ds_agents.fixtures import load_fixture
from ds_agents.state import (
    COLUMN_SCOPED_CATEGORIES,
    Contract,
    Disposition,
    NodeName,
    PipelineState,
    ReviewPass,
    ReviewVerdict,
    utc_now,
)

SCHEMA_VERSION = 1

# Mirrors the reasoning behind the artifact store's read cap: the viewer embeds these texts into
# one HTML file, and a single unbounded artifact (a wide importance table, a verbose report) would
# dominate the payload. 20 KB keeps every real feature-code and report artifact intact.
ARTIFACT_TEXT_LIMIT = 20_000

Fidelity = Literal["recorded", "reconstructed", "not_recorded"]


class CapturedArtifact(Contract):
    """One artifact's bytes, frozen beside the state that cites its id."""

    artifact_id: str
    filename: str
    text: str = Field(description="UTF-8 text, capped at ARTIFACT_TEXT_LIMIT. Empty if binary.")
    truncated: bool = False
    binary: bool = Field(
        default=False,
        description="The file did not decode as UTF-8 (a pickled model, say). `text` is empty; "
        "the file existed, which is worth recording even when its content is not embeddable.",
    )


class Capture(Contract):
    """What `--state-json` writes: the raw envelope the build step consumes.

    `state` stays a dict rather than a `PipelineState` so the envelope survives being read by a
    checkout whose state contract has since moved -- the build step calls
    `PipelineState.from_dump(capture.state)` and turns drift into a loud validation error at
    build time instead of a blank panel at view time.
    """

    schema_version: int = SCHEMA_VERSION
    captured_at: datetime = Field(default_factory=utc_now)
    dataset_id: str
    note: str = Field(
        default="",
        description="Why THIS replicate was kept. Model nondeterminism is the whole variance "
        "(the seed fixes only the data split), so an interesting run is found by repeating and "
        "choosing -- and the choice must be on the record: which replicate, out of how many.",
    )
    state: dict[str, Any] = Field(
        description="`state.model_dump(mode='json')`, WITHOUT exclude_none. Explicit nulls are "
        "the contract: a halted run has the same shape as a healthy one."
    )
    results_row: dict[str, Any]
    node_seconds: dict[str, float] = Field(
        description="Carried explicitly because `PipelineState.node_seconds` is a plain property "
        "and does not survive `model_dump()`."
    )
    artifacts: list[CapturedArtifact] = Field(default_factory=list)
    fixture_manifest: dict[str, Any] | None = Field(
        default=None,
        description="tests/fixtures/<id>/manifest.json for fixture runs, None for benchmarks. "
        "The answer key -- safe here because this envelope is a docs deliverable the agents "
        "never read, unlike the report artifact.",
    )


class Step(Contract):
    """One node execution, replayed. The unit the viewer steps through."""

    index: int = Field(ge=0)
    node: NodeName
    occurrence: int = Field(ge=1, description="1-based: which time this node has run so far.")
    is_last_occurrence: bool = Field(
        description="Gates `writes_values` and `detail`: only the last occurrence's writes "
        "survive on the final state. Earlier occurrences are `not_recorded`."
    )
    iteration: int = Field(
        ge=0,
        description="Review iteration in force when this step ran: the number of completed "
        "router passes before it. Matches `Objection.raised_at_iteration` for reviewer steps.",
    )

    # straight off the NodeEvent -- always `recorded`
    started: datetime
    ended: datetime | None
    wall_seconds: float | None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str | None = None

    # the single-writer contract, made visible
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    writes_values: dict[str, Any] | None = Field(
        default=None,
        description="The fields this node owns, valued from the final state. ONLY when "
        "`is_last_occurrence`; None otherwise, because those values are gone.",
    )

    fidelity: Fidelity
    headline: str = Field(description="One glance: what this step did, in plain words.")
    detail: dict[str, Any] | None = Field(
        default=None,
        description="The node's expanded panel content. Shape is per-node; every consumer "
        "guards on None. None when not the last occurrence, and for nodes with nothing to show.",
    )

    # router steps only
    routed_to: str | None = None
    verdict: ReviewVerdict | None = None
    new_objection_ids: list[str] = Field(default_factory=list)
    dispositions: dict[str, Disposition] = Field(default_factory=dict)

    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="PipelineErrors attributed to this node. Errors carry no occurrence stamp, "
        "so on a node that ran more than once they attach to its LAST occurrence -- an "
        "attribution choice, labelled in the viewer, not a recorded fact.",
    )


# Transcribed from the node contract table (docs/ARCHITECTURE.md, "Node contracts").
# `reads` are display strings -- dotted config paths and method-call notation for derived reads
# are deliberate, they name what the node actually asks for. `writes` are literal PipelineState
# field names; `test_capture.py` pins every one to a real field so a rename breaks a test
# instead of blanking a panel. Every node also appends to `errors` and `node_trace`; those are
# infrastructure, not listed per node.
NODE_READS: dict[str, list[str]] = {
    "intake": ["dataset_id", "task_description"],
    "profiler": ["spec", "dataset_id", "config.random_seed"],
    "feature_eng": [
        "spec",
        "profile",
        "split_artifact",
        "task_description",
        "binding_objections('feature_eng')",
        "open_objections('feature_eng')",
    ],
    "modeler": [
        "spec",
        "feature_code_artifact",
        "split_artifact",
        "final_features",
        "task_description",
        "config.random_seed",
        "config.run_id",
        "open_objections('modeler')",
    ],
    "reviewer": [
        "spec",
        "profile",
        "final_features",
        "dropped_features",
        "feature_summary",
        "candidates",
        "chosen_model",
        "top_importances",
        "task_description",
        "review_iterations",
        "open_objections()",
        "config.reviewer_prompt",
        "config.objection_closure",
        "feature_code_artifact (if config.reviewer_sees_code)",
    ],
    "router": [
        "reviewer_claim",
        "reviewer_dispositions",
        "review_iterations",
        "config.loop_cap",
        "config.reviewer_enabled",
        "would_be_open(...)",
    ],
    "reporter": ["everything except harness ground truth"],
}

NODE_WRITES: dict[str, list[str]] = {
    "intake": ["spec"],
    "profiler": ["profile", "split_artifact"],
    "feature_eng": [
        "feature_code_artifact",
        "feature_summary",
        "final_features",
        "dropped_features",
        "skipped_high_cardinality",
    ],
    "modeler": ["candidates", "chosen_model", "importance_artifact", "top_importances"],
    "reviewer": ["objections", "reviewer_claim", "reviewer_dispositions"],
    "router": ["review_iterations", "review_verdict", "review_passes"],
    "reporter": ["report_artifact"],
}


def envelope(state: PipelineState, run_root: Path, *, note: str = "") -> Capture:
    """Freeze `state` and the run root's artifacts into one committable envelope.

    Must be called while `run_root` still exists (the CLI defaults it to a throwaway tempdir).
    Artifact ids are recovered by matching the ids the state cites as filename prefixes --
    the store names files `<artifact_id>-<name>` -- and any file no state field cites is still
    captured, keyed by its own filename, because snippet-dropped intermediates are part of what
    the run did.
    """
    cited = _cited_artifact_ids(state)
    artifacts: list[CapturedArtifact] = []
    artifacts_dir = run_root / "artifacts"
    if artifacts_dir.is_dir():
        for path in sorted(artifacts_dir.iterdir(), key=lambda p: p.name):
            if not path.is_file():
                continue
            artifacts.append(_capture_artifact(path, cited))

    fixture_manifest: dict[str, Any] | None = None
    if state.config.dataset_source == "fixture":
        # SystemExit propagates on purpose -- a fixture id that no longer resolves is operator
        # error (the state was built against a checkout this one no longer matches), not a
        # condition this function should paper over.
        fixture_manifest = load_fixture(state.dataset_id).manifest.model_dump(mode="json")

    return Capture(
        dataset_id=state.dataset_id,
        note=note,
        state=state.model_dump(mode="json"),
        results_row=state.results_row(),
        node_seconds=dict(state.node_seconds),
        artifacts=artifacts,
        fixture_manifest=fixture_manifest,
    )


def timeline(state: PipelineState) -> list[Step]:
    """Replay `state.node_trace` into ordered, fidelity-labelled steps.

    The clocks, pinned by `nodes/router.py` and `nodes/reviewer.py`:
    - The k-th reviewer invocation (0-indexed) runs with `review_iterations == k` and stamps its
      new objections `raised_at_iteration = k`.
    - The k-th completed router pass mints `ReviewPass(iteration = k + 1)` (the router
      increments before minting). Router trace events align to `review_passes` sorted by
      iteration, LEFT-ALIGNED and tolerant of a shortfall: the reviewer-off arm and the
      claim-is-None path mint a router event with no pass at all.
    - A step's `iteration` is the number of router events that precede it in the trace.

    Fidelity per step: `recorded` for reviewer and router steps (their history fields are
    genuinely per-pass); `reconstructed` for the last occurrence of every other node (single
    writer, so the final value is what that occurrence wrote); `not_recorded` for any earlier
    occurrence of a node that ran again.
    """
    full_dump = state.model_dump(mode="json")
    trace = state.node_trace

    total_occurrences: dict[str, int] = {}
    for event in trace:
        total_occurrences[event.node] = total_occurrences.get(event.node, 0) + 1

    # Last trace index per node -- where a node's errors land, per the Step docstring.
    last_index_by_node: dict[str, int] = {}
    for idx, event in enumerate(trace):
        last_index_by_node[event.node] = idx

    errors_by_node: dict[str, list[dict[str, Any]]] = {}
    for error in state.errors:
        # A node named by an error with no trace event at all is skipped -- `last_index_by_node`
        # simply never matches any step's index, so nothing attaches it. Should not happen; a
        # PipelineError is always appended alongside the node's own NodeEvent.
        errors_by_node.setdefault(error.node, []).append(error.model_dump(mode="json"))

    sorted_passes = sorted(state.review_passes, key=lambda p: p.iteration)

    seen_occurrences: dict[str, int] = {}
    steps: list[Step] = []
    for index, event in enumerate(trace):
        node = event.node
        seen_occurrences[node] = seen_occurrences.get(node, 0) + 1
        occurrence = seen_occurrences[node]
        is_last_occurrence = occurrence == total_occurrences.get(node, 0)
        # Number of router events strictly before this one -- the trace slice up to (not
        # including) `index`.
        iteration = sum(1 for e in trace[:index] if e.node == "router")

        reads = list(NODE_READS.get(node, []))
        writes = list(NODE_WRITES.get(node, []))
        writes_values: dict[str, Any] | None = None
        if is_last_occurrence and writes:
            writes_values = {field: full_dump.get(field) for field in writes}

        routed_to: str | None = None
        verdict: ReviewVerdict | None = None
        new_objection_ids: list[str] = []
        dispositions: dict[str, Disposition] = {}
        detail: dict[str, Any] | None = None

        if node == "router":
            fidelity: Fidelity = "recorded"
            # `iteration` here IS the 0-indexed position of this router event among router events
            # (the count of router events before it), which is exactly the alignment index into
            # `sorted_passes`.
            aligned = sorted_passes[iteration] if iteration < len(sorted_passes) else None
            if aligned is None:
                verdict = "pending"
            else:
                routed_to = aligned.routed_to
                dispositions = dict(aligned.dispositions)
                new_objection_ids = list(aligned.new_objection_ids)
                if aligned.claim == "pass":
                    verdict = "pass"
                elif aligned.iteration >= state.config.loop_cap:
                    verdict = "exhausted"
                else:
                    verdict = "block"
            headline = _router_headline(verdict, routed_to, iteration + 1, state.config.loop_cap)
        elif node == "reviewer":
            fidelity = "recorded"
            objections_raised = [
                o.model_dump(mode="json")
                for o in state.objections
                if o.raised_at_iteration == iteration
            ]
            detail = {"objections_raised": objections_raised}
            headline = _reviewer_headline(state, iteration, is_last_occurrence, objections_raised)
        else:
            fidelity = "reconstructed" if is_last_occurrence else "not_recorded"
            if is_last_occurrence:
                detail = _detail_for(node, full_dump, state)
            headline = _headline_for(node, is_last_occurrence, full_dump, state)

        steps.append(
            Step(
                index=index,
                node=node,
                occurrence=occurrence,
                is_last_occurrence=is_last_occurrence,
                iteration=iteration,
                started=event.started,
                ended=event.ended,
                wall_seconds=event.wall_seconds,
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                cost_usd=event.cost_usd,
                model=event.model,
                reads=reads,
                writes=writes,
                writes_values=writes_values,
                fidelity=fidelity,
                headline=headline,
                detail=detail,
                routed_to=routed_to,
                verdict=verdict,
                new_objection_ids=new_objection_ids,
                dispositions=dispositions,
                errors=(
                    errors_by_node.get(node, []) if last_index_by_node.get(node) == index else []
                ),
            )
        )
    return steps


# ---------------------------------------------------------------------------------------------
# Private helpers. Everything below is implementation detail of `envelope` and `timeline` above;
# nothing here is imported by anything but this module's own tests.
# ---------------------------------------------------------------------------------------------


def _cited_artifact_ids(state: PipelineState) -> set[str]:
    """Every artifact id the state cites by name, from the five sources named in `envelope`'s
    docstring. Used to recover an id from a filename the store wrote as `<id>-<name>`."""
    ids: set[str] = set()
    for artifact_id in (
        state.split_artifact,
        state.feature_code_artifact,
        state.importance_artifact,
        state.report_artifact,
    ):
        if artifact_id:
            ids.add(artifact_id)
    for candidate in state.candidates:
        if candidate.model_artifact:
            ids.add(candidate.model_artifact)
    if state.chosen_model is not None and state.chosen_model.model_artifact:
        ids.add(state.chosen_model.model_artifact)
    return ids


def _capture_artifact(path: Path, cited: set[str]) -> CapturedArtifact:
    """One file under `<run_root>/artifacts` -> one `CapturedArtifact`."""
    artifact_id = next((cid for cid in sorted(cited) if path.name.startswith(cid + "-")), path.name)
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return CapturedArtifact(artifact_id=artifact_id, filename=path.name, text="", binary=True)
    truncated = len(text) > ARTIFACT_TEXT_LIMIT
    return CapturedArtifact(
        artifact_id=artifact_id,
        filename=path.name,
        text=text[:ARTIFACT_TEXT_LIMIT] if truncated else text,
        truncated=truncated,
    )


def _fallback_headline(is_last_occurrence: bool) -> str:
    """The honest sentence when there is nothing (more) to say -- degrade, never crash."""
    if is_last_occurrence:
        return "ran (no result recorded)"
    return "ran (values overwritten by a later pass -- not recorded)"


def _headline_for(
    node: str, is_last_occurrence: bool, full_dump: dict[str, Any], state: PipelineState
) -> str:
    """One-glance sentence for a non-reviewer, non-router step. Only ever called with real
    values when `is_last_occurrence`; every branch still guards for a halted run's missing piece.
    """
    if not is_last_occurrence:
        return _fallback_headline(is_last_occurrence)

    if node == "intake":
        spec = full_dump.get("spec")
        if not spec:
            return _fallback_headline(is_last_occurrence)
        return f"chose target {spec['target']!r} -- {spec['task_type']}, scored by {spec['metric']}"

    if node == "profiler":
        profile = full_dump.get("profile")
        if not profile:
            return _fallback_headline(is_last_occurrence)
        n_flags = len(profile.get("leakage_candidates") or [])
        plural = "" if n_flags == 1 else "s"
        return (
            f"profiled {profile.get('n_rows', '?')} rows x {profile.get('n_columns', '?')} "
            f"columns; flagged {n_flags} leakage candidate{plural}"
        )

    if node == "feature_eng":
        final = full_dump.get("final_features")
        if final is None:
            return _fallback_headline(is_last_occurrence)
        dropped = full_dump.get("dropped_features") or []
        forced = _forced_drop_columns(state, "feature_eng")
        forced_note = f" ({len(forced)} forced out by objections)" if forced else ""
        return f"kept {len(final)} features, dropped {len(dropped)}{forced_note}"

    if node == "modeler":
        chosen = full_dump.get("chosen_model")
        if not chosen:
            return _fallback_headline(is_last_occurrence)
        metric = (full_dump.get("spec") or {}).get("metric", "score")
        claimed = chosen.get("claimed_holdout_score")
        claim_part = (
            f"claimed {metric} {claimed:.3f}" if claimed is not None else "no claimed score"
        )
        return f"chose {chosen.get('name', '?')}; {claim_part} on its own holdout"

    if node == "reporter":
        if full_dump.get("report_artifact"):
            return "wrote the run report"
        return _fallback_headline(is_last_occurrence)

    # "generalist" (single-agent ablation) or any future node with no bespoke sentence.
    return "ran"


def _reviewer_headline(
    state: PipelineState, iteration: int, is_last_occurrence: bool, objections_raised: list[Any]
) -> str:
    """Every reviewer occurrence is `recorded` data, so this runs regardless of
    `is_last_occurrence` -- only the CLAIM needs a fallback, from the aligned `ReviewPass` when one
    exists, or from `reviewer_claim` on the final occurrence when no pass ever completed (the
    reviewer crashed, or the run halted before the router ran).
    """
    aligned: ReviewPass | None = next(
        (p for p in state.review_passes if p.iteration == iteration + 1), None
    )
    if aligned is not None:
        claim: str | None = aligned.claim
    elif is_last_occurrence:
        claim = state.reviewer_claim
    else:
        claim = None

    n = len(objections_raised)
    raised_part = "raised nothing" if n == 0 else f"raised {n} objection{'' if n == 1 else 's'}"
    if claim is None:
        return f"{raised_part} (claim not recorded)"
    return f"{raised_part} and claimed {claim}"


def _router_headline(
    verdict: ReviewVerdict | None, routed_to: str | None, pass_number: int, loop_cap: int
) -> str:
    if verdict == "pending":
        return "no reviewer claim to route (pending)"
    destination = routed_to or "reporter"
    dest_phrase = "sent to reporter" if destination == "reporter" else f"sent back to {destination}"
    return f"verdict {verdict} -- {dest_phrase} (pass {pass_number} of cap {loop_cap})"


def _detail_for(
    node: str, full_dump: dict[str, Any], state: PipelineState
) -> dict[str, Any] | None:
    """The expanded-panel payload for a non-reviewer, non-router step's LAST occurrence."""
    if node == "intake":
        return {"spec": full_dump.get("spec")}
    if node == "profiler":
        return {
            "profile": full_dump.get("profile"),
            "split_artifact": full_dump.get("split_artifact"),
        }
    if node == "feature_eng":
        forced = _forced_drop_columns(state, "feature_eng")
        return {
            "final_features": full_dump.get("final_features"),
            "dropped_features": full_dump.get("dropped_features"),
            "skipped_high_cardinality": full_dump.get("skipped_high_cardinality"),
            "feature_summary": full_dump.get("feature_summary"),
            "feature_code_artifact": full_dump.get("feature_code_artifact"),
            "forced_drop_columns": forced,
        }
    if node == "modeler":
        return {
            "candidates": full_dump.get("candidates"),
            "chosen_model": full_dump.get("chosen_model"),
            "top_importances": full_dump.get("top_importances"),
            "importance_artifact": full_dump.get("importance_artifact"),
        }
    if node == "reporter":
        return {"report_artifact": full_dump.get("report_artifact")}
    # "generalist" or any future node with nothing bespoke to show.
    return None


def _forced_drop_columns(state: PipelineState, target_node: str) -> list[str]:
    """Columns a binding objection is forcing out of `target_node`'s matrix, for display.

    Deliberately NOT a call to `PipelineState.binding_objections`. `tests/test_state.py`'s
    `TestTheSingleCallerInvariant` walks every `.py` file under `src/ds_agents/` (not just
    `nodes/`) and pins that method to its one existing caller, `nodes/feature_eng.py::_forced_drops`
    -- the docstring on `binding_objections` says as much: "the invariant to check before adding a
    second." Whether a read-only viewer computing the same predicate for a finished run's report
    counts as the kind of second caller that invariant means to prevent is `state.py`'s call to
    make, not this module's -- so this reproduces the identical logic (`latest_dispositions()` and
    `effective_target()`, both public and uncontended) rather than either breaking the test or
    quietly loosening it, INCLUDING the category filter `feature_eng._forced_drops` applies on
    top of `binding_objections`: only a column-scoped objection forces a drop, so a non-scoped
    category that happens to name a column must not display as one. This is a recorded contract
    mismatch, not a silent workaround; NEXT.md carries the question of a blessed path.
    """
    releasing = (
        {"withdrawn"}
        if state.config.forced_drop_release == "withdrawn_only"
        else {"resolved", "withdrawn"}
    )
    latest = state.latest_dispositions()
    released = {oid for oid, disposition in latest.items() if disposition in releasing}
    return sorted(
        {
            column
            for objection in state.objections
            if objection.id not in released
            and objection.category in COLUMN_SCOPED_CATEGORIES
            and state.effective_target(objection) == target_node
            for column in objection.columns
        }
    )
