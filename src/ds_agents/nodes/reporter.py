"""reporter: render one human-readable markdown report for the run and cite it as an artifact.

Reads everything on `PipelineState` except `planted_leakage_columns`, `verified_holdout_score`,
and the two baseline points. Writes `report_artifact` (plus `node_trace`, and `errors` on the one
failure path). Tools: `write_artifact` only -- the report cites artifact IDs
(`feature_code_artifact`, `importance_artifact`) rather than inlining their contents. No model
call.

Two properties matter more than the prettiness of the markdown:

1. This is the last node, and it is reached on the hard-failure path where `spec`, `profile`,
   `chosen_model`, and `final_features` are all `None`. The eval harness needs a row for every
   dataset including the ones that blew up -- if this node raises, the hardest datasets vanish
   from the results and every published table biases upward. Every section below guards on
   `None` rather than assuming an upstream node ran.
2. This node must never render ground truth. `planted_leakage_columns`, `verified_holdout_score`,
   and the baseline columns are harness-written answer keys that happen to be in scope because the
   node receives the whole state. In the real pipeline they are still `None` here (the harness
   fills them in after the graph ends), but nothing below reads them regardless -- writing them
   into an artifact would leak the answer key into a file a single-generalist arm might read
   through the artifact store.
"""

from typing import Any

from ds_agents.nodes._run import NodeRun
from ds_agents.state import ModelResult, NodeEvent, Objection, PipelineState
from ds_agents.tools.llm import StructuredModel
from ds_agents.tools.protocol import ToolError, Tools

# A fit error rendered whole turns one markdown table cell into a paragraph. The full text is
# never lost: it is on `ModelResult.fit_error` and in the `PipelineError` the modeler raised.
MODEL_NOTE_LIMIT = 120

REPORT_HEADER = (
    "Written by the reporter node. Nothing here is a grade: `claimed_holdout_score` is what the "
    "system said about itself, scored on a holdout the system chose. The independent number is "
    "written by the harness after the graph ends and is deliberately not in this file."
)


def _fmt(value: Any) -> str:
    """None renders as `-`, floats to 4 places. Every table cell goes through this."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(out)


def _run_section(state: PipelineState) -> list[str]:
    config = state.config
    rows = [
        ["run_id", config.run_id],
        ["dataset_id", state.dataset_id],
        ["arm", config.arm],
        ["reviewer_enabled", config.reviewer_enabled],
        ["reviewer_model", config.reviewer_model],
        ["reviewer_prompt", config.reviewer_prompt],
        ["reviewer_sees_code", config.reviewer_sees_code],
        ["naming", config.naming],
        ["loop_cap", config.loop_cap],
        ["random_seed", config.random_seed],
        ["started_at", state.started_at],
        # No wall_seconds row: `ended_at` is stamped by `run_pipeline` after this node returns,
        # so it is always None here. Reporting "-" for the run's duration inside the run reads
        # like a bug rather than the self-reference it is.
    ]
    return ["## Run", "", _table(["field", "value"], rows), ""]


def _task_section(state: PipelineState) -> list[str]:
    lines = ["## Task", ""]
    spec = state.spec
    if spec is None:
        lines += ["_intake produced no spec; see Errors._", ""]
        return lines
    rows = [
        ["target", spec.target],
        ["task_type", spec.task_type],
        ["metric", spec.metric],
        ["greater_is_better", spec.greater_is_better],
        ["split_strategy", spec.split_strategy],
        ["split_key", spec.split_key],
        ["positive_class", spec.positive_class],
    ]
    lines += [_table(["field", "value"], rows), ""]
    return lines


def _profile_section(state: PipelineState) -> list[str]:
    lines = ["## Data profile", ""]
    profile = state.profile
    if profile is None:
        lines += ["_profiler did not run; see Errors._", ""]
        return lines
    lines += [f"{profile.n_rows} rows x {profile.n_columns} columns.", ""]
    if profile.target_balance:
        balance = ", ".join(f"{k}: {_pct(v)}" for k, v in profile.target_balance.items())
        lines += [f"Target balance: {balance}.", ""]
    rows = [[c.name, c.dtype, _pct(c.missing_fraction), c.n_unique] for c in profile.columns]
    table = _table(["column", "dtype", "missing", "distinct"], rows)
    lines += [table if table else "No columns recorded.", ""]
    return lines


def _leakage_section(state: PipelineState) -> list[str]:
    lines = ["## Leakage candidates (profiler)", ""]
    profile = state.profile
    if profile is None:
        lines += ["_profiler did not run; see Errors._", ""]
        return lines
    if not profile.leakage_candidates:
        lines += ["None nominated.", ""]
        return lines
    rows = [[c.column, c.suspicion, c.evidence, c.reason] for c in profile.leakage_candidates]
    lines += [_table(["column", "suspicion", "evidence", "reason"], rows), ""]
    return lines


def _feature_section(state: PipelineState) -> list[str]:
    lines = ["## Feature engineering", ""]
    if state.final_features is None:
        lines += ["_feature_eng did not run; see Errors._", ""]
        return lines
    lines += [f"feature_code_artifact: {_fmt(state.feature_code_artifact)}", ""]
    if state.feature_summary:
        lines += [state.feature_summary, ""]
    kept = ", ".join(state.final_features) if state.final_features else "-"
    dropped = ", ".join(state.dropped_features) if state.dropped_features else "-"
    lines += [
        f"Kept ({len(state.final_features)}): {kept}",
        f"Dropped ({len(state.dropped_features)}): {dropped}",
        "",
    ]
    return lines


def _model_note(candidate: ModelResult) -> str:
    # Two different facts, kept separate: `fit_error` is why the estimator could not be fit at
    # all; empty `cv_scores` with no error is a fit that produced no score, a different cause.
    #
    # Flattened and escaped BEFORE truncating, since this is the one cell in the report whose
    # content comes from an exception rather than from a field this repo shapes. `_table` joins
    # cells on `|` and `_fmt` escapes nothing, so a multi-line sklearn message would silently
    # break the row it renders into -- truncation alone doesn't fix that, since a newline inside
    # the first 120 characters breaks the table just as thoroughly as one after them.
    if candidate.fit_error:
        flat = " ".join(candidate.fit_error.split()).replace("|", "\\|")
        return f"no successful fit: {flat[:MODEL_NOTE_LIMIT]}"
    return "no successful fit" if not candidate.cv_scores else "-"


def _model_section(state: PipelineState) -> list[str]:
    lines = ["## Modelling", ""]
    if not state.candidates and state.chosen_model is None:
        lines += ["_modeler did not run; see Errors._", ""]
        return lines

    if state.candidates:
        rows = [
            [
                c.name,
                c.cv_mean,
                ", ".join(_fmt(s) for s in c.cv_scores) if c.cv_scores else "-",
                c.claimed_holdout_score,
                _model_note(c),
            ]
            for c in state.candidates
        ]
        lines += [
            _table(["candidate", "cv mean", "cv scores", "claimed holdout", "note"], rows),
            "",
        ]

    metric = state.spec.metric if state.spec else None
    if state.chosen_model is None:
        lines += ["Chosen: none; see Errors.", ""]
    else:
        lines += [
            f"Chosen: {state.chosen_model.name}.",
            f"Claimed {_fmt(metric)} on the agents' holdout (a claim, not a verified score): "
            f"{_fmt(state.chosen_model.claimed_holdout_score)}.",
            "",
        ]

    if state.top_importances:
        rows = [[name, value] for name, value in state.top_importances]
        lines += [
            "Permutation importance (source columns, agents' holdout):",
            "",
            _table(["column", "importance"], rows),
            "",
        ]
    else:
        lines += ["No importances recorded.", ""]
    lines += [f"importance_artifact: {_fmt(state.importance_artifact)}", ""]
    return lines


def _objection_row(o: Objection, status: str) -> list[Any]:
    return [
        o.id,
        o.category,
        o.subcategory,
        o.target_node,
        ", ".join(o.columns) if o.columns else "-",
        o.severity,
        status,
        o.evidence,
    ]


def _review_section(state: PipelineState) -> list[str]:
    lines = ["## Review", ""]
    lines += [
        f"Verdict: {state.review_verdict}. "
        f"Iterations: {state.review_iterations} of {state.config.loop_cap}. "
        f"Last claim: {_fmt(state.reviewer_claim)}.",
        "",
    ]
    if not state.config.reviewer_enabled:
        lines += ["_The reviewer was disabled for this run._", ""]

    # The fold itself lives on `PipelineState.latest_dispositions()`; this keeps the whole
    # history rather than filtering down to what is still open, because the report needs to show
    # resolved and withdrawn objections too, distinguishably from ones the reviewer never
    # revisited.
    status = dict(state.latest_dispositions())
    if not state.objections:
        lines += ["No objections raised.", ""]
    else:
        rows = [_objection_row(o, status.get(o.id, "not_reviewed")) for o in state.objections]
        headers = [
            "id",
            "category",
            "subcategory",
            "target",
            "columns",
            "severity",
            "status",
            "evidence",
        ]
        lines += [_table(headers, rows), ""]

    if state.review_passes:
        rows = [
            [
                rp.iteration,
                rp.claim,
                rp.routed_to,
                ", ".join(f"{k}:{v}" for k, v in rp.dispositions.items())
                if rp.dispositions
                else "-",
            ]
            for rp in sorted(state.review_passes, key=lambda r: r.iteration)
        ]
        lines += [_table(["iteration", "claim", "routed_to", "dispositions"], rows), ""]
    return lines


def _errors_section(state: PipelineState) -> list[str]:
    lines = ["## Errors", ""]
    if not state.errors:
        lines += ["None.", ""]
        return lines
    rows = [[e.node, e.recoverable, e.message] for e in state.errors]
    lines += [_table(["node", "recoverable", "message"], rows), ""]
    return lines


def _trace_row(e: NodeEvent) -> list[Any]:
    return [
        e.node,
        e.wall_seconds,
        _fmt(e.model),
        f"{e.input_tokens}/{e.output_tokens}",
        e.cost_usd,
    ]


def _trace_section(state: PipelineState) -> list[str]:
    lines = ["## Node trace", ""]
    if not state.node_trace:
        lines += ["No node executions recorded.", ""]
        return lines
    rows = [_trace_row(e) for e in state.node_trace]
    total_seconds = sum(e.wall_seconds for e in state.node_trace if e.wall_seconds is not None)
    rows.append(["total", total_seconds, "-", "-", state.total_cost_usd])
    lines += [_table(["node", "seconds", "model", "in/out tokens", "cost"], rows), ""]
    # The reporter's own event is minted after this renders, so it cannot appear in a table it is
    # writing. The results row the harness emits is built from the finished state and does include
    # it, so the two disagree by one row on purpose.
    lines += [
        "_The reporter's own event is not in this table: it is minted after the report is "
        "written. The harness's results row is built from the finished state and includes it._",
        "",
    ]
    return lines


def _render(state: PipelineState) -> str:
    lines: list[str] = [f"# ds-agents run report -- {state.dataset_id}", "", REPORT_HEADER, ""]
    lines += _run_section(state)
    lines += _task_section(state)
    lines += _profile_section(state)
    lines += _leakage_section(state)
    lines += _feature_section(state)
    lines += _model_section(state)
    lines += _review_section(state)
    lines += _errors_section(state)
    lines += _trace_section(state)
    return "\n".join(lines)


def reporter(state: PipelineState, *, tools: Tools, model: StructuredModel) -> dict[str, Any]:
    run = NodeRun("reporter")
    content = _render(state)

    try:
        artifact_id = tools.write_artifact(
            name=f"report-{state.dataset_id}.md",
            content=content,
            kind="report",
            extra={
                "run_id": state.config.run_id,
                "review_verdict": state.review_verdict,
                "chosen_model": state.chosen_model.name if state.chosen_model else None,
            },
        )
    except ToolError as exc:
        return run.failure(f"could not write report artifact: {exc}")

    return {"report_artifact": artifact_id, "node_trace": [run.event()]}
