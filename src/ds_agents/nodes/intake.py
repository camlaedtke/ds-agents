"""intake: decide what the run is trying to do.

Reads `dataset_id`, `task_description`, `config`. Writes `spec`. Tool: `read_artifact` only.

Intake is the one node that has to name the target, which means it has to see the columns. It gets
them from the dataset's artifact metadata rather than from `run_python`, so the first node in the
graph cannot execute code before anything has been profiled. See docs/DECISIONS.md 2026-08-26.

It deliberately does NOT look at the target's relationship to anything. Noticing that a column
encodes the answer is the profiler's job to flag and the reviewer's to argue, and an intake that
quietly dropped a suspicious column would make the leak invisible to the thing being measured.
"""

import json
from typing import Any

from pydantic import ValidationError

from ds_agents.nodes._run import NodeRun
from ds_agents.state import Contract, Metric, PipelineState, TaskSpec, TaskType
from ds_agents.tools.llm import StructuredModel
from ds_agents.tools.local import dataset_artifact_id
from ds_agents.tools.protocol import ToolError, Tools

SAMPLE_BYTES = 2000

INTAKE_SYSTEM = """You are the intake step of a tabular data science pipeline.

Given a dataset's schema and a one-line task description, decide the prediction target, the task \
type, and the evaluation metric. Return only the structured decision.

Rules:
- The target must be one of the listed columns, spelled exactly.
- Choose the metric that matches the task type and the description. Prefer roc_auc for binary \
classification unless the description asks for something else.
- Use `split_strategy` "temporal" only if a column clearly orders the rows in time, and \
"grouped" only if rows repeat per entity. Either one requires `split_key`.
- Do not comment on data quality, suspicious columns, or leakage. Another step does that."""


class IntakeDecision(Contract):
    """What intake asks the model for. Mirrors `TaskSpec` plus a rationale for the trace."""

    target: str
    task_type: TaskType
    metric: Metric
    split_strategy: str = "stratified"
    split_key: str | None = None
    positive_class: str | None = None
    rationale: str = ""


def _user_message(state: PipelineState, meta_extra: dict[str, Any], sample: str) -> str:
    facts = {
        "task_description": state.task_description,
        "dataset_id": state.dataset_id,
        "n_rows": meta_extra.get("n_rows"),
        "columns": meta_extra.get("columns", []),
        "dtypes": meta_extra.get("dtypes", {}),
        "n_unique": meta_extra.get("n_unique", {}),
    }
    return f"{json.dumps(facts, indent=2)}\n\nFirst rows as CSV:\n{sample}"


def intake(state: PipelineState, *, tools: Tools, model: StructuredModel) -> dict[str, Any]:
    run = NodeRun("intake")
    try:
        payload = tools.read_artifact(dataset_artifact_id(state.dataset_id), max_bytes=SAMPLE_BYTES)
    except ToolError as exc:
        # Unrecoverable: with no schema there is nothing for a later node to work from.
        return run.failure(f"could not read dataset artifact: {exc}", recoverable=False)

    columns: list[str] = payload.meta.extra.get("columns", [])
    sample = "\n".join(payload.content.splitlines()[:6])

    try:
        decision = run.record(
            model.generate(
                system=INTAKE_SYSTEM,
                user=_user_message(state, payload.meta.extra, sample),
                schema=IntakeDecision,
            )
        )
    except Exception as exc:  # noqa: BLE001
        # Broad on purpose: a real client's timeout or rate-limit error must still leave a
        # PipelineError and a NodeEvent, or the run vanishes from the results instead of
        # appearing as the failure it was.
        return run.failure(f"intake produced no usable decision: {exc}", recoverable=False)

    # The model naming a column that does not exist is a real and common failure. Caught here it
    # is one error row; carried forward it is a profiler crash blamed on the wrong node.
    if decision.target not in columns:
        return run.failure(
            f"target {decision.target!r} is not a column in {state.dataset_id!r}",
            recoverable=False,
        )
    if decision.split_key is not None and decision.split_key not in columns:
        return run.failure(
            f"split_key {decision.split_key!r} is not a column in {state.dataset_id!r}",
            recoverable=False,
        )

    try:
        spec = TaskSpec(
            target=decision.target,
            task_type=decision.task_type,
            metric=decision.metric,
            split_strategy=decision.split_strategy,
            split_key=decision.split_key,
            positive_class=decision.positive_class,
        )
    except ValidationError as exc:
        return run.failure(f"intake decision is not a valid TaskSpec: {exc}", recoverable=False)

    return {"spec": spec, "node_trace": [run.event()]}
