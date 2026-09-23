"""Test doubles for the tool surface and the model client.

Node tests use these rather than `LocalTools`, so a node test fails for a reason inside the node.
A node test that shells out to a real subprocess is testing pandas.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from ds_agents import split_manifest
from ds_agents.cli import _run_state
from ds_agents.fixtures import load_fixture
from ds_agents.runnable import Runnable
from ds_agents.state import ArtifactId, PipelineState
from ds_agents.tools.llm import Completion, StubModel, _payload
from ds_agents.tools.protocol import ArtifactMeta, ArtifactPayload, RunResult, ToolError

TOY_CSV = Path(__file__).parent / "fixtures" / "toy" / "toy.csv"
TOY_COLUMNS = [
    "customer_id",
    "tenure_months",
    "monthly_charges",
    "support_tickets_90d",
    "region",
    "plan_tier",
    "account_status_code",
    "churned",
]


class FakeTools:
    """Canned `run_python` results, in call order, and an in-memory artifact store."""

    def __init__(
        self,
        run_results: list[RunResult | Exception] | None = None,
        artifacts: dict[ArtifactId, ArtifactPayload] | None = None,
    ) -> None:
        self.run_results = list(run_results or [])
        self.artifacts = dict(artifacts or {})
        self.code_run: list[str] = []
        self.metrics: list[tuple[str, str, float]] = []

    def run_python(self, code: str, timeout_s: int = 60) -> RunResult:
        self.code_run.append(code)
        if not self.run_results:
            raise AssertionError("node made more run_python calls than the test scripted")
        result = self.run_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def read_artifact(
        self, artifact_id: ArtifactId, max_bytes: int | None = None
    ) -> ArtifactPayload:
        if artifact_id not in self.artifacts:
            raise ToolError(f"no artifact {artifact_id!r}")
        return self.artifacts[artifact_id]

    def write_artifact(
        self,
        name: str,
        content: str,
        kind: str = "text",
        extra: dict[str, Any] | None = None,
    ) -> ArtifactId:
        artifact_id = f"art-{len(self.artifacts):03d}"
        self.artifacts[artifact_id] = ArtifactPayload(
            meta=ArtifactMeta(
                id=artifact_id, name=name, kind=kind, n_bytes=len(content), extra=extra or {}
            ),
            content=content,
        )
        return artifact_id

    def log_metric(self, run_id: str, name: str, value: float) -> None:
        self.metrics.append((run_id, name, float(value)))


class ScriptedModel:
    """Returns a preset value per schema and remembers the prompts it was given.

    Raising on an unscripted schema is deliberate: a node quietly acquiring a second model call
    should break its test, because that is a cost and a failure mode the trace has to show.
    """

    def __init__(self, answers: dict[type[BaseModel], BaseModel | Exception], name: str = "fake"):
        self.answers = answers
        self.name = name
        self.calls: list[tuple[str, str, type[BaseModel]]] = []

    def generate[M: BaseModel](self, *, system: str, user: str, schema: type[M]) -> Completion[M]:
        self.calls.append((system, user, schema))
        if schema not in self.answers:
            raise AssertionError(f"node asked for {schema.__name__}, which the test did not script")
        answer = self.answers[schema]
        if isinstance(answer, Exception):
            raise answer
        return Completion(
            value=answer, model=self.name, input_tokens=11, output_tokens=7, cost_usd=0.0001
        )


class QueuedModel:
    """Wraps `StubModel` and answers ONE schema from a queue, one answer per call; every other
    schema falls through to the real stub.

    Built for the review loop, which runs against `LocalTools` and the real toy CSV rather than
    canned facts: intake, profiler, feature_eng, and modeler need their ordinary stub answers, and
    only the reviewer's schema needs to be scripted pass-to-pass (block, then pass, say). An
    answer may be the `BaseModel` value directly, or a callable that receives the prompt's parsed
    facts (the same JSON block `StubModel` reads) and returns one -- so a later pass can react to
    ids the node only mints during the run, like an objection's `id`.

    An answer may also be an `Exception`, which is raised instead of returned, the same way
    `ScriptedModel` does it. The reviewer's block-retry needs a second call that fails while the
    first succeeded, which a single scripted answer per schema cannot express.
    """

    def __init__(
        self,
        schema: type[BaseModel],
        answers: list[BaseModel | Exception | Callable[[dict[str, Any]], BaseModel]],
        name: str = "queued",
    ) -> None:
        self.schema = schema
        self.answers = list(answers)
        self.name = name
        self._stub = StubModel(name=name)
        self.calls: list[tuple[str, str, type[BaseModel]]] = []

    def generate[M: BaseModel](self, *, system: str, user: str, schema: type[M]) -> Completion[M]:
        if schema is not self.schema:
            return self._stub.generate(system=system, user=user, schema=schema)
        self.calls.append((system, user, schema))
        if not self.answers:
            raise AssertionError(f"QueuedModel ran out of scripted answers for {schema.__name__}")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        value = answer(_payload(user)) if callable(answer) else answer
        return Completion(
            value=value, model=self.name, input_tokens=11, output_tokens=7, cost_usd=0.0001
        )


def runnable(name: str) -> Runnable:
    """A fixture by name, as the `Runnable` that `_run_state` and the naming functions take.

    Tests used to pass `load_fixture(name)` straight in. They cannot any more, and that is the
    point: `Runnable.from_fixture` is one of the two doors into a run, and the other one writes an
    empty answer key on purpose.
    """
    return Runnable.from_fixture(load_fixture(name))


def _toy_state(model_name: str = "haiku", reviewer_model_name: str | None = None) -> PipelineState:
    """The toy fixture's state, by name. Kept as its own function because several test modules
    call it, and each would otherwise re-derive the same `Runnable`."""
    return _run_state(
        Runnable.from_fixture(load_fixture("toy")),
        model_name=model_name,
        reviewer_model_name=reviewer_model_name,
    )


def manifest_from(
    *,
    n_rows: int,
    holdout: list[int],
    fold_valid: list[list[int]],
    strategy: str = "stratified",
    seed: int = 0,
    target: str = "y",
    holdout_fraction: float = 0.2,
) -> dict:
    """A manifest built in the test process, FOR TEST FIXTURES ONLY.

    This is not the implementation -- `split_manifest.ENCODER_SRC` is, and it is the only thing
    the profiler runs. This exists so a unit-test fixture can say which rows are in which fold in
    the same vocabulary the old explicit-list fixtures used, instead of a hand-typed
    200-character string. `tests/test_split_manifest.py` pins that it agrees with `ENCODER_SRC` on
    a real split, which is what keeps it a convenience rather than a second answer.
    """
    if len(fold_valid) > split_manifest.MAX_FOLDS:
        raise ValueError(f"n_folds={len(fold_valid)} exceeds {split_manifest.MAX_FOLDS}")
    slots: list[str | None] = [None] * n_rows
    for i in holdout:
        slots[int(i)] = split_manifest.HOLDOUT_CHAR
    for k, valid in enumerate(fold_valid):
        for i in valid:
            slots[int(i)] = str(k)
    missing = [i for i, c in enumerate(slots) if c is None]
    if missing:
        raise ValueError(f"{len(missing)} rows are in no partition (first: {missing[0]})")
    assignment = "".join(c for c in slots if c is not None)
    return {
        "version": split_manifest.SPLIT_MANIFEST_VERSION,
        "encoding": split_manifest.ENCODING,
        "fold_train": split_manifest.FOLD_TRAIN_RULE,
        "strategy": strategy,
        "seed": seed,
        "target": target,
        "n_rows": int(n_rows),
        "n_folds": len(fold_valid),
        "holdout_fraction": holdout_fraction,
        "assignment": assignment,
        "counts": {
            "train": sum(1 for c in assignment if c != split_manifest.HOLDOUT_CHAR),
            "holdout": sum(1 for c in assignment if c == split_manifest.HOLDOUT_CHAR),
            "folds": [len(valid) for valid in fold_valid],
        },
        "assignment_sha256": hashlib.sha256(assignment.encode()).hexdigest(),
    }


@pytest.fixture
def toy_dataset_artifact() -> ArtifactPayload:
    """What `read_artifact` hands intake for the toy dataset."""
    head = "\n".join(TOY_CSV.read_text().splitlines()[:6])
    return ArtifactPayload(
        meta=ArtifactMeta(
            id="dataset:toy",
            name="toy.csv",
            kind="table",
            n_bytes=len(head),
            extra={
                "columns": TOY_COLUMNS,
                "n_rows": 200,
                "dtypes": {c: "object" for c in TOY_COLUMNS},
                "n_unique": {c: 200 for c in TOY_COLUMNS} | {"churned": 2, "region": 4},
            },
        ),
        content=head,
        truncated=True,
    )
