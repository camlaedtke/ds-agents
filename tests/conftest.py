"""Test doubles for the tool surface and the model client.

Node tests use these rather than `LocalTools`, so a node test fails for a reason inside the node.
A node test that shells out to a real subprocess is testing pandas.
"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from ds_agents.state import ArtifactId
from ds_agents.tools.llm import Completion
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
