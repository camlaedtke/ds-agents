"""intake writes `spec` and nothing else. The failure paths matter as much as the happy one:
a run that dies here still has to leave an error row, because a dataset that disappears from the
results biases every table upward."""

import pytest
from conftest import TOY_COLUMNS, FakeTools, ScriptedModel

from ds_agents.nodes.intake import IntakeDecision, intake
from ds_agents.state import PipelineState

pytestmark = pytest.mark.fast


def state() -> PipelineState:
    return PipelineState(dataset_id="toy", task_description="Predict churned, report roc_auc.")


def decision(**overrides) -> IntakeDecision:
    kwargs = {
        "target": "churned",
        "task_type": "binary",
        "metric": "roc_auc",
        "rationale": "binary target named in the description",
    }
    return IntakeDecision(**{**kwargs, **overrides})


def test_writes_spec_and_a_trace_event(toy_dataset_artifact):
    tools = FakeTools(artifacts={"dataset:toy": toy_dataset_artifact})
    model = ScriptedModel({IntakeDecision: decision()})

    update = intake(state(), tools=tools, model=model)

    assert set(update) == {"spec", "node_trace"}, "intake must not write anyone else's fields"
    assert update["spec"].target == "churned"
    assert update["spec"].metric == "roc_auc"
    assert update["spec"].greater_is_better is True
    (event,) = update["node_trace"]
    assert event.node == "intake"
    assert (event.input_tokens, event.output_tokens) == (11, 7)
    assert event.model == "fake"


def test_the_schema_reaches_the_prompt(toy_dataset_artifact):
    """The model cannot name a target it was never shown. If this regresses, intake starts
    guessing from the task description alone and the failure looks like a bad model."""
    tools = FakeTools(artifacts={"dataset:toy": toy_dataset_artifact})
    model = ScriptedModel({IntakeDecision: decision()})

    intake(state(), tools=tools, model=model)

    (_system, user, _schema) = model.calls[0]
    for column in TOY_COLUMNS:
        assert column in user


def test_target_that_is_not_a_column_is_an_error_not_a_spec(toy_dataset_artifact):
    """A hallucinated column caught here is one error row. Carried forward it is a profiler
    crash blamed on the wrong node."""
    tools = FakeTools(artifacts={"dataset:toy": toy_dataset_artifact})
    model = ScriptedModel({IntakeDecision: decision(target="churn_flag")})

    update = intake(state(), tools=tools, model=model)

    assert "spec" not in update
    (error,) = update["errors"]
    assert error.node == "intake"
    assert error.recoverable is False
    assert "churn_flag" in error.message
    assert len(update["node_trace"]) == 1, "the event is appended on the error path too"


def test_missing_dataset_artifact_is_unrecoverable(toy_dataset_artifact):
    update = intake(state(), tools=FakeTools(artifacts={}), model=ScriptedModel({}))

    assert "spec" not in update
    (error,) = update["errors"]
    assert error.recoverable is False


def test_split_key_must_exist_when_the_strategy_needs_one(toy_dataset_artifact):
    tools = FakeTools(artifacts={"dataset:toy": toy_dataset_artifact})
    model = ScriptedModel(
        {IntakeDecision: decision(split_strategy="temporal", split_key="signup_date")}
    )

    update = intake(state(), tools=tools, model=model)

    assert "spec" not in update
    assert "signup_date" in update["errors"][0].message


def test_an_unexpected_client_error_is_still_an_error_row(toy_dataset_artifact):
    """Timeouts and rate limits come from the client, not from the schema. Uncaught, the run
    disappears from the results instead of appearing as a failure."""
    tools = FakeTools(artifacts={"dataset:toy": toy_dataset_artifact})
    model = ScriptedModel({IntakeDecision: TimeoutError("read timed out")})

    update = intake(state(), tools=tools, model=model)

    assert "spec" not in update
    assert "read timed out" in update["errors"][0].message
    assert len(update["node_trace"]) == 1
