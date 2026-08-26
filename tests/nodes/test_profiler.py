"""profiler writes `profile` and `split_artifact`.

The tests that matter here are the ones about what the model is allowed to put on the state: a
leakage candidate naming a column that does not exist cannot be scored against ground truth, and
counting it would fill the false-alarm column with the model's typos rather than its judgement.
"""

import json

import pytest
from conftest import FakeTools, ScriptedModel

from ds_agents.nodes.profiler import LeakageNomination, profiler
from ds_agents.state import LeakageCandidate, PipelineState, TaskSpec
from ds_agents.tools.protocol import RunResult, ToolError

pytestmark = pytest.mark.fast

STATS = {
    "n_rows": 200,
    "n_columns": 3,
    "columns": [
        {
            "name": "support_tickets_90d",
            "dtype": "int64",
            "missing_fraction": 0.0,
            "n_unique": 9,
            "sample_values": ["1", "2", "0"],
        },
        {
            "name": "account_status_code",
            "dtype": "object",
            "missing_fraction": 0.0,
            "n_unique": 4,
            "sample_values": ["CLOSED_R2", "ACTIVE_S1"],
        },
        {
            "name": "churned",
            "dtype": "int64",
            "missing_fraction": 0.0,
            "n_unique": 2,
            "sample_values": ["0", "1"],
        },
    ],
    "target_balance": {"0": 0.745, "1": 0.255},
    "target_association": {"support_tickets_90d": 0.0941, "account_status_code": 0.518},
    "target_association_errors": {},
}


def state() -> PipelineState:
    return PipelineState(
        dataset_id="toy",
        task_description="Predict churned, report roc_auc.",
        spec=TaskSpec(target="churned", task_type="binary", metric="roc_auc"),
    )


def leak(column: str = "account_status_code") -> LeakageCandidate:
    return LeakageCandidate(
        column=column,
        reason="status is assigned after the churn decision",
        evidence="normalized mutual information with the target is 0.518",
        suspicion="high",
    )


def tools_for(stats: dict | None = None, split_ok: bool = True) -> FakeTools:
    payload = json.dumps(STATS if stats is None else stats)
    split = RunResult(
        stdout="/artifacts/split_manifest.json",
        artifacts_written=["art-007-split-manifest"] if split_ok else [],
        exit_code=0 if split_ok else 1,
        stderr="" if split_ok else "ValueError: n_splits=5 needs more samples",
    )
    return FakeTools(run_results=[RunResult(stdout=payload), split])


def test_writes_profile_split_and_candidates():
    tools = tools_for()
    model = ScriptedModel({LeakageNomination: LeakageNomination(candidates=[leak()])})

    update = profiler(state(), tools=tools, model=model)

    assert set(update) == {"profile", "split_artifact", "node_trace"}
    profile = update["profile"]
    assert profile.n_rows == 200
    assert profile.target_balance == {"0": 0.745, "1": 0.255}
    assert [c.name for c in profile.columns] == [
        "support_tickets_90d",
        "account_status_code",
        "churned",
    ]
    assert [c.column for c in profile.leakage_candidates] == ["account_status_code"]
    assert update["split_artifact"] == "art-007-split-manifest"


def test_association_numbers_reach_the_prompt_but_the_target_row_does_not():
    """The model is asked to discriminate a leak from a strong legitimate feature, which it can
    only do from the numbers. The target's own row would be a perfect self-association and is
    noise in the prompt."""
    tools = tools_for()
    model = ScriptedModel({LeakageNomination: LeakageNomination(candidates=[])})

    profiler(state(), tools=tools, model=model)

    (_system, user, _schema) = model.calls[0]
    facts = json.loads(user)
    by_name = {c["name"]: c for c in facts["columns"]}
    assert "churned" not in by_name
    assert by_name["account_status_code"]["mutual_info_with_target"] == 0.518
    assert by_name["support_tickets_90d"]["mutual_info_with_target"] == 0.0941


def test_candidate_naming_an_unknown_column_is_dropped():
    tools = tools_for()
    model = ScriptedModel(
        {LeakageNomination: LeakageNomination(candidates=[leak("acct_status"), leak()])}
    )

    update = profiler(state(), tools=tools, model=model)

    assert [c.column for c in update["profile"].leakage_candidates] == ["account_status_code"]


def test_candidate_naming_the_target_is_dropped():
    """ "the target leaks the target" is true and useless, and it would score as a false alarm."""
    tools = tools_for()
    model = ScriptedModel({LeakageNomination: LeakageNomination(candidates=[leak("churned")])})

    update = profiler(state(), tools=tools, model=model)

    assert update["profile"].leakage_candidates == []


def test_failed_profiling_snippet_writes_an_error_and_no_profile():
    tools = FakeTools(run_results=[RunResult(exit_code=1, stderr="KeyError: 'churned'")])

    update = profiler(state(), tools=tools, model=ScriptedModel({}))

    assert "profile" not in update
    (error,) = update["errors"]
    assert "KeyError" in error.message
    assert len(update["node_trace"]) == 1


def test_failed_split_keeps_the_profile_and_records_the_error():
    """A missing split manifest is bad but not fatal: the profile is still worth having, and the
    run has to reach the reviewer for the failure to be visible in the results row."""
    tools = tools_for(split_ok=False)
    model = ScriptedModel({LeakageNomination: LeakageNomination(candidates=[])})

    update = profiler(state(), tools=tools, model=model)

    assert update["profile"].n_rows == 200
    assert "split_artifact" not in update
    assert "split manifest not written" in update["errors"][0].message


def test_no_spec_is_unrecoverable():
    bare = PipelineState(dataset_id="toy", task_description="predict something")

    update = profiler(bare, tools=FakeTools(), model=ScriptedModel({}))

    assert update["errors"][0].recoverable is False
    assert "profile" not in update


def test_association_errors_are_surfaced_not_swallowed():
    """A column whose association reads `null` is a column the model cannot flag, which looks
    exactly like a clean one."""
    stats = {**STATS, "target_association_errors": {"monthly_charges": "ValueError: NaN"}}
    tools = tools_for(stats)
    model = ScriptedModel({LeakageNomination: LeakageNomination(candidates=[])})

    update = profiler(state(), tools=tools, model=model)

    assert "monthly_charges" in update["errors"][0].message
    assert update["profile"].n_rows == 200


def test_a_raised_tool_error_on_the_stats_snippet_is_caught():
    """`RunResult(exit_code=1)` and a raised `ToolError` are different paths. A tool that refuses
    outright must not crash the node out of the graph."""
    tools = FakeTools(run_results=[ToolError("sandbox unavailable")])

    update = profiler(state(), tools=tools, model=ScriptedModel({}))

    assert "profile" not in update
    assert "sandbox unavailable" in update["errors"][0].message
    assert len(update["node_trace"]) == 1


def test_a_raised_tool_error_on_the_split_snippet_keeps_the_profile():
    tools = FakeTools(run_results=[RunResult(stdout=json.dumps(STATS)), ToolError("no sandbox")])
    model = ScriptedModel({LeakageNomination: LeakageNomination(candidates=[])})

    update = profiler(state(), tools=tools, model=model)

    assert update["profile"].n_rows == 200
    assert "split_artifact" not in update
    assert "no sandbox" in update["errors"][0].message


def test_an_unimplemented_split_strategy_is_refused_rather_than_mislabelled():
    """A random split written into a manifest that says "grouped" puts the same entity on both
    sides of the partition while the file claims otherwise -- which is the exact contamination
    the manifest exists to make falsifiable."""
    grouped = PipelineState(
        dataset_id="toy",
        task_description="Predict churned per customer.",
        spec=TaskSpec(
            target="churned",
            task_type="binary",
            metric="roc_auc",
            split_strategy="grouped",
            split_key="customer_id",
        ),
    )
    tools = FakeTools(run_results=[RunResult(stdout=json.dumps(STATS))])
    model = ScriptedModel({LeakageNomination: LeakageNomination(candidates=[])})

    update = profiler(grouped, tools=tools, model=model)

    assert "split_artifact" not in update
    (error,) = update["errors"]
    assert error.recoverable is False, "no trustworthy score is possible without the right split"
    assert "not implemented" in error.message
    assert tools.code_run == [tools.code_run[0]], "the split snippet must not have run"


def test_an_unexpected_client_error_still_leaves_an_error_and_an_event():
    """A real client raises its own timeouts and rate limits. An uncaught one would drop the run
    from the results entirely instead of recording it as the failure it was."""
    tools = tools_for()
    model = ScriptedModel({LeakageNomination: TimeoutError("read timed out")})

    update = profiler(state(), tools=tools, model=model)

    assert update["profile"].leakage_candidates == []
    assert any("read timed out" in e.message for e in update["errors"])
    assert len(update["node_trace"]) == 1
