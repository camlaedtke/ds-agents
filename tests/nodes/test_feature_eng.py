"""feature_eng writes `feature_code_artifact`, `feature_summary`, `final_features`,
`dropped_features`.

The tests that matter here are the ones about the forced-drop set: the target, the id-column
class, and any open leakage-shaped objection must survive no matter what the model says, and
`final_features` must stay in SOURCE-column vocabulary or `results_row()`'s
`leakage_remediated` computation silently scores every run as remediated.
"""

import json

import pytest
from conftest import FakeTools, ScriptedModel

from ds_agents import split_manifest
from ds_agents.nodes.feature_eng import (
    FeatureDrop,
    FeaturePlan,
    KeptColumn,
    feature_eng,
)
from ds_agents.state import (
    ColumnProfile,
    LeakageCandidate,
    Objection,
    PipelineState,
    ProfileReport,
    ReviewPass,
    RunConfig,
    TaskSpec,
)
from ds_agents.tools.protocol import ArtifactMeta, ArtifactPayload, RunResult, ToolError

pytestmark = pytest.mark.fast

# This node only ever reads `train` out of the split (medians, one-hot levels, everything else is
# fitted on train rows), so what the folds look like is irrelevant to every test in this file --
# only that the 160 train rows are accounted for. Under this encoding a row is only "train" if it
# validates in SOME fold (there is no character for "train, never validated"), so a single-fold
# stand-in has to put every train row in that one fold's valid set. That fold's own derived train
# (the complement) is then empty, which is the same degenerate shape the old explicit-list fixture
# had -- an honest fixture for a node that never looks past `SPLIT["train"]`, not a claim about
# what a real k-fold split looks like.
SPLIT_MANIFEST = split_manifest.manifest_from(
    n_rows=200,
    holdout=list(range(160, 200)),
    fold_valid=[list(range(160))],
    strategy="stratified",
    seed=20260822,
    target="churned",
)

SPLIT_ARTIFACT_ID = "art-007-split-manifest"

# Mirrors the JSON actually produced by FEATURE_SNIPPET when run against the toy fixture with
# drop=["customer_id"] and the split above -- verified in a throwaway subprocess run, not just
# eyeballed. See the task report for the raw output.
SNIPPET_OUT = {
    "n_rows": 200,
    "n_train_rows": 160,
    "dropped": ["customer_id"],
    "final_features": [
        "tenure_months",
        "monthly_charges",
        "support_tickets_90d",
        "region",
        "plan_tier",
        "account_status_code",
    ],
    "matrix_columns": [
        "tenure_months",
        "monthly_charges",
        "support_tickets_90d",
        "region=east",
        "region=north",
        "region=south",
        "region=west",
        "plan_tier=basic",
        "plan_tier=plus",
        "plan_tier=premium",
        "account_status_code=ACTIVE_S1",
        "account_status_code=CLOSED_R2",
    ],
    "numeric": ["tenure_months", "monthly_charges", "support_tickets_90d"],
    "categorical": ["region", "plan_tier", "account_status_code"],
    "skipped_high_cardinality": [],
    "medians": {
        "tenure_months": 34.5,
        "monthly_charges": 69.26,
        "support_tickets_90d": 2.0,
    },
    "levels": {
        "region": ["east", "north", "south", "west"],
        "plan_tier": ["basic", "plus", "premium"],
        "account_status_code": ["ACTIVE_S1", "CLOSED_R2"],
    },
    "n_matrix_columns": 12,
    "n_nan_in_matrix": 0,
    "columns_match_order": True,
    "code_path": "/artifacts/feature_transform.py",
}


def state(**overrides) -> PipelineState:
    defaults = dict(
        dataset_id="toy",
        task_description="Predict churned, report roc_auc.",
        spec=TaskSpec(target="churned", task_type="binary", metric="roc_auc"),
        profile=ProfileReport(
            n_rows=200,
            n_columns=8,
            columns=[
                ColumnProfile(
                    name="customer_id", dtype="object", missing_fraction=0.0, n_unique=200
                ),
                ColumnProfile(
                    name="tenure_months", dtype="int64", missing_fraction=0.0, n_unique=64
                ),
                ColumnProfile(
                    name="monthly_charges", dtype="float64", missing_fraction=0.08, n_unique=181
                ),
                ColumnProfile(
                    name="support_tickets_90d", dtype="int64", missing_fraction=0.0, n_unique=8
                ),
                ColumnProfile(name="region", dtype="object", missing_fraction=0.0, n_unique=4),
                ColumnProfile(name="plan_tier", dtype="object", missing_fraction=0.0, n_unique=3),
                ColumnProfile(
                    name="account_status_code", dtype="object", missing_fraction=0.0, n_unique=2
                ),
                ColumnProfile(name="churned", dtype="int64", missing_fraction=0.0, n_unique=2),
            ],
        ),
        split_artifact=SPLIT_ARTIFACT_ID,
    )
    defaults.update(overrides)
    return PipelineState(**defaults)


def split_payload() -> ArtifactPayload:
    content = json.dumps(SPLIT_MANIFEST)
    return ArtifactPayload(
        meta=ArtifactMeta(id=SPLIT_ARTIFACT_ID, name="split_manifest.json", kind="json"),
        content=content,
    )


def tools_for(
    snippet_out: dict | None = None,
    run_result: RunResult | Exception | None = None,
) -> FakeTools:
    if run_result is None:
        run_result = RunResult(
            stdout=json.dumps(SNIPPET_OUT if snippet_out is None else snippet_out),
            artifacts_written=["art-011-feature-transform"],
        )
    return FakeTools(
        run_results=[run_result],
        artifacts={SPLIT_ARTIFACT_ID: split_payload()},
    )


def empty_plan_model() -> ScriptedModel:
    return ScriptedModel({FeaturePlan: FeaturePlan(drops=[], kept_despite_flag=[])})


def test_writes_the_four_feature_fields_and_nothing_else():
    tools = tools_for()
    model = empty_plan_model()

    update = feature_eng(state(), tools=tools, model=model)

    assert set(update) == {
        "feature_code_artifact",
        "feature_summary",
        "final_features",
        "dropped_features",
        "node_trace",
    }
    assert update["feature_code_artifact"] == "art-011-feature-transform"


def test_final_features_are_source_names_not_dummy_names():
    """This is what protects `leakage_remediated`: `results_row()` intersects `final_features`
    with `planted_leakage_columns`, which are source names. A one-hot name like `region=north`
    here would empty that intersection and score every run as remediated even with the leak
    fully present."""
    tools = tools_for()
    model = empty_plan_model()

    update = feature_eng(state(), tools=tools, model=model)

    assert "region" in update["final_features"]
    assert not any("=" in f for f in update["final_features"])


def test_the_id_column_is_always_dropped_even_if_the_model_says_nothing():
    tools = tools_for()
    model = ScriptedModel({FeaturePlan: FeaturePlan()})

    update = feature_eng(state(), tools=tools, model=model)

    assert "customer_id" in update["dropped_features"]
    assert "customer_id" in tools.code_run[0]


def test_an_open_leakage_objection_forces_the_drop():
    objection = Objection(
        category="leakage",
        subcategory="planted_status_code",
        target_node="feature_eng",
        columns=["account_status_code"],
        evidence="91% agreement with the target",
        severity="high",
        raised_at_iteration=0,
    )
    tools = tools_for()
    model = empty_plan_model()

    feature_eng(state(objections=[objection]), tools=tools, model=model)

    assert "account_status_code" in tools.code_run[0]


def _planted_column_objection() -> Objection:
    return Objection(
        category="leakage",
        subcategory="planted_status_code",
        target_node="feature_eng",
        columns=["account_status_code"],
        evidence="91% agreement with the target",
        severity="high",
        raised_at_iteration=0,
    )


def _pass_disposing(objection: Objection, disposition: str, iteration: int = 0) -> ReviewPass:
    return ReviewPass(
        iteration=iteration,
        claim="pass",
        routed_to="reporter",
        dispositions={objection.id: disposition},
    )


def test_a_resolved_objection_still_forces_its_drop():
    """DELIBERATE REVERSAL of `test_a_resolved_objection_does_not_force_a_drop`, which pinned the
    opposite behaviour from the first day this node handled objections until 2026-08-28. Recorded
    here rather than in a commit message because the old assertion was right about the mechanism
    and wrong about the intent, and that distinction is the whole content of the change.

    The old test guarded that `open_objections()` was read and not the raw `objections` list. That
    is still guarded -- `binding_objections` folds the same dispositions in the same order. What
    changed is WHICH disposition releases a column. On this pipeline the fix for a column objection
    IS the drop, so `resolved` cannot also mean "put it back": `_forced_drops` recomputes on every
    invocation, so a resolved objection plus a later return to feature_eng for some unrelated
    reason silently resurrected the leaked column. Harmless across a single pass, which is why the
    old test read as correct for so long; reachable across two, which `objection_routing=
    "by_category"` produces routinely and `objection_closure="on"` produces more.

    `withdrawn` is now the only release, and `test_a_withdrawn_objection_releases_its_drop` below
    is what stops this from meaning "objections are immortal". See DECISIONS.md 2026-08-28 (fourth
    entry) and evals/results/LOG.md, "A latent bug found while diagnosing this cell".
    """
    objection = _planted_column_objection()
    tools = tools_for()

    feature_eng(
        state(objections=[objection], review_passes=[_pass_disposing(objection, "resolved")]),
        tools=tools,
        model=empty_plan_model(),
    )

    assert "account_status_code" in tools.code_run[0]


def test_a_withdrawn_objection_releases_its_drop():
    """The escape hatch, and the reason the reversal above is not "objections are immortal".

    `withdrawn` means the reviewer no longer thinks it was a problem, which is the opposite claim
    to `resolved`. It is also the only route back for a false positive: `by_category` dropped
    `prior_claims_12m`, a legitimate strong feature, in 2 of 10 runs, and without this a reviewer
    mistake would be permanent for the rest of the run.
    """
    objection = _planted_column_objection()
    tools = tools_for()

    feature_eng(
        state(objections=[objection], review_passes=[_pass_disposing(objection, "withdrawn")]),
        tools=tools,
        model=empty_plan_model(),
    )

    assert "DROP = ['churned', 'customer_id']" in tools.code_run[0]


def test_a_not_reviewed_objection_still_forces_its_drop():
    """Silence is not release, matching REVIEWER_SYSTEM's rule that leaving an objection out of
    `dispositions` means "I did not look at it again"."""
    objection = _planted_column_objection()
    tools = tools_for()

    feature_eng(
        state(objections=[objection], review_passes=[_pass_disposing(objection, "not_reviewed")]),
        tools=tools,
        model=empty_plan_model(),
    )

    assert "account_status_code" in tools.code_run[0]


def test_a_resolved_objection_stays_dropped_across_a_second_return():
    """The bug's actual shape, as the regression guard: the objection is raised and acted on in
    pass 0, resolved in pass 1, and the run returns to this node again for some other reason. The
    column must not come back."""
    objection = _planted_column_objection()
    tools = tools_for()

    feature_eng(
        state(
            objections=[objection],
            review_passes=[
                _pass_disposing(objection, "still_open", iteration=0),
                _pass_disposing(objection, "resolved", iteration=1),
            ],
        ),
        tools=tools,
        model=empty_plan_model(),
    )

    assert "account_status_code" in tools.code_run[0]


def test_a_resolved_objection_comes_back_under_the_unsticky_arm():
    """The control arm reproduces the defect it claims to reproduce, asserted against the real
    snippet rather than against the state method.

    Without this the sticky-drop cell's control is an assumption: `forced_drop_release` could be
    recorded on every row and change nothing that reaches the sandbox, and the arm would read as a
    null result for the wrong reason. The exact strings are the ones from the live diagnosis on
    2026-08-28 -- with the objection binding the snippet reads `DROP = ['account_status_code',
    'churned', 'customer_id']`, and with it released, `DROP = ['churned', 'customer_id']`.
    """
    objection = _planted_column_objection()
    tools = tools_for()

    feature_eng(
        state(
            config=RunConfig(forced_drop_release="resolved_or_withdrawn"),
            objections=[objection],
            review_passes=[_pass_disposing(objection, "resolved")],
        ),
        tools=tools,
        model=empty_plan_model(),
    )

    assert "DROP = ['churned', 'customer_id']" in tools.code_run[0]
    assert "account_status_code" not in tools.code_run[0]


def test_the_second_return_is_where_the_two_arms_diverge():
    """The mechanism the cell is pre-registered to detect, in one test.

    A single pass cannot show the defect -- `_forced_drops` recomputes on entry, so resolution and
    resurrection have to be separated by a return to this node. That is why the pre-registration
    predicts the effect is confined to runs whose `route_sequence` contains two `feature_eng`
    entries, and why a rise concentrated anywhere else would falsify the mechanism even if the
    headline number replicated.
    """
    objection = _planted_column_objection()
    passes = [
        _pass_disposing(objection, "still_open", iteration=0),
        _pass_disposing(objection, "resolved", iteration=1),
    ]

    sticky = tools_for()
    feature_eng(
        state(objections=[objection], review_passes=passes),
        tools=sticky,
        model=empty_plan_model(),
    )
    assert "account_status_code" in sticky.code_run[0]

    unsticky = tools_for()
    feature_eng(
        state(
            config=RunConfig(forced_drop_release="resolved_or_withdrawn"),
            objections=[objection],
            review_passes=passes,
        ),
        tools=unsticky,
        model=empty_plan_model(),
    )
    assert "account_status_code" not in unsticky.code_run[0]


def test_the_drop_justification_is_identical_under_both_release_rules():
    """The confound guard. The 2026-08-28 fix changed TWO things: the predicate, and the
    justification string that goes into this node's `already_dropped` facts -- so it reaches the
    MODEL'S PROMPT, and `dropped_features` on the results row. Reverting the wording under the
    control arm would give the condition a second application site; leaving it arm-dependent would
    mean the two arms differ in prompt text as well as in the release rule, and the cell could not
    attribute its effect to either.

    So the current wording stays in both arms. It is still true in both: under
    `resolved_or_withdrawn` a forced drop can only come from an objection that is neither resolved
    nor withdrawn, so "not withdrawn" is correct there, merely weaker than the truth.
    """
    objection = _planted_column_objection()
    prompts = []
    for release in ("withdrawn_only", "resolved_or_withdrawn"):
        model = empty_plan_model()
        feature_eng(
            state(
                config=RunConfig(forced_drop_release=release),
                objections=[objection],
                review_passes=[_pass_disposing(objection, "still_open")],
            ),
            tools=tools_for(),
            model=model,
        )
        prompts.append(model.calls[0][1])

    assert ", not withdrawn: " in prompts[0]
    assert prompts[0] == prompts[1]


def _misaddressed_importance_objection() -> Objection:
    """The shape the reviewer actually produced live on 2026-08-28: the right column, in a
    column-scoped category, addressed to a node with no column lever."""
    return Objection(
        category="implausible_importance",
        subcategory="importance_dominance",
        target_node="modeler",
        columns=["account_status_code"],
        evidence="permutation importance 0.367, an order of magnitude above any other column",
        severity="high",
        raised_at_iteration=0,
    )


def test_a_modeler_addressed_importance_objection_is_ignored_under_the_default_routing():
    """The bug, as a test. `_forced_drops` reads `open_objections("feature_eng")`, so an objection
    naming the right column but addressed to `modeler` reaches nothing that can act on it -- and
    the modeler cannot either, because every candidate is fit on the transform this node froze."""
    tools = tools_for()

    feature_eng(
        state(objections=[_misaddressed_importance_objection()]),
        tools=tools,
        model=empty_plan_model(),
    )

    assert "DROP = ['churned', 'customer_id']" in tools.code_run[0]


def test_a_modeler_addressed_importance_objection_forces_the_drop_under_by_category():
    """The fix. Same objection, same node, same model: only `config.objection_routing` differs.

    This node did not change to make it happen -- `open_objections` resolves who acts, so there is
    exactly one place the condition is applied and this file inherits it.
    """
    tools = tools_for()

    feature_eng(
        state(
            config=RunConfig(objection_routing="by_category"),
            objections=[_misaddressed_importance_objection()],
        ),
        tools=tools,
        model=empty_plan_model(),
    )

    assert "account_status_code" in tools.code_run[0]


def test_the_model_can_add_a_drop_but_never_remove_a_forced_one():
    plan = FeaturePlan(
        drops=[
            FeatureDrop(
                column="support_tickets_90d",
                reason="redundant",
                justification="near-zero permutation importance in a prior run",
            )
        ],
        kept_despite_flag=[
            # An attempt to "un-drop" a forced column. Must have no effect.
            KeptColumn(column="customer_id", why_not_leakage="it's fine, trust me")
        ],
    )
    tools = tools_for()
    model = ScriptedModel({FeaturePlan: plan})

    update = feature_eng(state(), tools=tools, model=model)

    code = tools.code_run[0]
    assert "customer_id" in code
    assert "support_tickets_90d" in code
    # The forced drop must not appear in the assembled summary as a deliberate keep.
    assert "customer_id -- it's fine, trust me" not in update["feature_summary"]


def test_a_drop_naming_an_unknown_column_is_ignored():
    plan = FeaturePlan(
        drops=[FeatureDrop(column="not_a_real_column", reason="other", justification="typo test")]
    )
    tools = tools_for()
    model = ScriptedModel({FeaturePlan: plan})

    feature_eng(state(), tools=tools, model=model)

    assert "not_a_real_column" not in tools.code_run[0]


def test_the_profile_and_the_leakage_candidates_reach_the_prompt():
    tools = tools_for()
    model = empty_plan_model()

    feature_eng(
        state(
            profile=ProfileReport(
                n_rows=200,
                n_columns=8,
                columns=state().profile.columns,
                leakage_candidates=[
                    LeakageCandidate(
                        column="account_status_code",
                        reason="status is assigned after the churn decision",
                        evidence="normalized mutual information with the target is 0.518",
                        suspicion="high",
                    )
                ],
            )
        ),
        tools=tools,
        model=model,
    )

    (_system, user, _schema) = model.calls[0]
    facts = json.loads(user)
    assert facts["target"] == "churned"
    assert "churned" not in {c["name"] for c in facts["columns"]}
    by_name = {c["column"]: c for c in facts["profiler_leakage_candidates"]}
    assert by_name["account_status_code"]["evidence"] == (
        "normalized mutual information with the target is 0.518"
    )
    assert {d["column"] for d in facts["already_dropped"]} >= {"churned", "customer_id"}


def test_no_split_artifact_is_unrecoverable():
    tools = FakeTools()
    model = empty_plan_model()

    update = feature_eng(state(split_artifact=None), tools=tools, model=model)

    assert update["errors"][0].recoverable is False
    assert "contaminat" in update["errors"][0].message
    assert len(update["node_trace"]) == 1
    assert tools.code_run == []


def test_no_profile_is_unrecoverable():
    tools = FakeTools()
    model = empty_plan_model()

    update = feature_eng(state(profile=None), tools=tools, model=model)

    assert update["errors"][0].recoverable is False
    assert "profile" not in update
    assert tools.code_run == []


def test_a_failed_snippet_writes_an_error_and_no_feature_fields():
    tools = tools_for(run_result=RunResult(exit_code=1, stderr="KeyError: 'churned'"))
    model = empty_plan_model()

    update = feature_eng(state(), tools=tools, model=model)

    assert "feature_code_artifact" not in update
    assert "final_features" not in update
    (error,) = update["errors"]
    assert "KeyError" in error.message
    assert len(update["node_trace"]) == 1


def test_a_raised_tool_error_is_caught_not_crashed():
    tools = tools_for(run_result=ToolError("sandbox unavailable"))
    model = empty_plan_model()

    update = feature_eng(state(), tools=tools, model=model)

    assert "feature_code_artifact" not in update
    assert "sandbox unavailable" in update["errors"][0].message
    assert len(update["node_trace"]) == 1


def test_nan_in_the_matrix_is_an_error_not_a_silent_pass():
    """The matrix went to the modeler with holes in it that the modeler would die on -- not a
    close call, so nothing gets written."""
    tools = tools_for(snippet_out={**SNIPPET_OUT, "n_nan_in_matrix": 3})
    model = empty_plan_model()

    update = feature_eng(state(), tools=tools, model=model)

    assert set(update) == {"node_trace", "errors"}
    assert "3" in update["errors"][0].message
    assert "NaN" in update["errors"][0].message


def test_a_skipped_high_cardinality_column_is_reported_and_counted_as_dropped():
    snippet_out = {
        **SNIPPET_OUT,
        "dropped": ["customer_id", "some_free_text_column"],
        "skipped_high_cardinality": ["some_free_text_column"],
    }
    tools = tools_for(snippet_out=snippet_out)
    model = empty_plan_model()

    update = feature_eng(state(), tools=tools, model=model)

    assert "some_free_text_column" in update["dropped_features"]
    assert "feature_code_artifact" in update
    assert any("some_free_text_column" in e.message for e in update["errors"])


def test_a_model_client_error_leaves_the_forced_drops_and_an_error():
    tools = tools_for()
    model = ScriptedModel({FeaturePlan: TimeoutError("read timed out")})

    update = feature_eng(state(), tools=tools, model=model)

    assert "customer_id" in tools.code_run[0]
    assert "customer_id" in update["dropped_features"]
    assert any("read timed out" in e.message for e in update["errors"])
    assert len(update["node_trace"]) == 1


def test_the_dtype_and_cardinality_decision_reads_train_rows_not_the_full_frame():
    """Finding 6: the numeric/categorical/skipped classification used to read `df[name]` and
    `nunique()` over the FULL frame, while medians and one-hot levels were correctly train-fitted
    -- so holdout rows alone could push a column's cardinality over MAX_ONE_HOT_LEVELS and get it
    silently skipped from a transform that is otherwise train-fitted, which is the `contamination`
    category the reviewer exists to catch. Regression guard on the emitted snippet source: the
    classification loop must read `train`, never `df`."""
    tools = tools_for()
    model = empty_plan_model()

    feature_eng(state(), tools=tools, model=model)

    code = tools.code_run[0]
    assert "col = train[name]" in code
    assert "col = df[name]" not in code


def test_id_distinctness_threshold_is_a_named_constant_used_by_forced_drops(monkeypatch):
    """Finding 9: the 0.98 id-column cutoff was a bare literal inside `_forced_drops`. Lifted to
    module-level `ID_DISTINCTNESS_THRESHOLD` alongside `MAX_ONE_HOT_LEVELS` and
    `FEATURE_TIMEOUT_S`. Guards that `_forced_drops` actually reads the constant -- not a second
    hard-coded 0.98 the constant could quietly drift away from -- by proving a monkeypatched
    threshold changes what gets force-dropped."""
    import importlib

    # Not `import ds_agents.nodes.feature_eng as ...`: `nodes/__init__.py` does
    # `from ds_agents.nodes.feature_eng import feature_eng`, which rebinds the `feature_eng`
    # attribute on the `ds_agents.nodes` package from the submodule to the function. A dotted
    # `import ... as` resolves through that shadowed attribute; `importlib.import_module` goes
    # through `sys.modules` instead and gets the real submodule.
    feature_eng_module = importlib.import_module("ds_agents.nodes.feature_eng")

    assert feature_eng_module.ID_DISTINCTNESS_THRESHOLD == 0.98

    tools = tools_for()
    feature_eng(state(), tools=tools, model=empty_plan_model())
    assert "DROP = ['churned', 'customer_id']" in tools.code_run[0]

    # tenure_months is 64/200 = 32% distinct -- nowhere near an id at the real 98% threshold, but
    # crosses a threshold lowered to 20% (support_tickets_90d is only 4% distinct and stays out).
    monkeypatch.setattr(feature_eng_module, "ID_DISTINCTNESS_THRESHOLD", 0.20)
    tools_lowered = tools_for()
    feature_eng(state(), tools=tools_lowered, model=empty_plan_model())
    assert "DROP = ['churned', 'customer_id', 'tenure_months']" in tools_lowered.code_run[0]


def test_a_truncated_split_manifest_is_unrecoverable():
    """Finding 12: a truncated split manifest would silently fit medians and one-hot levels on
    fewer rows than the pinned split actually names, with nothing downstream able to tell. The
    modeler already refuses a truncated feature code artifact for the same reason; this is the
    symmetric guard on the split manifest here."""
    truncated_payload = ArtifactPayload(
        meta=ArtifactMeta(id=SPLIT_ARTIFACT_ID, name="split_manifest.json", kind="json"),
        content=json.dumps(SPLIT_MANIFEST),
        truncated=True,
    )
    tools = FakeTools(artifacts={SPLIT_ARTIFACT_ID: truncated_payload})
    model = empty_plan_model()

    update = feature_eng(state(), tools=tools, model=model)

    assert update["errors"][0].recoverable is False
    assert "truncated" in update["errors"][0].message
    assert tools.code_run == []


def test_an_unexpected_artifact_count_is_an_error_not_a_blind_index():
    """Finding 12: `feature_run.artifacts_written[0]` used to index blind. Zero written artifacts
    means nothing for the modeler to exec; more than one means the snippet wrote something
    unexpected. Either way, `[0]` would silently accept a schema this node was never built to
    hand off."""
    zero_written = tools_for(run_result=RunResult(stdout=json.dumps(SNIPPET_OUT)))
    update = feature_eng(state(), tools=zero_written, model=empty_plan_model())
    assert "feature_code_artifact" not in update
    assert "expected exactly one" in update["errors"][0].message
    assert "0" in update["errors"][0].message

    two_written = tools_for(
        run_result=RunResult(
            stdout=json.dumps(SNIPPET_OUT),
            artifacts_written=["art-011-feature-transform", "art-012-unexpected"],
        )
    )
    update = feature_eng(state(), tools=two_written, model=empty_plan_model())
    assert "feature_code_artifact" not in update
    assert "expected exactly one" in update["errors"][0].message
    assert "2" in update["errors"][0].message


def test_the_summary_is_assembled_from_the_snippet_not_the_model():
    """`FeaturePlan` has no free `summary` field, so this is mostly a structural guarantee -- but
    the numbers in the assembled string must trace to the snippet's JSON, not to anything the
    model could have said."""
    tools = tools_for()
    model = empty_plan_model()

    update = feature_eng(state(), tools=tools, model=model)

    summary = update["feature_summary"]
    assert "6 source columns -> 12 matrix columns" in summary
    assert "160 train rows" in summary
    assert SPLIT_ARTIFACT_ID in summary
    assert "customer_id" in summary
