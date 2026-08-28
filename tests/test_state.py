"""The contract every node depends on. If these fail, nothing downstream means anything."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ds_agents.state import (
    ColumnProfile,
    LeakageCandidate,
    ModelResult,
    NodeEvent,
    Objection,
    PipelineError,
    PipelineState,
    ProfileReport,
    ReviewPass,
    RunConfig,
    TaskSpec,
    utc_now,
)

pytestmark = pytest.mark.fast


def leak_objection(**overrides) -> Objection:
    kwargs = {
        "category": "leakage",
        "subcategory": "post_hoc_status_code",
        "target_node": "feature_eng",
        "columns": ["account_status_code"],
        "evidence": "agrees with target on 91% of rows",
        "severity": "high",
        "raised_at_iteration": 0,
    }
    return Objection(**{**kwargs, **overrides})


def populated_state() -> PipelineState:
    """Every optional field set, so round-tripping actually exercises the schema."""
    objection = leak_objection()
    return PipelineState(
        dataset_id="toy",
        task_description="predict churn",
        spec=TaskSpec(target="churned", task_type="binary", metric="roc_auc"),
        profile=ProfileReport(
            n_rows=200,
            n_columns=8,
            columns=[
                ColumnProfile(
                    name="account_status_code",
                    dtype="object",
                    missing_fraction=0.0,
                    n_unique=4,
                    sample_values=["CLOSED_R2", "ACTIVE_S1"],
                ),
                ColumnProfile(
                    name="monthly_charges",
                    dtype="float64",
                    missing_fraction=0.03,
                    n_unique=187,
                    sample_values=["77.72", "35.30"],
                ),
            ],
            leakage_candidates=[
                LeakageCandidate(
                    column="account_status_code",
                    reason="assigned after the churn decision, not available at predict time",
                    evidence="agrees with target on 91% of rows",
                    suspicion="high",
                )
            ],
            target_balance={"0": 0.745, "1": 0.255},
        ),
        split_artifact="art-split-1",
        feature_code_artifact="art-features-1",
        feature_summary="one-hot region and plan_tier",
        final_features=["support_tickets_90d", "tenure_months"],
        dropped_features=["account_status_code"],
        candidates=[ModelResult(name="logreg", cv_scores=[0.8, 0.82, 0.79])],
        chosen_model=ModelResult(name="logreg", cv_scores=[0.8], claimed_holdout_score=0.81),
        importance_artifact="art-importance-1",
        top_importances=[("support_tickets_90d", 0.62)],
        review_iterations=1,
        objections=[objection],
        review_passes=[
            ReviewPass(
                iteration=1,
                claim="block",
                routed_to="feature_eng",
                dispositions={objection.id: "still_open"},
                new_objection_ids=[objection.id],
            )
        ],
        reviewer_claim="block",
        review_verdict="block",
        report_artifact="art-report-1",
        planted_leakage_columns=["account_status_code"],
        verified_holdout_score=0.74,
        baseline_score=0.70,
        errors=[
            PipelineError(
                node="profiler",
                message="monthly_charges has missing values; imputed with median",
            )
        ],
        node_trace=[NodeEvent(node="intake", started=datetime(2026, 8, 22, tzinfo=UTC))],
        ended_at=datetime(2026, 8, 22, 0, 5, tzinfo=UTC),
    )


class TestRoundTrip:
    def test_dumped_state_can_be_read_back(self):
        state = populated_state()
        assert PipelineState.from_dump(json.loads(state.model_dump_json())) == state

    def test_computed_fields_survive_into_the_dump(self):
        """The harness reads these out of the JSONL; a plain @property would vanish silently."""
        dumped = json.loads(populated_state().model_dump_json())
        for key in ("total_cost_usd", "loop_exhausted", "wall_seconds", "score_ratio"):
            assert key in dumped
        assert dumped["spec"]["greater_is_better"] is True
        assert dumped["node_trace"][0]["wall_seconds"] is None

    def test_minimal_state_needs_only_intake_inputs(self):
        state = PipelineState(dataset_id="toy", task_description="predict churn")
        assert state.review_verdict == "pending"
        assert state.objections == []
        assert state.config.loop_cap == 3


class TestContractEnforcement:
    def test_unknown_field_is_rejected(self):
        """A node inventing a field is a contract violation, not a convenience."""
        with pytest.raises(ValidationError):
            PipelineState(dataset_id="toy", task_description="x", sneaky_field="nope")

    def test_objection_category_is_closed(self):
        with pytest.raises(ValidationError):
            leak_objection(category="vibes")

    def test_reviewer_cannot_route_to_a_non_routable_node(self):
        with pytest.raises(ValidationError):
            leak_objection(target_node="reporter")

    def test_leakage_objection_without_columns_is_rejected(self):
        """leakage_caught is a set comparison. An objection naming no column cannot be scored."""
        with pytest.raises(ValidationError, match="requires at least one column"):
            leak_objection(columns=[])

    def test_non_column_scoped_objection_may_omit_columns(self):
        assert (
            Objection(
                category="overfit",
                subcategory="cv_holdout_gap",
                target_node="modeler",
                evidence="cv 0.99 vs holdout 0.62",
                severity="high",
                raised_at_iteration=1,
            ).columns
            == []
        )

    def test_run_config_is_frozen(self):
        """A node must not be able to raise its own loop cap mid-run."""
        state = PipelineState(dataset_id="toy", task_description="x")
        with pytest.raises(ValidationError):
            state.config.loop_cap = 99

    def test_keyed_split_strategies_require_a_key(self):
        with pytest.raises(ValidationError, match="requires split_key"):
            TaskSpec(target="y", task_type="binary", metric="roc_auc", split_strategy="temporal")

    def test_estimator_params_accept_nested_values(self):
        """Real params include lists and dicts. A ValidationError here would cost the whole run."""
        result = ModelResult(name="xgb", params={"max_depth": [3, 5], "cb": {"eta": 0.1}})
        assert result.params["max_depth"] == [3, 5]

    def test_sample_values_are_capped(self):
        with pytest.raises(ValidationError):
            ColumnProfile(
                name="c",
                dtype="object",
                missing_fraction=0.0,
                n_unique=9,
                sample_values=list("abcdef"),
            )


class TestReviewLoop:
    def test_loop_exhausted_reads_the_frozen_cap(self):
        state = PipelineState(dataset_id="toy", task_description="x", config=RunConfig(loop_cap=3))
        assert not state.loop_exhausted
        state.review_iterations = 3
        assert state.loop_exhausted

    def test_open_objections_respects_review_pass_dispositions(self):
        """Silence and acceptance must not look the same."""
        raised = leak_objection()
        state = PipelineState(dataset_id="toy", task_description="x", objections=[raised])
        assert len(state.open_objections()) == 1

        state.review_passes = [
            ReviewPass(
                iteration=1,
                claim="pass",
                routed_to="reporter",
                dispositions={raised.id: "resolved"},
            )
        ]
        assert state.open_objections() == []

    def test_an_unmentioned_objection_stays_open(self):
        """A reviewer that goes quiet has not accepted anything."""
        raised = leak_objection()
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            objections=[raised],
            review_passes=[
                ReviewPass(
                    iteration=1,
                    claim="pass",
                    routed_to="reporter",
                    dispositions={raised.id: "not_reviewed"},
                )
            ],
        )
        assert len(state.open_objections()) == 1

    def test_open_objections_filters_by_target_node(self):
        state = populated_state()
        assert state.open_objections(target_node="feature_eng")[0].category == "leakage"
        assert state.open_objections(target_node="modeler") == []


class TestOpenObjectionsOrdering:
    def test_sorts_by_iteration_not_list_position(self):
        """Resolved in pass 1, reopened in pass 2: the objection must read as OPEN.

        `review_passes` is built with iteration=2 listed before iteration=1 to prove the
        method sorts by `iteration`, not by whatever order the list happens to be in.
        """
        objection = leak_objection()
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            objections=[objection],
            review_passes=[
                ReviewPass(
                    iteration=2,
                    claim="block",
                    routed_to="feature_eng",
                    dispositions={objection.id: "still_open"},
                ),
                ReviewPass(
                    iteration=1,
                    claim="pass",
                    routed_to="reporter",
                    dispositions={objection.id: "resolved"},
                ),
            ],
        )
        assert state.open_objections() == [objection]


class TestClosureIsMeasured:
    """`objections_resolved`, `objections_withdrawn`, `objections_falsely_resolved`.

    The closure arm's primary endpoint is whether the reviewer closes anything at all, and its
    falsifier is whether it closes DISHONESTLY -- an objection marked `resolved` while its column
    is still in `final_features`. That failure mode does not exist in any earlier cell, because no
    earlier reviewer resolved anything, so nothing on the row would have caught it.
    """

    def _state(
        self,
        objections: list[Objection],
        dispositions: dict[str, str],
        final_features: list[str] | None = None,
        reviewer_enabled: bool = True,
        with_pass: bool = True,
    ) -> PipelineState:
        passes = []
        if with_pass:
            passes = [
                ReviewPass(
                    iteration=0, claim="pass", routed_to="reporter", dispositions=dispositions
                )
            ]
        return PipelineState(
            config=RunConfig(reviewer_enabled=reviewer_enabled),
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=["account_status_code"],
            objections=objections,
            review_passes=passes,
            final_features=final_features,
        )

    def test_a_reviewer_that_closed_nothing_is_a_real_zero_not_none(self):
        """The pre-closure finding, as a number. If this were `None` the control cell would have
        no result at all."""
        ob = leak_objection()
        row = self._state([ob], {ob.id: "still_open"}, ["tenure_months"]).results_row()
        assert row["objections_resolved"] == 0
        assert row["objections_withdrawn"] == 0
        assert row["objections_falsely_resolved"] == 0

    def test_a_reviewer_that_never_ran_is_none(self):
        ob = leak_objection()
        row = self._state([ob], {}, ["tenure_months"], with_pass=False).results_row()
        assert row["objections_resolved"] is None
        assert row["objections_withdrawn"] is None
        assert row["objections_falsely_resolved"] is None

    def test_the_reviewer_off_arm_is_none(self):
        """The reviewer-off arm still runs the node as a no-op, so `review_passes` alone is not
        enough to say the reviewer was asked anything."""
        row = self._state([], {}, ["tenure_months"], reviewer_enabled=False).results_row()
        assert row["objections_resolved"] is None

    def test_resolved_and_withdrawn_are_counted_separately(self):
        """They are opposite claims about the reviewer, and `binding_objections` acts on the
        difference, so a row that summed them could not be used to reason about what was dropped."""
        kept = leak_objection(columns=["a"])
        gone = leak_objection(columns=["b"])
        row = self._state(
            [kept, gone], {kept.id: "withdrawn", gone.id: "resolved"}, ["tenure_months"]
        ).results_row()
        assert row["objections_resolved"] == 1
        assert row["objections_withdrawn"] == 1

    def test_an_honest_resolution_does_not_count_as_false(self):
        ob = leak_objection(columns=["account_status_code"])
        row = self._state([ob], {ob.id: "resolved"}, ["tenure_months"]).results_row()
        assert row["objections_falsely_resolved"] == 0

    def test_resolving_an_objection_whose_column_is_still_in_the_matrix_is_false(self):
        """The arm's falsifier. A prompt that buys termination by teaching the reviewer to say
        'fixed' is worse than no prompt, and this is the only column that would show it."""
        ob = leak_objection(columns=["account_status_code"])
        row = self._state(
            [ob], {ob.id: "resolved"}, ["tenure_months", "account_status_code"]
        ).results_row()
        assert row["objections_falsely_resolved"] == 1

    def test_a_resolved_non_column_scoped_objection_is_never_false(self):
        """An `overfit` objection names no column, so absence from `final_features` cannot be
        checked for it and it must not be scored either way."""
        ob = leak_objection(category="overfit", subcategory="cv_gap", columns=[])
        row = self._state([ob], {ob.id: "resolved"}, ["tenure_months"]).results_row()
        assert row["objections_falsely_resolved"] == 0

    def test_an_empty_matrix_makes_false_resolution_unmeasurable(self):
        """Same rule as `leakage_remediated`: an empty matrix contains no column, so every
        resolution would score honest and the inflation would flatter the arm under test. The two
        counts beside it stay real integers."""
        ob = leak_objection()
        row = self._state([ob], {ob.id: "resolved"}, []).results_row()
        assert row["objections_falsely_resolved"] is None
        assert row["objections_resolved"] == 1

    def test_objections_open_at_end_is_unchanged_by_any_of_this(self):
        ob = leak_objection()
        row = self._state([ob], {ob.id: "resolved"}, ["tenure_months"]).results_row()
        assert row["objections_open_at_end"] == 0


class TestABindingObjectionOutlivesItsResolution:
    """`binding_objections` answers "what must stay OUT of the matrix". `open_objections` answers
    "what is still being complained about". They are different questions and 2026-08-28 is the day
    that stopped being a distinction without a difference.

    `_forced_drops` recomputes from scratch on every entry to `feature_eng`, so while it read
    `open_objections`, a `resolved` objection stopped forcing its drop and the next return to that
    node -- for any reason at all -- put the leaked column back. Verified live against the node:
    with the objection open the snippet read `DROP = ['account_status_code', 'churned',
    'customer_id']`; with it resolved, `DROP = ['churned', 'customer_id']`.
    """

    def _state(
        self,
        objection: Objection,
        disposition: str | None,
        routing: str = "as_addressed",
    ) -> PipelineState:
        passes = []
        if disposition is not None:
            passes = [
                ReviewPass(
                    iteration=0,
                    claim="pass",
                    routed_to="reporter",
                    dispositions={objection.id: disposition},
                )
            ]
        return PipelineState(
            config=RunConfig(objection_routing=routing),
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=["leaky_col"],
            objections=[objection],
            review_passes=passes,
        )

    def test_a_resolved_objection_is_still_binding(self):
        """`resolved` means the problem is fixed, and on this pipeline the fix IS the drop. An
        objection that stopped binding on resolution would un-fix itself."""
        objection = leak_objection(columns=["leaky_col"])
        assert self._state(objection, "resolved").binding_objections() == [objection]

    def test_a_withdrawn_objection_is_not_binding(self):
        """The only release. `withdrawn` is the reviewer saying it was never a problem, which is
        the opposite claim to `resolved` and the only route back for a false positive."""
        objection = leak_objection(columns=["leaky_col"])
        assert self._state(objection, "withdrawn").binding_objections() == []

    def test_a_not_reviewed_objection_is_still_binding(self):
        objection = leak_objection(columns=["leaky_col"])
        assert self._state(objection, "not_reviewed").binding_objections() == [objection]

    def test_open_objections_is_unchanged_by_any_of_this(self):
        """The guard that the two questions stayed separate. `open_objections` must keep closing on
        `resolved`: the router asks it where to send the run, and an objection that kept routing
        upstream after being resolved would loop to the cap on every run and make `exhausted`
        structurally guaranteed -- destroying the signal the closure arm exists to make honest."""
        objection = leak_objection(columns=["leaky_col"])
        state = self._state(objection, "resolved")
        assert state.open_objections() == []
        assert state.binding_objections() == [objection]

    def test_binding_objections_inherits_the_effective_target(self):
        """Stops the new method becoming a SECOND answer to "who acts on this". Same misaddressed
        objection as TestObjectionRouting, resolved, under `by_category`: it must bind on
        `feature_eng`, because that is the node the graph decided acts."""
        objection = leak_objection(
            category="implausible_importance",
            subcategory="importance_dominance",
            target_node="modeler",
            columns=["leaky_col"],
        )
        state = self._state(objection, "resolved", routing="by_category")
        assert state.binding_objections("feature_eng") == [objection]
        assert state.binding_objections("modeler") == []

    def test_the_last_disposition_wins_not_the_first(self):
        """Resolved in pass 0, withdrawn in pass 1: the release must land. Same fold, same
        ordering rule as `open_objections`, because they share one implementation."""
        objection = leak_objection(columns=["leaky_col"])
        state = PipelineState(
            config=RunConfig(),
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=["leaky_col"],
            objections=[objection],
            review_passes=[
                ReviewPass(
                    iteration=1,
                    claim="pass",
                    routed_to="reporter",
                    dispositions={objection.id: "withdrawn"},
                ),
                ReviewPass(
                    iteration=0,
                    claim="block",
                    routed_to="feature_eng",
                    dispositions={objection.id: "resolved"},
                ),
            ],
        )
        assert state.binding_objections() == []


class TestTheForcedDropReleaseRuleIsARecordedCondition:
    """`config.forced_drop_release` is the control arm for the sticky-drop fix above.

    The fix moved `leakage_remediated` 5/10 -> 9/10, the largest single effect in this project, and
    its only evidence was two cells run at different commits. `resolved_or_withdrawn` reproduces the
    pre-2026-08-28 release rule ON THIS COMMIT, so the claim "the release rule caused the effect"
    has a counterfactual instead of a code-boundary footnote.

    It is a defect reproduction, not a design fork, which is why the default is inverted relative to
    every other condition on `RunConfig` and why that inversion is pinned by a test rather than left
    to a comment. See DECISIONS.md 2026-08-28 (fifth entry).
    """

    def _state(
        self,
        objection: Objection,
        disposition: str | None,
        *,
        release: str = "withdrawn_only",
        routing: str = "as_addressed",
    ) -> PipelineState:
        passes = []
        if disposition is not None:
            passes = [
                ReviewPass(
                    iteration=0,
                    claim="pass",
                    routed_to="reporter",
                    dispositions={objection.id: disposition},
                )
            ]
        return PipelineState(
            config=RunConfig(objection_routing=routing, forced_drop_release=release),
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=["leaky_col"],
            objections=[objection],
            review_passes=passes,
        )

    def test_the_default_release_rule_is_the_fixed_behaviour_and_deliberately_not_the_old_one(self):
        """The inversion, pinned, because it looks like an inconsistency and is not.

        `naming`, `reviewer_prompt`, `objection_routing` and `objection_closure` all default to the
        arm that reproduces every committed row byte for byte, because each is a real design
        question with two defensible answers. This one is not: `resolved_or_withdrawn` is a bug. A
        future session that "restores consistency" by flipping this default would silently make the
        buggy pipeline the shipped one.
        """
        assert RunConfig().forced_drop_release == "withdrawn_only"

    def test_the_unsticky_arm_is_exactly_open_objections_again(self):
        """The control arm's fidelity guarantee. It is not an approximation of the pre-fix
        predicate; it IS that predicate, so a difference between the arms cannot be an artefact of
        having reimplemented the bug slightly differently."""
        objections = [
            leak_objection(columns=["a"], target_node="feature_eng"),
            leak_objection(columns=["b"], target_node="feature_eng"),
            leak_objection(columns=["c"], target_node="modeler"),
            leak_objection(columns=["d"], target_node="modeler"),
        ]
        dispositions = dict(
            zip(
                [o.id for o in objections],
                ["still_open", "resolved", "withdrawn", "not_reviewed"],
                strict=True,
            )
        )
        state = PipelineState(
            config=RunConfig(forced_drop_release="resolved_or_withdrawn"),
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=["leaky_col"],
            objections=objections,
            review_passes=[
                ReviewPass(
                    iteration=0, claim="pass", routed_to="reporter", dispositions=dispositions
                )
            ],
        )
        for target in (None, "feature_eng", "modeler"):
            assert state.binding_objections(target) == state.open_objections(target), target

    def test_a_resolved_objection_stops_binding_only_under_the_unsticky_arm(self):
        """The pair for `test_a_resolved_objection_is_still_binding`. This single difference is the
        whole of what the cell measures."""
        objection = leak_objection(columns=["leaky_col"])
        assert self._state(objection, "resolved").binding_objections() == [objection]
        assert (
            self._state(objection, "resolved", release="resolved_or_withdrawn").binding_objections()
            == []
        )

    def test_a_not_reviewed_objection_binds_under_both_release_rules(self):
        """Silence is never release, in either arm. `not_reviewed` is a label the model is not
        allowed to produce, so the rule against silence-as-approval must not become arm-dependent --
        otherwise the two arms would differ in more than the release rule."""
        objection = leak_objection(columns=["leaky_col"])
        for release in ("withdrawn_only", "resolved_or_withdrawn"):
            assert self._state(objection, "not_reviewed", release=release).binding_objections() == [
                objection
            ], release

    def test_the_release_rule_never_changes_open_objections(self):
        """Blast radius. The router and the reviewer both read `open_objections`, and a condition
        that leaked into it would route a resolved objection upstream forever in one arm, making
        `exhausted` structurally guaranteed there and confounding the cell with a loop-length
        difference."""
        objection = leak_objection(columns=["leaky_col"])
        for disposition in ("still_open", "resolved", "withdrawn", "not_reviewed"):
            withdrawn_only = self._state(objection, disposition).open_objections()
            unsticky = self._state(
                objection, disposition, release="resolved_or_withdrawn"
            ).open_objections()
            assert withdrawn_only == unsticky, disposition

    def test_the_release_rule_and_the_routing_condition_stay_orthogonal(self):
        """The two conditions answer different questions -- who acts, and what releases -- and the
        cell crosses `by_category` with both release rules, so a coupling between them would make
        every number in it uninterpretable."""
        objection = leak_objection(
            category="implausible_importance",
            subcategory="importance_dominance",
            target_node="modeler",
            columns=["leaky_col"],
        )
        sticky = self._state(objection, "resolved", routing="by_category")
        assert sticky.binding_objections("feature_eng") == [objection]

        unsticky = self._state(
            objection, "resolved", release="resolved_or_withdrawn", routing="by_category"
        )
        assert unsticky.binding_objections("feature_eng") == []
        # Still routed by category in both arms: the release rule decides WHETHER it binds, never
        # WHO it binds on.
        assert unsticky.binding_objections("modeler") == []


class TestTheSingleCallerInvariant:
    """`binding_objections` is documented in two places as having exactly one caller in the graph,
    and until now nothing enforced it.

    The router's `_route_for_block` and the reviewer's `_user_message` must keep asking
    `open_objections`. A resolved objection that still routed the run upstream would loop to the cap
    on every run; one still shown to the reviewer would be re-adjudicated forever. Either would
    destroy the signal the closure and sticky-drop cells exist to measure, and neither would fail a
    test -- both are silently-wrong-number failures, which is the class this repo cares most about.
    """

    def test_binding_objections_has_exactly_one_caller_in_the_graph(self):
        import ast
        from pathlib import Path

        import ds_agents

        package_root = Path(ds_agents.__file__).parent
        callers: set[tuple[str, str]] = set()
        for path in sorted(package_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "binding_objections"
                    ):
                        callers.add((str(path.relative_to(package_root)), node.name))

        assert callers == {("nodes/feature_eng.py", "_forced_drops")}, (
            f"`binding_objections` must have exactly one caller in the graph and now has "
            f"{sorted(callers)}. It answers 'what must stay OUT of the matrix' and releases only "
            f"on `withdrawn`. Anything deciding where to ROUTE a run, or what to SHOW the "
            f"reviewer, must ask `open_objections` instead: a resolved objection that still routed "
            f"upstream would loop to the cap on every run and make `exhausted` structurally "
            f"guaranteed, and one still shown to the reviewer would be re-adjudicated forever."
        )


class TestObjectionRouting:
    """`config.objection_routing` decides WHO ACTS on an objection, and only that.

    The bug being fixed, live on 2026-08-28: the reviewer raises `implausible_importance` -- a
    column-scoped category -- and addresses it to `modeler`, which is a defensible reading of its
    own prompt and a node with no column lever at all. `feature_eng` force-drops objected columns
    but only sees `open_objections("feature_eng")`, so a correct objection sent one node sideways
    produced exactly the same results row as a hallucinated one. 0 of 21 runs that never routed to
    `feature_eng` remediated, against 3 of 6 that did.
    """

    def _state(self, objections: list[Objection], routing: str = "as_addressed") -> PipelineState:
        return PipelineState(
            config=RunConfig(objection_routing=routing),
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=["leaky_col"],
            objections=objections,
        )

    def _misaddressed(self) -> Objection:
        """The exact shape the reviewer produced live: right column, wrong node."""
        return leak_objection(
            category="implausible_importance",
            subcategory="importance_dominance",
            target_node="modeler",
            columns=["leaky_col"],
        )

    def test_the_default_routing_obeys_the_reviewers_choice(self):
        """The BEFORE picture as a unit test: feature_eng cannot see it, so nothing can act."""
        objection = self._misaddressed()
        state = self._state([objection])
        assert state.open_objections("modeler") == [objection]
        assert state.open_objections("feature_eng") == []

    def test_by_category_gives_a_column_scoped_objection_to_feature_eng(self):
        """The AFTER picture. Same objection, same reviewer, opposite answer to who acts."""
        objection = self._misaddressed()
        state = self._state([objection], routing="by_category")
        assert state.open_objections("feature_eng") == [objection]
        assert state.open_objections("modeler") == []

    def test_a_non_column_scoped_objection_is_never_rerouted(self):
        """Pins the rule to COLUMN_SCOPED_CATEGORIES rather than "everything addressed to the
        modeler". An `overfit` complaint really is the modeler's, and rerouting it would send the
        run upstream to a node with nothing to drop."""
        objection = leak_objection(
            category="overfit",
            subcategory="cv_holdout_gap",
            target_node="modeler",
            columns=[],
        )
        state = self._state([objection], routing="by_category")
        assert state.open_objections("modeler") == [objection]
        assert state.open_objections("feature_eng") == []

    def test_the_raw_target_node_survives_the_reroute(self):
        """This is the "recorded condition, not a thumb on the scale" argument, as an assertion.

        `by_category` overrides where the run goes. It must NOT overwrite what the reviewer said,
        because `objections_by_target_node` is the only evidence that the reviewer's dispatch
        judgement was the problem -- and folding the effective target into that counter would
        delete the finding in exactly the arm that exists to demonstrate it.
        """
        state = self._state([self._misaddressed()], routing="by_category")
        assert state.objections[0].target_node == "modeler"
        row = state.results_row()
        assert row["objections_by_target_node"] == {"feature_eng": 0, "modeler": 1}
        assert row["objection_routing"] == "by_category"

    def test_objections_rerouted_counts_only_the_overridden_ones(self):
        """Three objections, one of which the graph disagrees with the reviewer about."""
        objections = [
            self._misaddressed(),
            leak_objection(columns=["leaky_col"]),
            leak_objection(
                category="overfit", subcategory="cv_gap", target_node="modeler", columns=[]
            ),
        ]
        assert self._state(objections, "by_category").results_row()["objections_rerouted"] == 1
        assert self._state(objections).results_row()["objections_rerouted"] == 0

    def test_n_final_features_is_the_price_tag_on_the_reroute(self):
        """`leakage_remediated` is None on an EMPTY matrix but True on a one-column one, and
        `by_category` turns a reviewer false positive into a really dropped feature. Without a
        width beside it, a run that "remediated" by force-dropping most of the fixture reads
        identically to one that dropped only the trap."""
        state = self._state([], routing="by_category")
        state.final_features = ["one_survivor"]
        row = state.results_row()
        assert row["n_final_features"] == 1
        assert row["leakage_remediated"] is True

    def test_n_final_features_is_null_when_feature_eng_never_ran(self):
        assert self._state([]).results_row()["n_final_features"] is None


class TestDerivedNumbers:
    def test_score_ratio_is_direction_aware(self):
        higher = PipelineState(
            dataset_id="d",
            task_description="x",
            spec=TaskSpec(target="y", task_type="binary", metric="roc_auc"),
            verified_holdout_score=0.8,
            baseline_score=0.7,
        )
        lower = PipelineState(
            dataset_id="d",
            task_description="x",
            spec=TaskSpec(target="y", task_type="regression", metric="rmse"),
            verified_holdout_score=0.7,
            baseline_score=0.8,
        )
        assert higher.score_ratio > 1
        assert lower.score_ratio > 1, "beating an RMSE baseline means a LOWER score"

    def test_cv_mean_is_none_without_scores(self):
        assert ModelResult(name="empty").cv_mean is None
        assert ModelResult(name="m", cv_scores=[0.5, 0.7]).cv_mean == pytest.approx(0.6)

    def test_total_cost_sums_the_node_trace(self):
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            node_trace=[
                NodeEvent(node="intake", started=datetime(2026, 8, 22, tzinfo=UTC), cost_usd=0.01),
                NodeEvent(
                    node="profiler", started=datetime(2026, 8, 22, tzinfo=UTC), cost_usd=0.02
                ),
            ],
        )
        assert state.total_cost_usd == pytest.approx(0.03)

    def test_timestamps_are_timezone_aware(self):
        """Naive datetimes make committed JSONL inconsistent across machines."""
        assert PipelineState(dataset_id="t", task_description="x").started_at.tzinfo is not None

    def test_zero_baseline_r2_gives_none_ratio_without_raising(self):
        """The predict-the-mean baseline for r2 is exactly 0.0. A ratio against it is meaningless,
        not an exception."""
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            spec=TaskSpec(target="y", task_type="regression", metric="r2"),
            verified_holdout_score=0.4,
            baseline_score=0.0,
        )
        assert state.score_ratio is None
        assert state.results_row()["score_ratio"] is None

    def test_zero_verified_rmse_gives_none_ratio_without_raising(self):
        """rmse 0.0 is what a perfect leaked copy scores. Dividing by it must not raise."""
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            spec=TaskSpec(target="y", task_type="regression", metric="rmse"),
            verified_holdout_score=0.0,
            baseline_score=1.2,
        )
        assert state.score_ratio is None
        assert state.results_row()["score_ratio"] is None


class TestResultsRow:
    def test_catching_the_planted_leak_scores_as_a_catch(self):
        row = populated_state().results_row()
        assert row["leakage_caught"] is True
        assert row["leakage_recall"] == 1.0
        assert row["false_alarm"] == 0

    def test_caught_and_remediated_are_separate_numbers(self):
        """Flagging the leak and actually removing it are different events."""
        state = populated_state()
        state.final_features = ["support_tickets_90d", "account_status_code"]
        row = state.results_row()
        assert row["leakage_caught"] is True
        assert row["leakage_remediated"] is False

    def test_flagging_a_clean_column_is_a_false_alarm_not_a_catch(self):
        state = populated_state()
        state.objections = [leak_objection(columns=["region"])]
        row = state.results_row()
        assert row["leakage_caught"] is False
        assert row["false_alarm_columns"] == ["region"]
        assert row["leakage_precision"] == 0.0

    def test_claimed_and_verified_scores_are_both_reported(self):
        """The gap between what the agent said and what we measured is itself a finding."""
        row = populated_state().results_row()
        assert row["claimed_holdout_score"] == 0.81
        assert row["verified_holdout_score"] == 0.74
        assert row["holdout_claim_gap"] == pytest.approx(0.07)

    def test_row_is_json_serialisable(self):
        json.dumps(populated_state().results_row())

    def test_the_naming_condition_is_on_the_row(self):
        """A leakage number is not interpretable without it. The two naming arms run over
        byte-identical rows, so a row missing this is indistinguishable from the other arm's."""
        assert populated_state().results_row()["naming"] == "descriptive"

    def test_an_opaque_run_says_so(self):
        state = populated_state()
        state.config = RunConfig(naming="opaque")
        assert state.results_row()["naming"] == "opaque"

    def test_the_reviewer_prompt_condition_is_on_the_row_and_defaults_to_base(self):
        """Same rule as `naming`: a reviewer number that does not say which prompt produced it is
        confounded by the prompt, and the arms are otherwise byte-identical."""
        assert populated_state().results_row()["reviewer_prompt"] == "base"
        state = populated_state()
        state.config = RunConfig(reviewer_prompt="which_column")
        assert state.results_row()["reviewer_prompt"] == "which_column"


class TestProfilerColumnsOnTheRow:
    """The profiler's nominations, scored separately from the reviewer's objections.

    They answer different questions and on the trap fixtures they give different answers: the
    profiler nominates the planted column and the reviewer, shown the same run, says nothing. The
    name-transparency ablation moves this number and not the reviewer's, so without these fields
    the experiment's dependent variable is absent from its own results file.
    """

    def _state(self, nominated: list[str], planted: list[str]) -> PipelineState:
        return PipelineState(
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=planted,
            profile=ProfileReport(
                n_rows=200,
                n_columns=8,
                leakage_candidates=[
                    LeakageCandidate(
                        column=column, reason="looks post hoc", evidence="nmi 0.4", suspicion="high"
                    )
                    for column in nominated
                ],
            ),
        )

    def test_nominating_the_planted_column_scores_as_a_catch(self):
        row = self._state(["leaky_col"], ["leaky_col"]).results_row()
        assert row["profiler_caught"] is True
        assert row["profiler_recall"] == 1.0
        assert row["profiler_false_alarm"] == 0
        assert row["profiler_nominated"] == ["leaky_col"]

    def test_one_of_two_traps_is_half_recall(self):
        """claims_timing plants two columns, so this is a real value and not a rounding of 1.0."""
        row = self._state(["trap_a"], ["trap_a", "trap_b"]).results_row()
        assert row["profiler_recall"] == 0.5
        assert row["profiler_caught"] is True

    def test_nominating_a_clean_column_is_a_profiler_false_alarm(self):
        """The blind spot this closes: an id column flagged upstream never reached a results row,
        because the reviewer never objected to it and nothing else was counting."""
        row = self._state(["customer_id"], ["leaky_col"]).results_row()
        assert row["profiler_caught"] is False
        assert row["profiler_false_alarm"] == 1
        assert row["profiler_recall"] == 0.0

    def test_looking_and_finding_nothing_is_zero_not_null(self):
        row = self._state([], ["leaky_col"]).results_row()
        assert row["profiler_caught"] is False
        assert row["profiler_recall"] == 0.0
        assert row["profiler_nominated"] == []

    def test_a_profiler_that_never_ran_is_null_not_zero(self):
        """A node that crashed nominated nothing in a different sense than one that declined to,
        and averaging those together over a benchmark would be a lie."""
        state = PipelineState(
            dataset_id="toy", task_description="x", planted_leakage_columns=["leaky_col"]
        )
        row = state.results_row()
        assert row["profiler_nominated"] is None
        assert row["profiler_caught"] is None
        assert row["profiler_recall"] is None
        assert row["profiler_false_alarm"] is None

    def test_no_planted_columns_means_no_recall(self):
        row = self._state(["a"], []).results_row()
        assert row["profiler_recall"] is None
        assert row["profiler_false_alarm"] == 1


class TestReviewerColumnsOnTheRow:
    """The reviewer's columns, over every column-scoped category.

    The gap this closes, observed live on 2026-08-27: the reviewer named columns under
    `implausible_importance`, and `leakage_flagged` counts only leakage and contamination, so a
    reviewer that names the trap in that category scored as a miss. `leakage_*` keeps its
    two-category definition -- the committed naming-ablation rows were written under it -- and
    these fields carry the wider question: did the reviewer name the trap column at all.
    """

    def _state(self, objections: list[Objection], planted: list[str]) -> PipelineState:
        return PipelineState(
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=planted,
            objections=objections,
            review_passes=[ReviewPass(iteration=1, claim="pass", routed_to="reporter")],
        )

    def test_naming_the_trap_under_leakage_scores_as_a_catch(self):
        row = self._state([leak_objection(columns=["leaky_col"])], ["leaky_col"]).results_row()
        assert row["reviewer_caught"] is True
        assert row["reviewer_recall"] == 1.0
        assert row["reviewer_false_alarm"] == 0
        assert row["reviewer_nominated"] == ["leaky_col"]

    def test_naming_the_trap_under_implausible_importance_also_counts(self):
        """The divergence is deliberate: the same row scores `leakage_caught` False, because that
        field keeps the two-category definition the naming-ablation rows were written under."""
        objection = leak_objection(
            category="implausible_importance",
            subcategory="importance_dominance",
            columns=["leaky_col"],
        )
        row = self._state([objection], ["leaky_col"]).results_row()
        assert row["reviewer_caught"] is True
        assert row["reviewer_nominated"] == ["leaky_col"]
        assert row["leakage_caught"] is False
        assert row["leakage_flagged"] == []

    def test_one_of_two_traps_is_half_recall(self):
        row = self._state([leak_objection(columns=["trap_a"])], ["trap_a", "trap_b"]).results_row()
        assert row["reviewer_recall"] == 0.5
        assert row["reviewer_caught"] is True

    def test_a_non_column_objection_counts_in_categories_but_nominates_nothing(self):
        objection = leak_objection(category="metric_mismatch", subcategory="cv_gap", columns=[])
        row = self._state([objection], ["leaky_col"]).results_row()
        assert row["reviewer_nominated"] == []
        assert row["objections_by_category"]["metric_mismatch"] == 1

    def test_naming_a_clean_column_is_a_reviewer_false_alarm(self):
        row = self._state([leak_objection(columns=["region"])], ["leaky_col"]).results_row()
        assert row["reviewer_caught"] is False
        assert row["reviewer_false_alarm"] == 1
        assert row["reviewer_recall"] == 0.0

    def test_looking_and_raising_nothing_is_zero_not_null(self):
        row = self._state([], ["leaky_col"]).results_row()
        assert row["reviewer_caught"] is False
        assert row["reviewer_recall"] == 0.0
        assert row["reviewer_nominated"] == []

    def test_a_reviewer_that_never_completed_a_pass_is_null_not_zero(self):
        state = PipelineState(
            dataset_id="toy", task_description="x", planted_leakage_columns=["leaky_col"]
        )
        row = state.results_row()
        assert row["reviewer_nominated"] is None
        assert row["reviewer_caught"] is None
        assert row["reviewer_recall"] is None
        assert row["reviewer_false_alarm"] is None

    def test_a_disabled_reviewer_is_null_even_with_a_pass_recorded(self):
        state = self._state([], ["leaky_col"])
        state.config = RunConfig(reviewer_enabled=False)
        row = state.results_row()
        assert row["reviewer_caught"] is None
        assert row["reviewer_nominated"] is None

    def test_objections_by_category_always_has_every_key(self):
        row = self._state([], []).results_row()
        assert set(row["objections_by_category"]) == {
            "leakage",
            "contamination",
            "overfit",
            "metric_mismatch",
            "implausible_importance",
            "spec_violation",
            "other",
        }
        assert all(count == 0 for count in row["objections_by_category"].values())


class TestResultsRowBranches:
    """One assertion-focused test per `results_row()` branch that a full `populated_state()`
    run can never exercise, because its fields are always set together."""

    def test_no_chosen_model_means_no_claim_or_gap(self):
        state = PipelineState(dataset_id="toy", task_description="x", verified_holdout_score=0.7)
        row = state.results_row()
        assert row["claimed_holdout_score"] is None
        assert row["holdout_claim_gap"] is None

    def test_no_planted_leakage_means_no_recall_or_remediation(self):
        state = PipelineState(dataset_id="toy", task_description="x", final_features=["a"])
        row = state.results_row()
        assert row["leakage_recall"] is None
        assert row["leakage_remediated"] is None

    def test_no_flags_means_no_precision(self):
        state = PipelineState(dataset_id="toy", task_description="x", planted_leakage_columns=["c"])
        row = state.results_row()
        assert row["leakage_precision"] is None

    def test_empty_final_features_is_not_remediation(self):
        """A crashed feature_eng leaves final_features=[], which trivially contains no planted
        column. That must not score as remediation."""
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=["leaky_col"],
            final_features=[],
        )
        row = state.results_row()
        assert row["leakage_remediated"] is None

    def test_holdout_claim_gap_is_positive_for_overstatement_in_both_directions(self):
        """Positive must always mean 'the agent overstated itself', whichever way the metric
        runs. Both states overstate by the same 0.05 margin."""
        roc_state = PipelineState(
            dataset_id="toy",
            task_description="x",
            spec=TaskSpec(target="y", task_type="binary", metric="roc_auc"),
            chosen_model=ModelResult(name="m", claimed_holdout_score=0.85),
            verified_holdout_score=0.80,
        )
        rmse_state = PipelineState(
            dataset_id="toy",
            task_description="x",
            spec=TaskSpec(target="y", task_type="regression", metric="rmse"),
            chosen_model=ModelResult(name="m", claimed_holdout_score=0.80),
            verified_holdout_score=0.85,
        )
        assert roc_state.results_row()["holdout_claim_gap"] == pytest.approx(0.05)
        assert rmse_state.results_row()["holdout_claim_gap"] == pytest.approx(0.05)

    def test_withdrawn_objection_still_counts_raised_but_not_standing(self):
        """An objection raised then withdrawn in a later pass stays in the ever-raised numbers
        but must drop out of the standing ones -- a reviewer that takes back a bad flag is
        behaving better than one that never looks again."""
        objection = leak_objection(columns=["region"])
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=["other_col"],
            objections=[objection],
            review_passes=[
                ReviewPass(
                    iteration=1,
                    claim="pass",
                    routed_to="reporter",
                    dispositions={objection.id: "withdrawn"},
                )
            ],
        )
        row = state.results_row()
        assert row["leakage_flagged"] == ["region"]
        assert row["false_alarm"] == 1
        assert row["leakage_flagged_standing"] == []
        assert row["false_alarm_standing"] == 0


class TestWhyTheLoopDidNotConverge:
    """The four fields that separate "the reviewer was wrong" from "the reviewer was right and
    told a node with no lever".

    The gap these close, observed live on 2026-08-28: 9 of 10 opaque Haiku runs named a planted
    trap and 1 of 10 removed it. The committed rows could not say why, because nothing on them
    recorded who the objection was addressed to, where the loop actually went, or whether an
    objected column was still in the matrix at the end. Diagnostic runs found two causes -- an
    `implausible_importance` objection routed to `modeler`, which has no column lever at all, and
    a reviewer that never dispositions its own objection `resolved` even after the drop lands.
    Each field below is what makes one of those visible in a results file.
    """

    def _state(
        self,
        objections: list[Objection],
        *,
        final_features: list[str] | None = None,
        passes: list[ReviewPass] | None = None,
    ) -> PipelineState:
        return PipelineState(
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=["leaky_col"],
            objections=objections,
            final_features=final_features,
            review_passes=passes or [ReviewPass(iteration=1, claim="pass", routed_to="reporter")],
        )

    def test_target_node_counts_split_by_who_was_asked(self):
        """The misrouting signature: two objections, one addressed to a node that can act and one
        to a node that cannot."""
        row = self._state(
            [
                leak_objection(columns=["leaky_col"]),
                leak_objection(
                    category="implausible_importance",
                    subcategory="importance_dominance",
                    target_node="modeler",
                    columns=["leaky_col"],
                ),
            ]
        ).results_row()
        assert row["objections_by_target_node"] == {"feature_eng": 1, "modeler": 1}

    def test_target_node_counts_always_have_every_key(self):
        """Same reason `objections_by_category` does: a missing key and a zero must not be the
        same thing to whoever averages these later."""
        row = self._state([]).results_row()
        assert row["objections_by_target_node"] == {"feature_eng": 0, "modeler": 0}

    def test_an_objected_column_still_in_the_matrix_is_unremediated(self):
        row = self._state(
            [leak_objection(columns=["leaky_col"])],
            final_features=["leaky_col", "region"],
        ).results_row()
        assert row["objected_columns_unremediated"] == ["leaky_col"]

    def test_dropping_the_objected_column_empties_the_list(self):
        row = self._state(
            [leak_objection(columns=["leaky_col"])], final_features=["region"]
        ).results_row()
        assert row["objected_columns_unremediated"] == []

    def test_unremediated_spans_every_column_scoped_category(self):
        """`implausible_importance` is the category the live reviewer actually uses, so a field
        that only looked at `leakage` would have reported an empty list on every run that
        motivated it."""
        row = self._state(
            [
                leak_objection(
                    category="implausible_importance",
                    subcategory="importance_dominance",
                    target_node="modeler",
                    columns=["leaky_col"],
                )
            ],
            final_features=["leaky_col"],
        ).results_row()
        assert row["objected_columns_unremediated"] == ["leaky_col"]

    def test_an_empty_matrix_is_null_not_an_empty_list(self):
        """Matches `leakage_remediated`: a feature_eng that produced nothing has not remediated
        anything, and scoring it as a clean list would inflate the headline rate with runs that
        produced no model."""
        row = self._state([leak_objection(columns=["leaky_col"])], final_features=[]).results_row()
        assert row["objected_columns_unremediated"] is None
        assert row["leakage_remediated"] is None

    def test_a_reviewer_that_never_looked_is_null_not_an_empty_list(self):
        """Same guard `reviewer_nominated` uses. An empty list here would say "columns were
        objected to and all of them were dropped", which is the opposite of what the
        reviewer-off arm did."""
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            planted_leakage_columns=["leaky_col"],
            final_features=["leaky_col"],
            config=RunConfig(reviewer_enabled=False),
        )
        row = state.results_row()
        assert row["objected_columns_unremediated"] is None
        assert row["reviewer_nominated"] is None

    def test_a_reviewer_that_looked_and_raised_nothing_is_an_empty_list(self):
        """The other side of the guard above: a completed pass that objected to nothing has
        nothing unremediated, and that zero is real rather than missing."""
        row = self._state([], final_features=["leaky_col"]).results_row()
        assert row["objected_columns_unremediated"] == []

    def test_route_sequence_and_new_objections_follow_iteration_order(self):
        """Constructed out of order on purpose: `review_passes` is an append-reduced list and
        nothing guarantees the order it arrives in."""
        first = leak_objection(columns=["leaky_col"])
        second = leak_objection(columns=["other_col"], raised_at_iteration=1)
        row = self._state(
            [first, second],
            final_features=["region"],
            passes=[
                ReviewPass(
                    iteration=2,
                    claim="block",
                    routed_to="reporter",
                    new_objection_ids=[second.id],
                ),
                ReviewPass(
                    iteration=1,
                    claim="block",
                    routed_to="feature_eng",
                    new_objection_ids=[first.id],
                ),
            ],
        ).results_row()
        assert row["route_sequence"] == ["feature_eng", "reporter"]
        assert row["new_objections_per_pass"] == [1, 1]

    def test_a_run_with_no_pass_has_empty_sequences(self):
        state = PipelineState(dataset_id="toy", task_description="x")
        row = state.results_row()
        assert row["route_sequence"] == []
        assert row["new_objections_per_pass"] == []


# --- the placeholder guard --------------------------------------------------------------------
# Nothing structurally stopped a StubModel run from being written to a results file. These cover
# the gate that does.


def test_a_trace_containing_a_stub_event_is_not_publishable():
    state = PipelineState(dataset_id="toy", task_description="t")
    state.node_trace = [
        NodeEvent(node="intake", started=utc_now(), model="claude-haiku-4-5"),
        NodeEvent(node="profiler", started=utc_now(), model="stub"),
    ]
    ok, reason = state.publishable()
    assert ok is False
    assert "stub" in reason
    assert state.placeholder_models() == ["stub"]


def test_a_fully_real_trace_is_publishable():
    state = PipelineState(dataset_id="toy", task_description="t")
    state.node_trace = [NodeEvent(node="intake", started=utc_now(), model="claude-haiku-4-5")]
    assert state.publishable() == (True, "")
    assert state.placeholder_models() == []


def test_an_empty_trace_is_not_publishable():
    # A row with no events would report cost 0.0 and look like a free, successful run.
    ok, reason = PipelineState(dataset_id="toy", task_description="t").publishable()
    assert ok is False
    assert "nothing ran" in reason


def test_a_node_that_called_no_model_does_not_count_as_a_placeholder():
    # The router and reporter run no model, so their events carry model=None. Treating that as a
    # placeholder would make every real run unpublishable.
    state = PipelineState(dataset_id="toy", task_description="t")
    state.node_trace = [
        NodeEvent(node="intake", started=utc_now(), model="claude-haiku-4-5"),
        NodeEvent(node="router", started=utc_now(), model=None),
    ]
    assert state.publishable()[0] is True
