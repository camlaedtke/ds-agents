"""Command line entry point.

This is also the ONLY place that reads the environment. `.env` is loaded here, not in a node and
not in `tools/`, because `state.py`'s contract says nodes never touch the environment -- which is
what lets `tools/local.py` hand the sandbox a stripped env with no API key in it. A node that
loaded dotenv would put the key one `os.environ` call away from agent-authored code.

With no API key this falls back to `StubModel`, which is not a model. Every event it produces is
stamped `model="stub"`, the run prints a warning, and `PipelineState.publishable()` refuses the
run, because a results row built from a stub would look like a system that never finds anything.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from ds_agents.fixtures import Fixture, available, load_fixture
from ds_agents.graph import run_pipeline
from ds_agents.naming import NAMINGS, Naming, materialize, rename_map
from ds_agents.naming import apply as apply_rename
from ds_agents.state import (
    OBJECTION_CLOSURES,
    OBJECTION_ROUTINGS,
    REVIEWER_PROMPTS,
    ObjectionClosure,
    ObjectionRouting,
    PipelineState,
    ReviewerPrompt,
    RunConfig,
)
from ds_agents.tools.llm import AnthropicModel, StubModel, api_key_present
from ds_agents.tools.local import LocalTools
from ds_agents.tools.mcp_client import MCPTools, stdio_params
from ds_agents.tools.pricing import UnknownModelError
from ds_agents.tools.protocol import Tools

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fixture_state(
    fixture: Fixture,
    *,
    model_name: str = "haiku",
    reviewer_model_name: str | None = None,
    naming: Naming = "descriptive",
    reviewer_prompt: ReviewerPrompt = "base",
    loop_cap: int = 3,
    objection_routing: ObjectionRouting = "as_addressed",
    objection_closure: ObjectionClosure = "off",
) -> PipelineState:
    """The starting state for one run of `fixture` under one naming condition.

    Everything after `fixture` is keyword-only. Every parameter here is a run condition, they are
    mostly strings, and `_run_once` passed all of them positionally: one transposition would have
    published a run under the wrong arm's label with nothing to catch it, because every value is
    individually valid and `RunConfig` is frozen at construction. The parking lot carried this as
    a hazard for two sessions; the star is the fix.

    The rename map is derived here from `naming` rather than passed in alongside it. An earlier
    version took both and let the caller supply them, which meant `naming="opaque"` with an empty
    map was constructible: the config would claim the opaque arm while the ground truth still
    carried the fixture's real column names, every objection would be compared against columns that
    do not exist in that arm's data, and the arm would score a silent zero. That is precisely the
    failure this whole change exists to prevent, so the two cannot be separate arguments.
    `rename_map` is pure and reads one line of the CSV, and its determinism is pinned by test, so
    deriving it twice costs nothing and cannot disagree with what `materialize` wrote.
    """
    rename = rename_map(fixture, naming)
    return PipelineState(
        config=RunConfig(
            reviewer_enabled=True,
            # Which column names the agents saw. Same reason as `reviewer_model` below: the two
            # naming arms run over byte-identical rows, so a row that did not carry this would be
            # indistinguishable from a row in the other arm.
            naming=naming,
            # Recorded, not decorative. `results_row()` reports `reviewer_model` straight off this
            # object, so a config that says "haiku" while --model sonnet ran would publish a
            # Sonnet-everywhere run under a Haiku label and silently corrupt the ablation table.
            # ARCHITECTURE.md's rule is that a row is self-describing from the state alone.
            default_model=model_name,
            reviewer_model=reviewer_model_name or model_name,
            # Same rule again: the two prompt arms are byte-identical except for one appended
            # bullet in the reviewer's system prompt, so an unrecorded prompt would confound
            # every reviewer-model number written after it existed.
            reviewer_prompt=reviewer_prompt,
            # Set at construction because `RunConfig` is frozen, and recorded for the same reason
            # every other condition is: the cap decides how many chances feature_eng gets to act
            # on an objection, so two rows written under different caps are not comparable and
            # must not be averaged by anyone who has forgotten which was which.
            loop_cap=loop_cap,
            # Recorded for the same reason as everything above it, and with a sharper edge: the
            # two arms differ in whether a column-scoped objection can be acted on at all, so
            # their remediation rates are not comparable and averaging them would report a
            # capability the `as_addressed` arm does not have.
            objection_routing=objection_routing,
            # Recorded because the arm's whole claim is about the reviewer's behaviour, and a
            # row that did not carry it would average a reviewer that was told what done
            # looks like with one that was not.
            objection_closure=objection_closure,
        ),
        dataset_id=fixture.dataset_id,
        # No `spec`: naming the target is intake's job, and pre-filling it here would skip the
        # node under test. The description is what a person would actually say.
        task_description=fixture.task_description,
        # Ground truth, written at construction so a bare run can grade itself. Nodes never set
        # this; the reviewer must find the leak without being told where it is. Renamed to match
        # what the agents were actually shown -- without the map applied here, `results_row()`
        # would compare objections against names that do not exist in the opaque arm's data and
        # every opaque run would score a silent zero.
        planted_leakage_columns=apply_rename(fixture.manifest.planted_columns, rename),
    )


def _toy_state(model_name: str = "haiku", reviewer_model_name: str | None = None) -> PipelineState:
    """The toy fixture's state, by name. Kept as its own function because tests call it."""
    return _fixture_state(
        load_fixture("toy"), model_name=model_name, reviewer_model_name=reviewer_model_name
    )


def _print_trace(state: PipelineState) -> None:
    print("\nnode trace")
    for event in state.node_trace:
        seconds = f"{event.wall_seconds:.2f}s" if event.wall_seconds is not None else "?"
        print(
            f"  {event.node:<10} {seconds:>8}  model={event.model or '-':<8} "
            f"tokens={event.input_tokens}/{event.output_tokens}  ${event.cost_usd:.4f}"
        )
    print(f"  {'total':<10} {state.wall_seconds or 0:>7.2f}s{'':>18}${state.total_cost_usd:.4f}")
    if state.errors:
        print("\nerrors")
        for error in state.errors:
            flag = "recoverable" if error.recoverable else "FATAL"
            print(f"  [{flag}] {error.node}: {error.message}")


def _select_model(config: RunConfig, *, no_live: bool = False):
    """Real client when a key resolves, placeholder otherwise, and say which out loud.

    Takes the frozen `RunConfig` rather than argv so the client and the results row cannot
    disagree about which model ran. Never silently downgrades: `--no-live` is the only way to ask
    for the stub when a key is present, so a benchmark run cannot quietly become a placeholder run
    because an env var went missing.
    """
    if no_live:
        return StubModel()
    if not api_key_present():
        print(
            "no ANTHROPIC_API_KEY found (looked in the environment and .env at the repo root); "
            "falling back to the placeholder",
            file=sys.stderr,
        )
        return StubModel()
    try:
        return AnthropicModel(model=config.default_model)
    except UnknownModelError as exc:
        raise SystemExit(str(exc)) from exc


def _select_reviewer_model(
    config: RunConfig, base: StubModel | AnthropicModel, *, no_live: bool = False
):
    """The reviewer's own client -- a second `AnthropicModel`, only when its name actually differs
    from the base model's.

    Returns `base` unchanged when `config.reviewer_model == config.default_model` (a second client
    for the identical id would just double the object, not the behaviour) or when `base` is
    already the placeholder (`--no-live`, or no key: nothing live for the reviewer to diverge to
    either). Otherwise this is `_select_model`'s twin -- same `--no-live` override, same refusal
    via `UnknownModelError` on an unpriced id -- so the Haiku/Sonnet reviewer ablation is a config
    change, never a second code path.
    """
    if no_live or isinstance(base, StubModel):
        return base
    if config.reviewer_model == config.default_model:
        return base
    try:
        return AnthropicModel(model=config.reviewer_model)
    except UnknownModelError as exc:
        raise SystemExit(str(exc)) from exc


def _select_tools(transport: str, root: Path, dataset: Path, dataset_id: str) -> Tools:
    """Same implementation either way; `mcp` puts a process boundary in front of it.

    `mcp` is the default because it is the configuration the results are produced under, and a
    default that quietly took the shortcut would mean the transport is only exercised by tests.
    `local` stays for debugging a node without a second process in the traceback.
    """
    if transport == "local":
        return LocalTools(root, dataset_path=dataset, dataset_id=dataset_id)
    print(f"tools over MCP: {sys.executable} -m mcp_server.server", file=sys.stderr)
    try:
        return MCPTools(stdio_params(root, dataset, dataset_id))
    except Exception as exc:
        # Same shape as `_select_model` refusing an unpriced model: a run that cannot reach its
        # tools has not failed, it never started, and a traceback would read like a pipeline bug.
        raise SystemExit(f"could not start the tool server: {exc!r}") from exc


def cmd_run(args: argparse.Namespace) -> int:
    try:
        fixture = load_fixture(args.dataset)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.repeat < 1:
        print(f"--repeat must be at least 1, got {args.repeat}", file=sys.stderr)
        return 2

    # Caught here rather than left to `RunConfig`'s `ge=0`, so a typo is the exit-code-2 pattern
    # beside it and not a Pydantic traceback.
    if args.loop_cap < 0:
        print(f"--loop-cap cannot be negative, got {args.loop_cap}", file=sys.stderr)
        return 2

    root = Path(args.artifacts_dir) if args.artifacts_dir else Path(tempfile.mkdtemp())
    # Materialised once per invocation, not once per repetition: every run in an arm must see the
    # same bytes, and rewriting the file ten times is ten chances for them not to.
    dataset_path, rename = materialize(fixture, args.naming, root / "input")
    if rename:
        print(f"naming: {args.naming} -- {dataset_path}", file=sys.stderr)

    exit_code = 0
    width = len(str(args.repeat - 1))
    for index in range(args.repeat):
        # One artifact store per run, because the store is per-run: a shared root would let run 2
        # read run 1's artifact ids, which is the one way these repetitions could stop being
        # independent.
        run_root = root if args.repeat == 1 else root / f"run-{index:0{width}d}"
        if args.repeat > 1:
            print(f"\n=== run {index + 1} of {args.repeat} ===", file=sys.stderr)
        state = _run_once(args, fixture, run_root, dataset_path)

        if args.repeat == 1:
            print(state.model_dump_json(indent=2, exclude_none=True))
        _print_trace(state)
        _print_summary(state, run_root)
        if args.results:
            _append_results_row(state, Path(args.results))
        if any(not e.recoverable for e in state.errors):
            exit_code = 1
    return exit_code


def _run_once(
    args: argparse.Namespace,
    fixture: Fixture,
    root: Path,
    dataset_path: Path,
) -> PipelineState:
    """One pipeline run, start to finish. Extracted so `--repeat` is a loop and not a second path.

    `dataset_path` rather than `fixture.csv_path`: under `--naming opaque` the agents see a
    materialised copy with a rewritten header. Nothing below this line knows that -- the rename is
    entirely above the tools boundary, which is why no node, tool or MCP change was needed for it.
    """
    tools = _select_tools(args.tools, root, dataset_path, fixture.dataset_id)
    state = _fixture_state(
        fixture,
        model_name=args.model,
        reviewer_model_name=args.reviewer_model,
        naming=args.naming,
        reviewer_prompt=args.reviewer_prompt,
        loop_cap=args.loop_cap,
        objection_routing=args.objection_routing,
        objection_closure=args.objection_closure,
    )
    model = _select_model(state.config, no_live=args.no_live)
    reviewer_model = _select_reviewer_model(state.config, model, no_live=args.no_live)
    if isinstance(model, StubModel):
        print(
            f"WARNING: running with {model.name!r}, which is a placeholder and not a model. It "
            f"nominates no leakage candidates by design. Numbers from this run are not results.",
            file=sys.stderr,
        )
    else:
        print(f"running live against {model.name}", file=sys.stderr)
        if reviewer_model is not model:
            print(f"reviewer running live against {reviewer_model.name}", file=sys.stderr)

    try:
        return run_pipeline(state, tools=tools, model=model, reviewer_model=reviewer_model)
    finally:
        # Releases this run's claim on the tools. Under `local` that leaves the shared sandbox
        # worker running, which is the point of sharing it -- it exits with this process. Under
        # `mcp` it disconnects the session and the server subprocess, and its worker, exit with it.
        tools.close()


def _append_results_row(state: PipelineState, path: Path) -> None:
    """One JSONL line per run, behind the same gate the Phase 4 harness will apply.

    `publishable()` is checked here and not by the caller because this is the only place a number
    leaves a run and lands in a file someone will later average. A stub run refused at this line is
    the difference between a results file and a file that looks like one.
    """
    publishable, reason = state.publishable()
    if not publishable:
        print(f"  results row  : REFUSED -- {reason}", file=sys.stderr)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(state.results_row()) + "\n")
    print(f"  results row  : appended to {path}", file=sys.stderr)


def _print_summary(state: PipelineState, root: Path) -> None:
    """What the run produced, and whether it may be published.

    The publishable line is the whole point of printing a summary: it is the same gate the Phase 4
    harness applies, shown on every manual run so a placeholder run is obvious before anyone
    quotes a number off it.
    """
    print("\nresult", file=sys.stderr)
    if state.chosen_model:
        chosen = state.chosen_model
        metric = state.spec.metric if state.spec else "?"
        print(
            f"  chosen model : {chosen.name} (claimed {metric} {chosen.claimed_holdout_score})",
            file=sys.stderr,
        )
    if state.final_features is not None:
        print(f"  features kept: {', '.join(state.final_features) or '(none)'}", file=sys.stderr)
    if state.dropped_features:
        print(f"  dropped      : {', '.join(state.dropped_features)}", file=sys.stderr)
    if state.top_importances:
        top = ", ".join(f"{n} {v:.3f}" for n, v in state.top_importances[:3])
        print(f"  top features : {top}", file=sys.stderr)
    print(f"  verdict      : {state.review_verdict}", file=sys.stderr)
    print(
        f"  objections   : {len(state.objections)} raised, {len(state.open_objections())} open",
        file=sys.stderr,
    )
    if state.report_artifact:
        print(f"  report       : {state.report_artifact}", file=sys.stderr)
    print(f"  artifacts    : {root}", file=sys.stderr)

    publishable, reason = state.publishable()
    print(
        f"  publishable  : {'yes' if publishable else f'NO -- {reason}'}",
        file=sys.stderr,
    )


def cmd_eval(args: argparse.Namespace) -> int:
    print(
        f"eval harness lands in Phase 4; subset {args.subset!r} not runnable yet", file=sys.stderr
    )
    return 2


def _build_parser() -> argparse.ArgumentParser:
    """Built here rather than inline in `main` so the flags can be tested without running a
    pipeline. `--naming`, `--repeat` and `--results` between them decide what a benchmark row
    means, and a typo in one of them is not something to discover from a wrong number later."""
    parser = argparse.ArgumentParser(prog="ds-agents", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline on one dataset")
    run.add_argument(
        "--dataset",
        default="toy",
        help=f"fixture to run (default: toy). Available: {', '.join(available()) or '(none)'}",
    )
    run.add_argument(
        "--artifacts-dir", default=None, help="where the run's artifacts land (default: a tempdir)"
    )
    run.add_argument(
        "--model",
        default="haiku",
        help="model for every node: a short name (haiku, sonnet) or a full id (default: haiku)",
    )
    run.add_argument(
        "--reviewer-model",
        default=None,
        help="model for the reviewer node only (default: --model). Set this and --model "
        "differently to run the Haiku/Sonnet reviewer ablation.",
    )
    run.add_argument(
        "--reviewer-prompt",
        default="base",
        choices=REVIEWER_PROMPTS,
        help="reviewer system prompt variant: the base prompt (default) or one appended rule "
        "asking which column explains an implausible score. This is the reviewer-prompt ablation.",
    )
    run.add_argument(
        "--tools",
        default="mcp",
        choices=("mcp", "local"),
        help="how nodes reach the sandbox and the store: over MCP (default) or in-process",
    )
    run.add_argument(
        "--naming",
        default="descriptive",
        choices=NAMINGS,
        help="column names the agents see: the fixture's own (default) or var_NN over identical "
        "rows. This is the name-transparency ablation; the arms differ in the header row only.",
    )
    run.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run the pipeline N times (default: 1). Each repetition gets its own artifacts "
        "directory. Model nondeterminism is the whole variance here -- the seed does not move.",
    )
    run.add_argument(
        "--loop-cap",
        type=int,
        default=3,
        help="how many completed reviewer passes a run may have (default: 3). The cap permits "
        "N passes and N-1 returns upstream; a block at the cap becomes the `exhausted` verdict, "
        "never a pass. This is the Phase 5 loop-cap ablation lever.",
    )
    run.add_argument(
        "--objection-routing",
        default="as_addressed",
        choices=OBJECTION_ROUTINGS,
        help="who acts on an objection: as the reviewer addressed it (default, and what every run "
        "before 2026-08-28 did), or by_category, which sends a column-scoped objection to "
        "feature_eng whatever the reviewer chose. This is the objection-routing ablation.",
    )
    run.add_argument(
        "--objection-closure",
        default="off",
        choices=OBJECTION_CLOSURES,
        help="whether the reviewer is told what 'done' looks like: off (default, and what every "
        "run before 2026-08-28 did), or on, which appends one rule saying an objection about a "
        "column is answered when that column is absent from final_features. This is the "
        "objection-closure ablation.",
    )
    run.add_argument(
        "--results",
        default=None,
        help="append one results_row() JSONL line per run to this path. Rows that fail "
        "publishable() are refused, not written.",
    )
    run.add_argument(
        "--no-live",
        action="store_true",
        help="force the placeholder model even when a key is available",
    )
    run.set_defaults(func=cmd_run)

    ev = sub.add_parser("eval", help="run the benchmark harness")
    ev.add_argument("--subset", default="ci")
    ev.set_defaults(func=cmd_eval)
    return parser


def main() -> int:
    # The one environment read in the whole package. `override=False` so an explicitly exported
    # key beats the file, which is what makes a one-off `ANTHROPIC_API_KEY=... uv run` work.
    load_dotenv(REPO_ROOT / ".env", override=False)
    args = _build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
