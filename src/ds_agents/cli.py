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
import difflib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ds_agents import rescore
from ds_agents.fixtures import load_fixture
from ds_agents.graph import run_pipeline
from ds_agents.holdout import PreparedDataset, prepare
from ds_agents.naming import NAMINGS, Naming, rename_map
from ds_agents.naming import apply as apply_rename
from ds_agents.provenance import git_commit
from ds_agents.runnable import Runnable, available, resolve
from ds_agents.state import (
    FORCED_DROP_RELEASES,
    OBJECTION_CLOSURES,
    OBJECTION_ROUTINGS,
    REVIEWER_PROMPTS,
    ForcedDropRelease,
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


def _run_state(
    runnable: Runnable,
    *,
    prepared: PreparedDataset | None = None,
    model_name: str = "haiku",
    reviewer_model_name: str | None = None,
    naming: Naming = "descriptive",
    reviewer_prompt: ReviewerPrompt = "base",
    loop_cap: int = 3,
    objection_routing: ObjectionRouting = "as_addressed",
    objection_closure: ObjectionClosure = "off",
    forced_drop_release: ForcedDropRelease = "withdrawn_only",
    commit: str | None = None,
) -> PipelineState:
    """The starting state for one run of `runnable` under one naming condition.

    Takes a `Runnable` rather than a `Fixture` so a manifest dataset can reach a run at all,
    and takes it as one object rather than as loose fields for the reason the star below
    exists: `planted_columns` travelling separately from the dataset it describes is how a
    run gets graded against the wrong answer key. The empty list a benchmark dataset carries
    is not special-cased here -- `results_row()` reads the emptiness and returns `None` from
    every leakage rate rather than scoring one.

    Everything after `runnable` is keyword-only. Every parameter here is a run condition, they are
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
    rename = rename_map(runnable, naming)
    # Derived from the SAME object the agents were mounted on, not recomputed from `runnable`.
    # A config claiming a 20% carve while the mounted CSV was the whole file is exactly the shape
    # of silent mislabelling the star above exists to prevent, one level further down.
    if prepared is not None and prepared.rename != rename:
        raise ValueError(
            f"{runnable.dataset_id}: the prepared file was written under a different rename than "
            f"{naming!r} produces, so the ground truth would name columns the agents never saw"
        )
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
            # Recorded, and the one condition here whose default is NOT the pre-2026-08-28
            # behaviour: `resolved_or_withdrawn` reproduces a defect rather than offering a second
            # defensible design. The two arms differ in whether a resolved objection's column can
            # come back into the matrix, so their remediation rates are not comparable and
            # averaging them would report a capability the control arm does not have.
            forced_drop_release=forced_drop_release,
            # Which tree ran. Not a condition anyone sets, but the only field that can tell two
            # rows apart when the difference between them is a bug fix rather than a flag -- which
            # is every unconditional change this project has made, the block-retry included.
            commit=commit,
            # What the agents were actually shown: how much was held back, which registry it came
            # from, and the hash of the exact bytes. Without the first, a benchmark row and a
            # fixture row look like the same kind of measurement; without the third, two rows
            # under one `dataset_id` that saw different files are indistinguishable.
            holdout_fraction=prepared.withheld_fraction if prepared else 0.0,
            dataset_source=runnable.source,
            dataset_hash=prepared.agent_sha256 if prepared else None,
        ),
        dataset_id=runnable.dataset_id,
        # No `spec`: naming the target is intake's job, and pre-filling it here would skip the
        # node under test. The description is what a person would actually say.
        task_description=runnable.task_description,
        # Ground truth, written at construction so a bare run can grade itself. Nodes never set
        # this; the reviewer must find the leak without being told where it is. Renamed to match
        # what the agents were actually shown -- without the map applied here, `results_row()`
        # would compare objections against names that do not exist in the opaque arm's data and
        # every opaque run would score a silent zero.
        planted_leakage_columns=apply_rename(runnable.planted_columns, rename),
        # Recorded at construction so a run that never reaches the grader still says how many rows
        # were held back from it. `rescore.apply` overwrites it with what was actually scored,
        # which is smaller wherever a withheld row had no label.
        n_withheld_rows=prepared.n_withheld_rows if prepared else None,
    )


def _toy_state(model_name: str = "haiku", reviewer_model_name: str | None = None) -> PipelineState:
    """The toy fixture's state, by name. Kept as its own function because tests call it."""
    return _run_state(
        Runnable.from_fixture(load_fixture("toy")),
        model_name=model_name,
        reviewer_model_name=reviewer_model_name,
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
        dataset = resolve(args.dataset)
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
    # Prepared once per invocation, not once per repetition: every run in an arm must see the
    # same bytes AND be graded on the same withheld rows, and doing it per run is N chances for
    # them not to be. `withheld` sits beside `input`, outside every run root -- see `holdout.py`.
    prepared = prepare(
        dataset,
        args.naming,
        into=root / "input",
        withheld_into=root / "withheld",
        seed=RunConfig().random_seed,
    )
    if prepared.rename:
        print(f"naming: {args.naming} -- {prepared.agent_csv}", file=sys.stderr)
    if prepared.withheld_csv is not None:
        print(
            f"withheld {prepared.n_withheld_rows} of "
            f"{prepared.n_withheld_rows + prepared.n_agent_rows} rows before the graph starts; "
            f"the agents see {prepared.n_agent_rows}",
            file=sys.stderr,
        )

    exit_code = 0
    width = len(str(args.repeat - 1))
    # Once, before the first run, not once per run: `--results` makes the tree dirty by writing to
    # it, so a per-run read would record run 0 at `<hash>` and every later run at `<hash>-dirty`
    # and split one cell in two. See `_run_once`.
    commit = git_commit()
    for index in range(args.repeat):
        # One artifact store per run, because the store is per-run: a shared root would let run 2
        # read run 1's artifact ids, which is the one way these repetitions could stop being
        # independent.
        # Always its own directory, even at `--repeat 1`. The grader writes under the run root
        # too, and a run root that is sometimes the invocation root is a second layout for the
        # containment test to have to know about.
        run_root = root / f"run-{index:0{width}d}"
        if args.repeat > 1:
            print(f"\n=== run {index + 1} of {args.repeat} ===", file=sys.stderr)
        state = _run_once(
            dataset,
            root=run_root,
            prepared=prepared,
            commit=commit,
            transport=args.tools,
            no_live=args.no_live,
            model_name=args.model,
            reviewer_model_name=args.reviewer_model,
            naming=args.naming,
            reviewer_prompt=args.reviewer_prompt,
            loop_cap=args.loop_cap,
            objection_routing=args.objection_routing,
            objection_closure=args.objection_closure,
            forced_drop_release=args.forced_drop_release,
        )

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
    runnable: Runnable,
    *,
    root: Path,
    prepared: PreparedDataset,
    commit: str | None,
    transport: str = "mcp",
    no_live: bool = False,
    **conditions: Any,
) -> PipelineState:
    """One pipeline run, start to finish. Extracted so `--repeat` is a loop and not a second path.

    `prepared` rather than `runnable.csv_path`: the agents see a materialised copy with a
    rewritten header under `--naming opaque`, and on a benchmark dataset they see a copy with 20%
    of the rows removed as well. Nothing below this line knows either -- both happen above the
    tools boundary, which is why no node, tool or MCP change was needed for either of them.

    The grader runs here rather than in the caller, after `run_pipeline` returns and before the
    run's tools close, because `read_inputs` needs the run's own store and nothing else does. Its
    result is written onto the state with `rescore.apply`, which never appends a `PipelineError`:
    `errored` means the run went wrong, and a grader that could not grade is a different fact.

    Takes keywords rather than the `argparse.Namespace` it used to, so the harness can call it
    without inventing a fake namespace. `conditions` is forwarded straight to `_run_state`,
    whose parameters are keyword-only -- the transposition guard that star exists for survives the
    hop, and the two callers cannot drift into two different ideas of what a run condition is.

    `commit` is a required keyword rather than a `git_commit()` call in the body, and it has no
    default, because the default was wrong in a way only a multi-run invocation could show. Reading
    provenance per run meant every run after the first saw a tree that the harness had itself
    dirtied by writing the results file, so run 0 recorded `<hash>` and runs 1..N recorded
    `<hash>-dirty`. Since `commit` is one of `evaldiff.CONDITION_FIELDS`, that split a single cell
    into two, which is the one thing the field exists to prevent. Callers snapshot it once before
    the first run: provenance describes the tree that produced the INVOCATION, and this is the same
    once-per-invocation rule `_live_run` already follows for `materialize`. `None` stays a
    legitimate value (git missing, not a checkout), which is why there is no sentinel default.
    """
    tools = _select_tools(transport, root, prepared.agent_csv, runnable.dataset_id)
    state = _run_state(runnable, prepared=prepared, commit=commit, **conditions)
    # Said out loud for the same reason the StubModel warning is: this arm reproduces a known
    # defect, and a run that produced numbers under it without anyone noticing would be worse than
    # no run. A stderr line reads nothing any node reads, so the condition still has exactly one
    # application site -- the release set in `PipelineState.binding_objections`.
    if state.config.forced_drop_release != "withdrawn_only":
        print(
            f"WARNING: --forced-drop-release {state.config.forced_drop_release} reproduces a known "
            f"defect: a resolved objection stops forcing its drop, so a leaked column can come "
            f"back on the next return to feature_eng. This arm exists only as the control of the "
            f"sticky-drop cell. Do not use it as a baseline for anything else.",
            file=sys.stderr,
        )
    model = _select_model(state.config, no_live=no_live)
    reviewer_model = _select_reviewer_model(state.config, model, no_live=no_live)
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
        state = run_pipeline(state, tools=tools, model=model, reviewer_model=reviewer_model)
        # Read while the run's store is still open; there is nothing to read it from afterwards.
        inputs = rescore.read_inputs(state, tools)
    finally:
        # Releases this run's claim on the tools. Under `local` that leaves the shared sandbox
        # worker running, which is the point of sharing it -- it exits with this process. Under
        # `mcp` it disconnects the session and the server subprocess, and its worker, exit with it.
        tools.close()

    if isinstance(inputs, rescore.RescoreOutcome):
        return rescore.apply(state, inputs)
    return rescore.apply(state, rescore.rescore(state, prepared, inputs, root=root / "rescore"))


def _append_results_row(
    state: PipelineState, path: Path, *, extra: dict[str, Any] | None = None
) -> bool:
    """One JSONL line per run, behind the gate the harness and `--results` both apply.

    `publishable()` is checked here and not by the caller because this is the only place a number
    leaves a run and lands in a file someone will later average. A stub run refused at this line is
    the difference between a results file and a file that looks like one. Returns whether it wrote,
    so the harness can count refusals -- a refused row changes a cell's denominator.

    `extra` carries the harness's write-time annotations (`cell`, `replicate`, ...), which are facts
    about the sampling design of an invocation rather than about what the run did. Everything that
    describes the RUN comes from `results_row()` off the frozen config, where no node could have
    read it and nothing outside the run can disagree with it.
    """
    publishable, reason = state.publishable()
    if not publishable:
        print(f"  results row  : REFUSED -- {reason}", file=sys.stderr)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(state.results_row() | (extra or {})) + "\n")
    print(f"  results row  : appended to {path}", file=sys.stderr)
    return True


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
    """The benchmark harness. Imports are deferred to the call for a real reason: `harness` imports
    `_run_once` and `_append_results_row` from this module, so a top-level import here would be a
    cycle. The clean fix is a `runner.py` holding the pieces both need; it is logged in DECISIONS
    as the refactor to take when it earns its keep, rather than churning three test modules today.
    """
    from ds_agents.harness import RESULTS_DIR, run_eval

    # Caught here beside `cmd_run`'s `--repeat` and `--loop-cap` checks, and for the same reason: a
    # zero here produces an empty plan and a bare "0 rows written", which reads like the harness
    # failed rather than like the flag was wrong.
    for flag, value in (("--replicates", args.replicates), ("--n", args.n)):
        if value < 1:
            print(f"{flag} must be at least 1, got {value}", file=sys.stderr)
            return 2

    try:
        report = run_eval(
            subset=args.subset,
            name=args.name,
            replicates=args.replicates,
            n=args.n,
            max_cost_usd=args.max_cost_usd,
            artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
            out_dir=Path(args.out_dir) if args.out_dir else RESULTS_DIR,
            transport=args.tools,
            no_live=args.no_live,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    # A cell that wrote nothing is not a cell. Exit non-zero so a scripted or CI invocation cannot
    # pass on an invocation that spent money and published no row.
    if not args.dry_run and report.rows_written == 0:
        return 1
    return 0


def cmd_eval_diff(args: argparse.Namespace) -> int:
    from ds_agents.evaldiff import compare, load_rows, render

    before, after = Path(args.before), Path(args.after)
    for path in (before, after):
        if not path.exists():
            print(f"no such results file: {path}", file=sys.stderr)
            return 2
    print(render(compare(load_rows(before), load_rows(after))))
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    """The benchmark registry's three verbs.

    `refresh` is the only write to `evals/datasets/` in the repo -- the manifest is a generated
    artifact and Edit/Write are denied there, which is what makes "no number in it was typed"
    a property of the tooling rather than a promise. `verify --online` is the counterpart: it
    re-fetches and diffs, so the file can be checked rather than trusted.
    """
    from ds_agents import benchmark

    if args.datasets_command == "list":
        names = benchmark.available()
        if not names:
            print("no manifest yet -- run `ds-agents datasets refresh`", file=sys.stderr)
            return 1
        manifest = benchmark.load_manifest()
        print(f"{len(manifest.datasets)} datasets, {len(manifest.excluded)} excluded")
        for entry in manifest.datasets:
            reference = (
                f"{entry.published_reference.value:.3f}" if entry.published_reference else "--"
            )
            print(
                f"  {entry.dataset_id:<28} data={entry.openml_data_id:<6} "
                f"task={entry.openml_task_id:<7} {entry.n_rows:>7} x {entry.n_features:<4} "
                f"usable={entry.n_usable_features:<4} pos={entry.positive_rate:.3f} "
                f"published_auc={reference}"
            )
        return 0

    from ds_agents import benchmark_build

    if args.datasets_command == "refresh":
        manifest = benchmark_build.refresh()
        print(
            f"wrote {benchmark.MANIFEST_PATH} -- {len(manifest.datasets)} datasets, "
            f"{len(manifest.excluded)} excluded"
        )
        low, high = manifest.selection.target_count
        if not low <= len(manifest.datasets) <= high:
            print(
                f"WARNING: {len(manifest.datasets)} datasets is outside the target range "
                f"{low}-{high}. The rule decided this; do not hand-pick to close the gap.",
                file=sys.stderr,
            )
        return 0

    if args.datasets_command == "verify":
        on_disk = benchmark.MANIFEST_PATH.read_text()
        if not args.online:
            benchmark.load_manifest()
            print("manifest parses and matches the schema (offline check only)")
            return 0
        rebuilt = benchmark_build.render(benchmark_build.build())
        if rebuilt == on_disk:
            print("manifest matches a fresh rebuild")
            return 0
        print("manifest DIFFERS from a fresh rebuild:", file=sys.stderr)
        for line in difflib.unified_diff(
            on_disk.splitlines(), rebuilt.splitlines(), "on-disk", "rebuilt", lineterm="", n=1
        ):
            print(line, file=sys.stderr)
        return 1

    raise AssertionError(f"unreachable datasets command {args.datasets_command!r}")


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
        help=(
            "dataset to run (default: toy). Fixtures carry a planted answer key; benchmark\n"
            "datasets from evals/datasets/manifest.yaml do not, and get a withheld holdout\n"
            "instead. Available: "
            + (", ".join(f"{n} ({s})" for n, s in available().items()) or "(none)")
        ),
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
        "--forced-drop-release",
        default="withdrawn_only",
        choices=FORCED_DROP_RELEASES,
        help="which disposition releases a column an objection forced out of the matrix: "
        "withdrawn_only (default, and correct), or resolved_or_withdrawn, which reproduces the "
        "pre-2026-08-28 defect where a resolved objection stopped forcing its drop. The second "
        "value is a known bug, not a design alternative: it exists only as the control arm of the "
        "sticky-drop cell and must not be the baseline of anything else.",
    )
    run.add_argument(
        "--results",
        default=None,
        help="append one results_row() JSONL line per run to this path. Rows that fail "
        "publishable() are refused, not written. For one-off cells and debugging: these rows carry "
        "no cell or replicate annotation, so they cannot enter a powered comparison. The harness "
        "is `ds-agents eval`.",
    )
    run.add_argument(
        "--no-live",
        action="store_true",
        help="force the placeholder model even when a key is available",
    )
    run.set_defaults(func=cmd_run)

    ev = sub.add_parser("eval", help="run the benchmark harness")
    ev.add_argument("--subset", default="ci", help="toy, ci, or full (full is not implemented yet)")
    ev.add_argument(
        "--name",
        required=True,
        help="what is different about this run, in a filename: `reviewer-haiku`, not `test3`. "
        "Lowercase letters, digits and hyphens.",
    )
    ev.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="how many times to repeat the whole subset (default: 1). A comparison needs at least "
        "2 per arm -- eval-diff refuses to call a single-replicate difference an effect.",
    )
    ev.add_argument("--n", type=int, default=1, help="runs per cell per replicate (default: 1)")
    ev.add_argument(
        "--max-cost-usd",
        type=float,
        default=0.50,
        help="stop cleanly before the run that would exceed this (default: 0.50)",
    )
    ev.add_argument("--artifacts-dir", default=None, help="where run artifacts land")
    ev.add_argument("--out-dir", default=None, help="where the results file lands")
    ev.add_argument("--tools", default="mcp", choices=("mcp", "local"))
    ev.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the cost estimate, spend nothing, write nothing",
    )
    ev.add_argument(
        "--no-live",
        action="store_true",
        help="force the placeholder model; every row will be refused by publishable()",
    )
    ev.set_defaults(func=cmd_eval)

    diff = sub.add_parser("eval-diff", help="compare two results files")
    diff.add_argument("before", help="the earlier results JSONL")
    diff.add_argument("after", help="the later results JSONL")
    diff.set_defaults(func=cmd_eval_diff)

    ds = sub.add_parser("datasets", help="the external benchmark dataset registry")
    ds_sub = ds.add_subparsers(dest="datasets_command", required=True)
    ds_sub.add_parser("list", help="what is in evals/datasets/manifest.yaml")
    ds_sub.add_parser("refresh", help="re-fetch and rewrite the manifest (the only writer)")
    ds_verify = ds_sub.add_parser("verify", help="check the manifest against its sources")
    ds_verify.add_argument(
        "--online",
        action="store_true",
        help="re-fetch everything and diff; without it only the schema is checked",
    )
    ds.set_defaults(func=cmd_datasets)

    return parser


def main() -> int:
    # The one environment read in the whole package. `override=False` so an explicitly exported
    # key beats the file, which is what makes a one-off `ANTHROPIC_API_KEY=... uv run` work.
    load_dotenv(REPO_ROOT / ".env", override=False)
    args = _build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
