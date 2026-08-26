"""Command line entry point.

Phase 1 runs the front of the graph: intake -> profiler. The remaining nodes land in later
phases, so a `run` today ends after profiling rather than producing a model.

With no API key configured this falls back to `StubModel`, which is not a model. Every event it
produces is stamped `model="stub"` and the run prints a warning, because a results row built from
a stub would look like a system that never finds anything.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from ds_agents.graph import run_pipeline
from ds_agents.state import PipelineState
from ds_agents.tools.llm import StubModel
from ds_agents.tools.local import LocalTools

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _toy_state() -> PipelineState:
    manifest_path = FIXTURES / "toy" / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"toy fixture not found at {manifest_path}. This path assumes an editable install; "
            f"run from a checkout with `uv run ds-agents`."
        )
    manifest = json.loads(manifest_path.read_text())
    return PipelineState(
        dataset_id="toy",
        # No `spec`: naming the target is intake's job, and pre-filling it here would skip the
        # node under test. The description is what a person would actually say.
        task_description=f"Predict {manifest['target']} and report {manifest['metric']}.",
        # Ground truth, written at construction so a bare toy run can grade itself. Nodes never
        # set this; the reviewer must find the leak without being told where it is.
        planted_leakage_columns=[leak["column"] for leak in manifest["planted_leakage"]],
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


def cmd_run(args: argparse.Namespace) -> int:
    if args.dataset != "toy":
        print(f"only the toy dataset exists so far, not {args.dataset!r}", file=sys.stderr)
        return 2

    root = Path(args.artifacts_dir) if args.artifacts_dir else Path(tempfile.mkdtemp())
    tools = LocalTools(root, dataset_path=FIXTURES / "toy" / "toy.csv", dataset_id="toy")
    model = StubModel()
    print(
        f"WARNING: running with {model.name!r}, which is a placeholder and not a model. It "
        f"nominates no leakage candidates by design. Numbers from this run are not results.",
        file=sys.stderr,
    )

    state = run_pipeline(_toy_state(), tools=tools, model=model)

    print(state.model_dump_json(indent=2, exclude_none=True))
    _print_trace(state)
    print(f"\nartifacts: {root}", file=sys.stderr)
    print("graph ends after profiler; the rest of the nodes land in Phase 1-3.", file=sys.stderr)
    return 1 if any(not e.recoverable for e in state.errors) else 0


def cmd_eval(args: argparse.Namespace) -> int:
    print(
        f"eval harness lands in Phase 4; subset {args.subset!r} not runnable yet", file=sys.stderr
    )
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(prog="ds-agents", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline on one dataset")
    run.add_argument("--dataset", default="toy")
    run.add_argument(
        "--artifacts-dir", default=None, help="where the run's artifacts land (default: a tempdir)"
    )
    run.set_defaults(func=cmd_run)

    ev = sub.add_parser("eval", help="run the benchmark harness")
    ev.add_argument("--subset", default="ci")
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
