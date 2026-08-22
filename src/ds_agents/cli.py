"""Command line entry point.

Phase 0 only builds the initial state and prints it. There is no graph yet, so `run` deliberately
stops after intake rather than printing a node trace that does not exist.
"""

import argparse
import json
import sys
from pathlib import Path

from ds_agents.state import PipelineState, TaskSpec

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _toy_state() -> PipelineState:
    manifest = json.loads((FIXTURES / "toy" / "manifest.json").read_text())
    return PipelineState(
        dataset_id="toy",
        task_description=f"Predict {manifest['target']} and report {manifest['metric']}.",
        spec=TaskSpec(
            target=manifest["target"],
            task_type=manifest["task_type"],
            metric=manifest["metric"],
        ),
        # Ground truth, written at construction so a bare toy run can grade itself. Nodes never
        # set this; the reviewer must find the leak without being told where it is.
        planted_leakage_columns=[leak["column"] for leak in manifest["planted_leakage"]],
    )


def cmd_run(args: argparse.Namespace) -> int:
    if args.dataset != "toy":
        print(f"only the toy dataset exists so far, not {args.dataset!r}", file=sys.stderr)
        return 2
    state = _toy_state()
    print(state.model_dump_json(indent=2, exclude_none=True))
    print("\nintake state only. The graph lands in Phase 1; see docs/PLAN.md.", file=sys.stderr)
    return 0


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
    run.set_defaults(func=cmd_run)

    ev = sub.add_parser("eval", help="run the benchmark harness")
    ev.add_argument("--subset", default="ci")
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
