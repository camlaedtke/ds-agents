"""What `ArtifactStore.register_dataset` costs, at every shape in the manifest.

Opt in:

    DS_AGENTS_TIMING_TESTS=1 uv run pytest tests/test_register_dataset_cost.py -s

Gated by an environment variable rather than a marker, following `tests/test_baseline_cost.py` and
`tests/test_benchmark_provenance.py`, so that the default `uv run pytest` stays both clean and
complete.

**Why this function gets its own table.** It runs BEFORE anything is measured. `register_dataset`
copies the CSV, does an unbounded `pd.read_csv`, and computes a per-column `nunique` -- and it does
that three times per benchmark run: once for the run's own tools, and once inside each of
`rescore.rescore` and `rescore.baseline`, which each construct a fresh `LocalTools`
(`src/ds_agents/rescore.py:475` and `:632`). None of that appears in `wall_seconds`,
which starts when the graph does, nor in `rescore_seconds` or `baseline_seconds`, which time the
sandbox call and not the tool construction around it. So it is real wall cost that no committed
results row has ever carried.

`docs/DECISIONS.md` recorded it for the datasets that could be run on 2026-09-01 -- 0.01s at
`phoneme` up to 0.17s at `nomao` -- as prose, in a commit that touched no code, so the numbers could
not be re-derived or extended. `docs/NEXT.md` then carried "Never measured" for the four datasets
that could not complete a run at that commit. This replaces both with a table anyone can re-run,
covering all thirteen.

**Only ONE of the three calls is bounded by anything, and that is the finding.** There is no
timeout on `register_dataset` itself.

- Call 1, the graph's own tools, goes through `cli._select_tools`. Under the default `--tools mcp`
  it runs inside the server subprocess before the MCP `initialize` handshake returns, so it is
  bounded by `MCPTools._CONNECT_TIMEOUT_S = 60` (`src/ds_agents/tools/mcp_client.py:48`). A breach
  raises `SystemExit`, which `harness.run_eval` re-raises rather than counting -- so it aborts a
  whole invocation rather than one run, and `MAX_CONSECUTIVE_FAILURES` never sees it. Under
  `--tools local` this call has no bound either.
- Calls 2 and 3, inside `rescore.rescore` and `rescore.baseline`, construct `LocalTools` directly
  and never pass through `_select_tools`. They are therefore **unbounded under BOTH transports**,
  always. A hang there does not raise, does not abort, and does not appear in any results column --
  it simply blocks the run forever, which is strictly worse than call 1's failure mode.

So `--tools` is not the axis. The graph's registration is the only one any timeout has ever been
able to see, and the two the grader makes are the ones with no brake at all. That is worth fixing
deliberately rather than in a pricing session; what this test does is establish the size of the
exposure so the fix can be scoped.

The assertion below is therefore made against `_CONNECT_TIMEOUT_S` for a SINGLE call -- the only
call that bound actually governs -- with the three-call per-run figure printed beside it as the
cost number rather than as an asserted one.

`HEADROOM` is 10.0 rather than `test_baseline_cost.py`'s 5.0, and the reason is not caution: the
60s connect budget is not registration's to spend. It also has to cover interpreter boot, importing
`mcp_server`, and the handshake itself. Registration has to be a small fraction of that budget, not
merely under it.

The assertion is bounded by headroom against the real production timeout rather than by an absolute
second count chosen here, which is the same reasoning `test_baseline_cost.py` gives. That is not
the same as being machine-independent -- it is still a wall-clock measurement, and a sufficiently
slow machine can legitimately trip it. The 10x headroom is what makes that unlikely rather than
impossible, and a failure should be read as "measure this again", not as an automatic bug.
"""

import os
import shutil
import time

import pandas as pd
import pytest

from ds_agents.benchmark import cached_csv_path, load_manifest
from ds_agents.tools.mcp_client import _CONNECT_TIMEOUT_S
from mcp_server.store import ArtifactStore

pytestmark = pytest.mark.skipif(
    os.environ.get("DS_AGENTS_TIMING_TESTS") != "1",
    reason="copies and parses every cached benchmark CSV; set DS_AGENTS_TIMING_TESTS=1",
)

# How much of the connect budget three registrations may consume before this test calls it a
# problem. See the module docstring: the budget is shared with interpreter boot, the `mcp_server`
# import and the MCP handshake, so registration must be a small fraction rather than merely under.
HEADROOM = 10.0

# Call sites per benchmark run: the run's own tools, plus a fresh `LocalTools` inside each of
# `rescore.rescore` and `rescore.baseline`. Pinned as a property by
# `TestTheDatasetIsRegisteredOncePerToolSurface` in tests/test_rescore.py rather than left as a
# number in a comment. Only the first is bounded by `_CONNECT_TIMEOUT_S`; see the module docstring.
CALLS_PER_RUN = 3


def test_registration_clears_the_only_bound_on_it_at_every_manifest_shape(tmp_path):
    entries = sorted(load_manifest().datasets, key=lambda e: e.n_rows * e.n_features)

    print(f"\n`ArtifactStore.register_dataset`, {CALLS_PER_RUN} calls per benchmark run")
    print(
        f"asserted on call 1 of {CALLS_PER_RUN} (the graph's tools, the only bounded one) against "
        f"MCPTools._CONNECT_TIMEOUT_S = {_CONNECT_TIMEOUT_S}s, headroom {HEADROOM}x."
    )
    print("calls 2 and 3 (rescore, baseline) are unbounded under both transports; x3 is cost.\n")
    header = (
        f"{'dataset':24s}{'MB':>7s}{'rows':>8s}{'cols':>6s}"
        f"{'copy':>8s}{'read':>8s}{'nunique':>9s}{'total':>8s}{'x3':>8s}{'% budget':>10s}"
    )
    print(header)

    slowest = 0.0
    measured = 0
    for entry in entries:
        source = cached_csv_path(entry)
        if not source.exists():
            print(f"{entry.dataset_id:24s}{'-- not cached, skipped':>56s}")
            continue

        # The real call, timed whole: this is the number that matters. The per-phase columns below
        # re-time the same work separately, only so the table says WHERE the time goes.
        store = ArtifactStore(tmp_path / entry.dataset_id)
        started = time.perf_counter()
        store.register_dataset(source, entry.dataset_id)
        elapsed = time.perf_counter() - started

        landed = tmp_path / f"{entry.dataset_id}-phases.csv"
        started = time.perf_counter()
        shutil.copyfile(source, landed)
        landed.chmod(0o444)
        copy_s = time.perf_counter() - started
        started = time.perf_counter()
        frame = pd.read_csv(landed)
        read_s = time.perf_counter() - started
        started = time.perf_counter()
        {c: int(frame[c].nunique(dropna=True)) for c in frame.columns}
        nunique_s = time.perf_counter() - started

        per_run = elapsed * CALLS_PER_RUN
        slowest = max(slowest, per_run)
        measured += 1
        share = 100 * per_run / _CONNECT_TIMEOUT_S
        print(
            f"{entry.dataset_id:24s}{source.stat().st_size / 1e6:7.1f}"
            f"{len(frame):8d}{len(frame.columns):6d}"
            f"{copy_s:8.2f}{read_s:8.2f}{nunique_s:9.2f}{elapsed:8.2f}{per_run:8.2f}{share:9.1f}%"
        )

        assert elapsed * HEADROOM < _CONNECT_TIMEOUT_S, (
            f"{entry.dataset_id} registers in {elapsed:.1f}s ({per_run:.1f}s across the "
            f"{CALLS_PER_RUN} calls a benchmark run makes) against the {_CONNECT_TIMEOUT_S}s "
            f"connect budget the graph's registration shares with interpreter boot and the MCP "
            f"handshake -- outside the {HEADROOM}x headroom this test requires. A breach on that "
            "call raises SystemExit, which run_eval re-raises, so it aborts the WHOLE eval "
            "invocation rather than one run; a breach on the rescore or baseline call is not "
            "bounded at all and simply hangs. Cache the registration across the three call sites, "
            "or bound the grader's two deliberately."
        )

    if not measured:
        pytest.skip("no benchmark CSVs cached; run `uv run ds-agents datasets refresh`")

    print(
        f"\nslowest: {slowest:.2f}s per run of a {_CONNECT_TIMEOUT_S}s budget "
        f"({100 * slowest / _CONNECT_TIMEOUT_S:.1f}%)\n"
    )
