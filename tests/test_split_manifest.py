"""The split manifest's encoder and decoder, exercised as the source they actually are.

`ENCODER_SRC` and `DECODER_SRC` are Python source strings, because the sandbox cannot import this
repo. Nothing else in the tree can lint them: ruff never looks inside a string literal, and the
node process must not `exec` them (CLAUDE.md's rule). So this module compiles them, which is what
makes them ONE implementation under test rather than a mirror that drifts. `test_the_decoder_runs_
in_the_real_sandbox` is not `fast` and exists so the in-process shortcut is never the only evidence
that this code runs where it has to run.

What is being defended, in order of how badly it fails when it breaks:

1. The complement rule. "A fold trains on the train rows it does not validate on" is true of
   `KFold` and `StratifiedKFold` and false of `TimeSeriesSplit`. The manifest records the rule and
   the decoder refuses any other value, so implementing `temporal` later cannot silently produce
   fold-training sets containing future rows.
2. The length check. Under the old explicit index lists, a truncated read was caught by
   `ArtifactPayload.truncated` and out-of-range ids were caught by `0 <= i < len(df)`. Under an
   assignment string every position is in range by construction, so a SHORT assignment would drop
   the tail of the frame from train and holdout and every fold at once, and report a plausible
   `n_train_rows`. That guard has to be deliberate now.
3. Round-trip fidelity against the real splitter, which is the only thing that proves the encoding
   is lossless rather than merely smaller.
"""

import json

import numpy as np
import pytest
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from ds_agents import split_manifest
from ds_agents.nodes.profiler import HOLDOUT_FRACTION, N_FOLDS

# Marked per class rather than for the module: the last test in this file launches a subprocess
# that imports pandas and scikit-learn, and `fast` is what the edit hook runs on every save.
fast = pytest.mark.fast


def _load(source: str, name: str):
    """Compile one of the snippet source constants and hand back the function it defines.

    `exec` here is not what CLAUDE.md forbids. The rule is about the node process, whose state
    holds `planted_leakage_columns` and whose job is to keep agent-authored code out; this is our
    own module constant in a test process, and compiling it is the only way to test the thing that
    actually ships rather than a copy of it.
    """
    namespace: dict = {}
    exec(compile(source, f"{name}.py", "exec"), namespace)  # noqa: S102
    return namespace[name]


encode_split = _load(split_manifest.ENCODER_SRC, "encode_split")
decode_split = _load(split_manifest.DECODER_SRC, "decode_split")


def _real_split(n_rows: int = 500, *, seed: int = 7, stratified: bool = True):
    """A split drawn by the same two sklearn objects `SPLIT_SNIPPET` uses.

    Not a hand-built partition: the point of the round-trip below is that the encoding survives
    what the splitter actually produces, including the fold order it produces it in.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n_rows) > 0.65).astype(int)
    rows = np.arange(n_rows)
    train_rows, holdout_rows = train_test_split(
        rows, test_size=HOLDOUT_FRACTION, random_state=seed, stratify=y if stratified else None
    )
    splitter = (
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        if stratified
        else KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    )
    folds = [
        (train_rows[tr].tolist(), train_rows[va].tolist())
        for tr, va in splitter.split(train_rows, y[train_rows] if stratified else None)
    ]
    return n_rows, sorted(int(i) for i in holdout_rows), folds


def _encode(n_rows, holdout, folds, **overrides):
    kwargs = {
        "n_rows": n_rows,
        "holdout": holdout,
        "fold_valid": [sorted(va) for _tr, va in folds],
        "strategy": "stratified",
        "seed": 7,
        "target": "y",
        "holdout_fraction": HOLDOUT_FRACTION,
    }
    kwargs.update(overrides)
    return encode_split(**kwargs)


@fast
class TestRoundTrip:
    @pytest.mark.parametrize("stratified", [True, False])
    def test_every_partition_the_splitter_drew_comes_back(self, stratified):
        """The whole claim, against the real splitter: nothing is lost but order."""
        n_rows, holdout, folds = _real_split(stratified=stratified)
        decoded = decode_split(_encode(n_rows, holdout, folds), n_rows)

        assert decoded["holdout"] == holdout
        assert set(decoded["train"]) == {i for tr, _va in folds for i in tr} | {
            i for _tr, va in folds for i in va
        }
        assert len(decoded["folds"]) == N_FOLDS
        for decoded_fold, (train, valid) in zip(decoded["folds"], folds, strict=True):
            assert set(decoded_fold["valid"]) == set(valid)
            assert set(decoded_fold["train"]) == set(train)

    def test_the_only_thing_lost_is_order(self):
        """Named so nobody later reads the round-trip above as lossless in the strict sense.

        `SPLIT_SNIPPET` sorts the fold lists before encoding, in its own commit, for exactly this
        reason. If that sort is ever removed this test is where the loss shows up.
        """
        n_rows, holdout, folds = _real_split()
        decoded = decode_split(_encode(n_rows, holdout, folds), n_rows)

        unsorted = [va for _tr, va in folds if va != sorted(va)]
        assert unsorted, "the splitter no longer returns fold membership in permutation order"
        for fold in decoded["folds"]:
            assert fold["valid"] == sorted(fold["valid"])
            assert fold["train"] == sorted(fold["train"])

    def test_the_manifest_fits_and_is_one_character_per_row(self):
        n_rows, holdout, folds = _real_split()
        manifest = _encode(n_rows, holdout, folds)

        assert len(manifest["assignment"]) == n_rows
        assert set(manifest["assignment"]) <= set("h01234")
        # The header is what separates the two numbers, and it must stay small enough that the
        # size projection in tests/test_split_manifest_size.py is dominated by the row count.
        assert len(json.dumps(manifest)) - n_rows < 500

    def test_the_node_process_fixture_helper_agrees_with_the_encoder(self):
        """`manifest_from` exists so a unit-test fixture is readable. This is what keeps it a
        convenience rather than a second answer to which rows were trained on."""
        n_rows, holdout, folds = _real_split()

        assert split_manifest.manifest_from(
            n_rows=n_rows,
            holdout=holdout,
            fold_valid=[sorted(va) for _tr, va in folds],
            strategy="stratified",
            seed=7,
            target="y",
            holdout_fraction=HOLDOUT_FRACTION,
        ) == _encode(n_rows, holdout, folds)


@fast
class TestTheSourceStringsAreBuiltFromOneNumber:
    """`MAX_FOLDS` used to be a Python constant plus two literals inside strings ruff never reads.

    Bumping it would have changed nothing about what ships to the sandbox and nothing would have
    said so. The two constants are rendered from `FOLD_DIGITS` and `MAX_FOLDS` at import; these
    tests are what makes that rendering load-bearing rather than decorative.
    """

    def test_no_placeholder_survives_rendering(self):
        for source in (split_manifest.ENCODER_SRC, split_manifest.DECODER_SRC):
            assert "__MAX_FOLDS__" not in source
            assert "__FOLD_DIGITS__" not in source

    def test_the_ceiling_the_encoder_enforces_is_the_module_constant(self):
        assert f"> {split_manifest.MAX_FOLDS}:" in split_manifest.ENCODER_SRC

    def test_the_alphabet_the_decoder_accepts_is_the_module_constant(self):
        assert repr(split_manifest.FOLD_DIGITS) in split_manifest.DECODER_SRC
        assert len(split_manifest.FOLD_DIGITS) == split_manifest.MAX_FOLDS, (
            "one character per fold, so the alphabet is the ceiling"
        )

    def test_the_ceiling_is_above_what_the_profiler_actually_asks_for(self):
        assert N_FOLDS <= split_manifest.MAX_FOLDS


@fast
class TestTheEncoderRefuses:
    def test_more_folds_than_the_alphabet_has_digits(self):
        """At 10 folds `str(k)` is two characters, the assignment runs longer than the frame, and
        every row after the first fold-10 row is mislabelled by a drifting offset. `N_FOLDS` is a
        module constant an ablation could plausibly touch, so this is asserted, not assumed."""
        with pytest.raises(ValueError, match="exceeds 10"):
            encode_split(
                n_rows=22,
                holdout=[],
                fold_valid=[[i, i + 11] for i in range(11)],
                strategy="stratified",
                seed=1,
                target="y",
                holdout_fraction=0.0,
            )

    def test_a_row_in_two_partitions_at_once(self):
        with pytest.raises(ValueError, match="two partitions"):
            encode_split(
                n_rows=4,
                holdout=[0],
                fold_valid=[[0, 1], [2, 3]],
                strategy="stratified",
                seed=1,
                target="y",
                holdout_fraction=0.25,
            )

    def test_a_row_in_no_partition(self):
        """There is no character for a discarded row, and that is the design: a strategy that
        drops rows must be unrepresentable rather than silently misrepresented."""
        with pytest.raises(ValueError, match="no partition"):
            encode_split(
                n_rows=5,
                holdout=[0],
                fold_valid=[[1], [2], [3]],
                strategy="stratified",
                seed=1,
                target="y",
                holdout_fraction=0.2,
            )


@fast
class TestTheDecoderRefuses:
    def test_a_fold_train_rule_it_does_not_implement(self):
        """The one that matters most. `TaskSpec` already declares `temporal`, where fold-train is a
        prefix rather than a complement. Without this the encoding stays perfectly writable and the
        decoder hands back fold-training sets containing future rows, with nothing raising."""
        n_rows, holdout, folds = _real_split()
        manifest = _encode(n_rows, holdout, folds) | {"fold_train": "prefix"}

        with pytest.raises(ValueError, match="complement"):
            decode_split(manifest, n_rows)

    def test_an_encoding_it_does_not_know(self):
        n_rows, holdout, folds = _real_split()
        manifest = _encode(n_rows, holdout, folds) | {"encoding": "explicit-lists-v0"}

        with pytest.raises(ValueError, match="assignment-v1"):
            decode_split(manifest, n_rows)

    def test_an_assignment_shorter_than_the_frame_it_is_decoded_against(self):
        """The guard that replaces `truncated`. A short assignment drops the tail of the frame from
        train AND holdout AND every fold, and reports a plausible `n_train_rows` while doing it."""
        n_rows, holdout, folds = _real_split()
        manifest = _encode(n_rows, holdout, folds)

        with pytest.raises(ValueError, match="frame"):
            decode_split(manifest, n_rows + 1)

    def test_an_assignment_that_disagrees_with_its_own_n_rows(self):
        n_rows, holdout, folds = _real_split()
        manifest = _encode(n_rows, holdout, folds)
        manifest["assignment"] = manifest["assignment"][:-1]

        with pytest.raises(ValueError, match="against n_rows"):
            decode_split(manifest, n_rows)

    def test_counts_that_disagree_with_what_decoded(self):
        """`counts` is the O(1) integrity check a reader does first. Written and never read, it
        would be decoration."""
        n_rows, holdout, folds = _real_split()
        manifest = _encode(n_rows, holdout, folds)
        manifest["counts"]["train"] += 1

        with pytest.raises(ValueError, match="counts"):
            decode_split(manifest, n_rows)

    def test_a_character_outside_the_alphabet(self):
        n_rows, holdout, folds = _real_split()
        manifest = _encode(n_rows, holdout, folds)
        manifest["assignment"] = "x" + manifest["assignment"][1:]

        with pytest.raises(ValueError, match="alphabet"):
            decode_split(manifest, n_rows)

    def test_a_manifest_with_no_assignment_at_all(self):
        """The old explicit-list form, and a hand-written `{}`, both land here. There is no
        dual-form fallback on purpose: a second decode path would be exercised only by fixtures,
        while the path production takes went untested."""
        with pytest.raises((KeyError, ValueError)):
            decode_split({}, 10)
        with pytest.raises((KeyError, ValueError)):
            decode_split({"train": [0, 1], "holdout": [2], "folds": []}, 3)


@fast
class TestShape:
    def test_a_fold_that_validates_on_nothing_still_appears(self):
        """The decoder indexes `range(n_folds)` rather than the characters it happens to see. A
        decoder that derived the fold list from `set(assignment)` would silently return four folds
        for a five-fold split and shrink the `n_folds` the modeler reports."""
        manifest = split_manifest.manifest_from(
            n_rows=6, holdout=[0], fold_valid=[[1, 2], [3, 4, 5], []]
        )
        decoded = decode_split(manifest, 6)

        assert len(decoded["folds"]) == 3
        assert decoded["folds"][2] == {"train": [1, 2, 3, 4, 5], "valid": []}

    def test_a_fold_trains_on_every_train_row_it_does_not_validate_on(self):
        n_rows, holdout, folds = _real_split()
        decoded = decode_split(_encode(n_rows, holdout, folds), n_rows)
        train = set(decoded["train"])

        for fold in decoded["folds"]:
            assert set(fold["train"]) == train - set(fold["valid"])
            assert not set(fold["valid"]) & set(decoded["holdout"])


def test_the_decoder_runs_in_the_real_sandbox(tmp_path):
    """Not `fast`: it launches a subprocess. The in-process compile above must never be the only
    evidence that this source runs where it is actually shipped -- inside four snippets, in a
    sandbox with a pruned `sys.path` and no access to this repo."""
    from ds_agents.tools.local import LocalTools

    n_rows, holdout, folds = _real_split(n_rows=120)
    manifest = _encode(n_rows, holdout, folds)
    payload = json.dumps(manifest)
    body = 'print(json.dumps({"train": len(SPLIT["train"]), "folds": len(SPLIT["folds"])}))\n'
    code = (
        f"{split_manifest.DECODER_SRC}\nimport json\n"
        f"SPLIT = decode_split(json.loads({payload!r}), {n_rows})\n" + body
    )

    csv = tmp_path / "d.csv"
    csv.write_text("a\n1\n")
    tools = LocalTools(tmp_path / "run", dataset_path=csv, dataset_id="d")
    result = tools.run_python(code, timeout_s=60)

    assert result.ok, result.stderr
    assert json.loads(result.stdout) == {"train": n_rows - len(holdout), "folds": N_FOLDS}
