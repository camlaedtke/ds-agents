"""The split manifest's on-disk form, and the only implementation of it.

This module's docstring IS the artifact's spec. The profiler writes one split manifest per run and
four snippets read it back -- `feature_eng`, `modeler`, and the grader's two bodies in `rescore.py`
-- so the representation is not private to the artifact store, and a second implementation of the
rule would be a second answer to "which rows were trained on".

## The form

```json
{"version": 1, "encoding": "assignment-v1", "fold_train": "complement",
 "strategy": "stratified", "seed": 20260822, "target": "churned",
 "n_rows": 800, "n_folds": 5, "holdout_fraction": 0.2,
 "assignment": "h0h31...", "counts": {"train": 640, "holdout": 160, "folds": [128, ...]},
 "assignment_sha256": "..."}
```

`assignment` is one character per row of the AGENT frame, indexed positionally: `"h"` for a
withheld-by-the-profiler holdout row, and `"0".."9"` for the fold that row *validates* in. Every
other partition is derived:

    train        = every position that is not "h"
    holdout      = every position that is "h"
    folds[k]     = {"valid": positions equal to str(k),
                    "train": positions that are neither "h" nor str(k)}

## Why one character per row

The previous form listed every index explicitly -- `train`, `holdout`, and five folds of
(train, valid), so roughly six times the agent row count in integers. `read_artifact` caps every
read at `DEFAULT_READ_BYTES`, and above ~36k rows those collided: four of the thirteen benchmark
datasets could not complete a run. This form is `n_rows` bytes plus ~380 of header: 78,831 B
measured on `higgs` against 2,690,410 B, which is 7.5% of the cap, with headroom to roughly a
million rows.

Measured alternatives, on a 5,000-row stratified 5-fold split: base64 of a byte array is 6,668 B,
1.33x WORSE than the 5,000 B digit string, because the digit string is already one printable byte
per row. Run-length as JSON pairs is 41,340 B, 8.3x worse -- mean run length 1.21, since a shuffled
fold assignment is incompressible by construction. Re-deriving the split from the seed inside each
snippet is smaller still and is refused twice over: the split stops being a recorded object, and it
would make the partition depend on the installed sklearn version, which `holdout._withhold_rows`
already refuses to do for the withheld carve.

## What `fold_train: "complement"` is doing here

"A fold's train rows are the train rows it does not validate on" is a property of `KFold` and
`StratifiedKFold`, not of a split. It is FALSE of `TimeSeriesSplit`, where fold-train is a prefix.
`TaskSpec` already declares `temporal` and `grouped`, and `profiler.SUPPORTED_SPLIT_STRATEGIES`
refuses them today precisely so a manifest never claims a partition that did not run. Without the
rule recorded IN the file, the day someone implements `temporal` this encoding stays perfectly
writable while the decoder hands back fold-trains containing future rows, and nothing raises. So
the rule is written down and the decoder refuses any other value.

## What is lost, irrecoverably

Fold ORDER. The splitter returns membership in permutation order; an assignment array can only
carry membership. `SPLIT_SNIPPET` sorts the fold lists in the same change that introduced this
encoding, so the loss is paid in one visible line rather than discovered later. It costs nothing
measurable here: `modeler.CANDIDATE_SPECS` holds `LogisticRegression`, `HistGradientBoosting` and
`Ridge`, none of which is order-sensitive, and the one bootstrap estimator in the repo -- the
baseline's RandomForest -- fits on `train`, which was already sorted.

## Why the guards below are not paranoia

Under explicit index lists, the consumers' `0 <= i < len(df)` bound was a real guard and
`truncated` was a real backstop. Under an assignment string every position is in range by
construction, so an assignment SHORTER than the frame -- a partial write, or the wrong frame
mounted -- would silently drop the tail from train AND holdout AND every fold, and report a
perfectly plausible `n_train_rows`. Shrinking the artifact removed an accidental guard, so
`decode_split` takes the frame length as a required argument and checks `counts` against what it
actually decoded.

## Why these are source strings

The sandbox cannot import this repo, so the encoder and decoder reach the snippets as source. They
are NOT concatenated into the snippet templates: the templates are `str.format` strings with
`{{`-escaped braces and this source is not escaped, so it is passed as a format ARGUMENT
(`{decoder}` / `{encoder}`), which `str.format` substitutes without re-scanning. Concatenating it
the way `rescore._SNIPPET_PRELUDE + _RESCORE_BODY` does would raise at format time.

Nothing here is `exec`'d in the node process; CLAUDE.md's rule stands.
`tests/test_split_manifest.py` compiles these constants in the test process, which is what makes
them one implementation under test rather than a mirror, and one non-`fast` test runs the same
source through the real sandbox so the in-process shortcut is never the only evidence.
"""

import hashlib

SPLIT_MANIFEST_VERSION = 1
ENCODING = "assignment-v1"

# The rule the decoder is allowed to apply. See the module docstring.
FOLD_TRAIN_RULE = "complement"

HOLDOUT_CHAR = "h"

# One character per fold, so the alphabet IS the ceiling: at eleven folds `str(k)` is two
# characters, the assignment runs longer than the frame, and every row after the first two-digit
# fold is mislabelled by a drifting offset with no exception anywhere. `N_FOLDS` is a module
# constant a plausible ablation touches, so the ceiling is asserted rather than assumed. Raising
# it means extending `FOLD_DIGITS`, which is the honest requirement -- and both source strings
# below are built from these two names, so there is one number rather than three drifting copies.
FOLD_DIGITS = "0123456789"
MAX_FOLDS = len(FOLD_DIGITS)


_ENCODER_TEMPLATE = '''
def encode_split(*, n_rows, holdout, fold_valid, strategy, seed, target, holdout_fraction):
    """Build the split manifest. The one implementation; see ds_agents/split_manifest.py."""
    import hashlib

    if len(fold_valid) > __MAX_FOLDS__:
        raise ValueError(
            "n_folds=%d exceeds __MAX_FOLDS__: str(k) would be two characters and every row after "
            "the first two-digit fold would be mislabelled by a drifting offset" % len(fold_valid)
        )
    slots = [None] * n_rows
    for i in holdout:
        i = int(i)
        if slots[i] is not None:
            raise ValueError("row %d is in two partitions at once" % i)
        slots[i] = "h"
    for k, valid in enumerate(fold_valid):
        for i in valid:
            i = int(i)
            if slots[i] is not None:
                raise ValueError("row %d is in two partitions at once" % i)
            slots[i] = str(k)
    missing = [i for i, c in enumerate(slots) if c is None]
    if missing:
        raise ValueError(
            "%d rows are in no partition (first: %d); the alphabet is 'h' plus one digit per "
            "fold and there is no character for a discarded row" % (len(missing), missing[0])
        )
    assignment = "".join(slots)
    counts = {
        "train": sum(1 for c in assignment if c != "h"),
        "holdout": sum(1 for c in assignment if c == "h"),
        "folds": [len(valid) for valid in fold_valid],
    }
    return {
        "version": 1,
        "encoding": "assignment-v1",
        "fold_train": "complement",
        "strategy": strategy,
        "seed": seed,
        "target": target,
        "n_rows": int(n_rows),
        "n_folds": len(fold_valid),
        "holdout_fraction": holdout_fraction,
        "assignment": assignment,
        "counts": counts,
        "assignment_sha256": hashlib.sha256(assignment.encode()).hexdigest(),
    }
'''


_DECODER_TEMPLATE = '''
def decode_split(manifest, n_rows):
    """Explicit row-id lists from an assignment manifest, checked against the frame it decodes on.

    `n_rows` is the length of the frame the caller actually read, and it is required rather than
    optional: under this encoding every position is in range by construction, so a short assignment
    would silently drop the tail from every partition. See ds_agents/split_manifest.py.
    """
    if manifest.get("encoding") != "assignment-v1":
        raise ValueError(
            "split manifest encoding is %r, not 'assignment-v1'" % (manifest.get("encoding"),)
        )
    if manifest.get("fold_train") != "complement":
        raise ValueError(
            "split manifest declares fold_train=%r; this decoder only knows 'complement', and "
            "applying it to any other rule would hand back fold-train rows the splitter never "
            "put there" % (manifest.get("fold_train"),)
        )
    assignment = manifest["assignment"]
    n_folds = int(manifest["n_folds"])
    if len(assignment) != int(manifest["n_rows"]):
        raise ValueError(
            "assignment is %d characters against n_rows=%d"
            % (len(assignment), int(manifest["n_rows"]))
        )
    if len(assignment) != int(n_rows):
        raise ValueError(
            "split manifest is for a %d-row frame but this frame has %d rows"
            % (len(assignment), int(n_rows))
        )
    digits = set(__FOLD_DIGITS__[:n_folds])
    allowed = digits | {"h"}
    bad = sorted(set(assignment) - allowed)
    if bad:
        raise ValueError("split assignment carries characters outside the alphabet: %r" % (bad,))

    train = []
    holdout = []
    # Indexed by fold, not derived from the characters that happen to appear: a fold with an empty
    # valid set is legal and must not vanish from the list.
    valid_by_fold = [[] for _ in range(n_folds)]
    for i, c in enumerate(assignment):
        if c == "h":
            holdout.append(i)
        else:
            train.append(i)
            valid_by_fold[int(c)].append(i)
    # One pass per fold over the assignment, not `i not in set(valid)` per element: the inner
    # membership test would be O(n_rows) per row on a 78k-row frame.
    folds = []
    for k in range(n_folds):
        key = str(k)
        folds.append(
            {
                "train": [i for i, c in enumerate(assignment) if c != "h" and c != key],
                "valid": list(valid_by_fold[k]),
            }
        )

    counts = manifest.get("counts")
    if counts is not None:
        decoded = {
            "train": len(train),
            "holdout": len(holdout),
            "folds": [len(f["valid"]) for f in folds],
        }
        if decoded != counts:
            raise ValueError(
                "split manifest counts %r disagree with what decoded: %r" % (counts, decoded)
            )
    if len(train) + len(holdout) != len(assignment):
        raise ValueError("train and holdout do not partition the frame")
    return {"train": train, "holdout": holdout, "folds": folds}
'''


def _render(template: str) -> str:
    """Substitute the alphabet and its ceiling into a snippet source constant.

    `.replace` on a distinctive token rather than `.format`: these strings are full of braces and
    are themselves passed as format ARGUMENTS to the snippet templates, so they must not be
    format strings of their own. This exists so `MAX_FOLDS` is one number instead of a Python
    constant plus two literals inside strings that ruff never reads and no bump would reach.
    """
    return template.replace("__FOLD_DIGITS__", repr(FOLD_DIGITS)).replace(
        "__MAX_FOLDS__", str(MAX_FOLDS)
    )


ENCODER_SRC = _render(_ENCODER_TEMPLATE)
DECODER_SRC = _render(_DECODER_TEMPLATE)


def manifest_from(
    *,
    n_rows: int,
    holdout: list[int],
    fold_valid: list[list[int]],
    strategy: str = "stratified",
    seed: int = 0,
    target: str = "y",
    holdout_fraction: float = 0.2,
) -> dict:
    """A manifest built in the node process, FOR TEST FIXTURES ONLY.

    This is not the implementation -- `ENCODER_SRC` is, and it is the only thing the profiler
    runs. This exists so a unit-test fixture can say which rows are in which fold in the same
    vocabulary the old explicit-list fixtures used, instead of a hand-typed 200-character string.
    `tests/test_split_manifest.py` pins that it agrees with `ENCODER_SRC` on a real split, which is
    what keeps it a convenience rather than a second answer.
    """
    if len(fold_valid) > MAX_FOLDS:
        raise ValueError(f"n_folds={len(fold_valid)} exceeds {MAX_FOLDS}")
    slots: list[str | None] = [None] * n_rows
    for i in holdout:
        slots[int(i)] = HOLDOUT_CHAR
    for k, valid in enumerate(fold_valid):
        for i in valid:
            slots[int(i)] = str(k)
    missing = [i for i, c in enumerate(slots) if c is None]
    if missing:
        raise ValueError(f"{len(missing)} rows are in no partition (first: {missing[0]})")
    assignment = "".join(c for c in slots if c is not None)
    return {
        "version": SPLIT_MANIFEST_VERSION,
        "encoding": ENCODING,
        "fold_train": FOLD_TRAIN_RULE,
        "strategy": strategy,
        "seed": seed,
        "target": target,
        "n_rows": int(n_rows),
        "n_folds": len(fold_valid),
        "holdout_fraction": holdout_fraction,
        "assignment": assignment,
        "counts": {
            "train": sum(1 for c in assignment if c != HOLDOUT_CHAR),
            "holdout": sum(1 for c in assignment if c == HOLDOUT_CHAR),
            "folds": [len(valid) for valid in fold_valid],
        },
        "assignment_sha256": hashlib.sha256(assignment.encode()).hexdigest(),
    }
