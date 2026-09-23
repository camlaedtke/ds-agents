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
(train, valid) -- roughly six times the agent row count in integers, large enough to collide with
`read_artifact`'s `DEFAULT_READ_BYTES` cap on the manifest's larger datasets. This form is `n_rows`
bytes plus a small header, since the digit string is already one printable byte per row.

Base64 of a byte array measured worse, since it does not beat one printable byte per row.
Run-length measured far worse, since a shuffled fold assignment is incompressible by construction.
Re-deriving the split from the seed inside each snippet is refused twice over: the split stops
being a recorded object, and it would make the partition depend on the installed sklearn version,
which `holdout._withhold_rows` already refuses to do for the withheld carve.

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
here: none of `modeler.CANDIDATE_SPECS`'s estimators are order-sensitive, and the baseline's
RandomForest fits on `train`, which was already sorted.

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

SPLIT_MANIFEST_VERSION = 1
ENCODING = "assignment-v1"

# The rule the decoder is allowed to apply. See the module docstring.
FOLD_TRAIN_RULE = "complement"

HOLDOUT_CHAR = "h"

# One character per fold, so the alphabet IS the ceiling: at eleven folds `str(k)` is two
# characters, the assignment runs longer than the frame, and every row after the first two-digit
# fold is mislabelled with no exception anywhere. The ceiling is asserted rather than assumed;
# raising it means extending `FOLD_DIGITS`, which both source strings below are built from.
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
