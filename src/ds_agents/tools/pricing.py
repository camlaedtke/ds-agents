"""Per-model token prices, and the one function that turns usage into dollars.

`cost_usd` is a published output of this project, not an implementation detail: the results'
cost-per-caught-leak number and the Haiku-vs-Sonnet reviewer ablation are both read straight off
`NodeEvent.cost_usd`. So this file has two rules.

1. An unknown model id **raises**. The tempting alternative -- price it at zero and move on --
   produces a results row that is quietly wrong in the flattering direction, and nothing
   downstream can tell a genuinely free run from an unpriced one. A crash is cheaper than a
   number nobody can trust.
2. Prices are written per million tokens exactly as published, and converted here. Pre-dividing
   them into per-token floats in the table would make them uncheckable against the price page.

Prices verified 2026-08-26 against the published Anthropic API rates. They are a snapshot: if a
committed results file needs to survive a price change, the fix is to record the rate alongside
the run, not to backfill this table.
"""

from dataclasses import dataclass

USD_PER_MILLION = 1_000_000


class UnknownModelError(LookupError):
    """No published price for this model id.

    Deliberately fatal. See the module docstring: an unpriced run that reports $0.00 is worse
    than a run that stops. `LookupError` rather than `KeyError` for two reasons: an incidental
    `except KeyError` upstream would swallow an error whose whole purpose is to be unswallowable,
    and `KeyError.__str__` reprs its argument, so the explanation would print wrapped in quotes
    with its inner quotes escaped.
    """


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens, as published.

    Cache tiers are carried even though Phase 1 sends no `cache_control`, because the profiler and
    reviewer prompts repeat a large fixed block per run and Phase 4 will want them. Costing a
    cached read at the full input rate would overstate the benchmark's cost by roughly the cache
    hit rate.
    """

    input_: float
    output: float
    cache_write: float
    cache_read: float


# Short names are what `RunConfig.default_model` and `--reviewer-model` carry, because a config
# field reading "haiku" survives a model refresh and one reading a dated id does not.
ALIASES: dict[str, str] = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
}

PRICES: dict[str, ModelPrice] = {
    # Cache write is the 5-minute tier (1.25x input); cache read is 0.1x input.
    "claude-haiku-4-5": ModelPrice(input_=1.00, output=5.00, cache_write=1.25, cache_read=0.10),
    "claude-sonnet-5": ModelPrice(input_=3.00, output=15.00, cache_write=3.75, cache_read=0.30),
    "claude-opus-5": ModelPrice(input_=5.00, output=25.00, cache_write=6.25, cache_read=0.50),
}


def resolve(name: str) -> str:
    """Turn a config short name into a model id. Full ids pass through unchanged."""
    return ALIASES.get(name, name)


def price_for(model: str) -> ModelPrice:
    resolved = resolve(model)
    try:
        return PRICES[resolved]
    except KeyError:
        raise UnknownModelError(
            f"no published price for model {model!r} (resolved to {resolved!r}). Add it to "
            f"PRICES with the rate from the price page rather than letting this run report "
            f"$0.00. Known: {', '.join(sorted(PRICES))}."
        ) from None


def cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Dollars for one API call.

    `input_tokens` from the Anthropic API already excludes cached tokens, so the four terms do not
    double-count.
    """
    p = price_for(model)
    return (
        input_tokens * p.input_
        + output_tokens * p.output
        + cache_write_tokens * p.cache_write
        + cache_read_tokens * p.cache_read
    ) / USD_PER_MILLION
