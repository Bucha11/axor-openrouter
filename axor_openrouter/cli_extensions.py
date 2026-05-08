"""Extra argparse arguments contributed by axor-openrouter to axor-cli.

axor-cli calls `register_args(parser)` during startup when the openrouter
adapter is selected.  The parsed values are forwarded to `make_session()`.
"""
from __future__ import annotations

import argparse


def register_args(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group("OpenRouter options")
    grp.add_argument(
        "--sort",
        default="quality",
        choices=["quality", "throughput", "latency", "price"],
        help="Provider sort strategy (default: quality).",
    )
    grp.add_argument(
        "--max-prompt",
        dest="max_prompt_price",
        type=float,
        default=None,
        metavar="PRICE",
        help="Maximum prompt price per million tokens (USD).",
    )
    grp.add_argument(
        "--fallback",
        dest="fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow provider fallbacks (default: enabled).",
    )
    grp.add_argument(
        "--tier-config",
        dest="tier_config",
        default=None,
        metavar="PATH",
        help="Path to TOML/YAML cascade tier config (default: built-in).",
    )
    grp.add_argument(
        "--byok",
        dest="byok",
        default=None,
        metavar="PROVIDER:KEY",
        help="Bring-your-own-key passthrough, e.g. 'openai:sk-...'.",
    )
    grp.add_argument(
        "--no-cache",
        dest="enable_cache",
        action="store_false",
        default=True,
        help="Disable prompt caching.",
    )
    grp.add_argument(
        "--response-cache",
        dest="response_cache",
        action="store_true",
        default=False,
        help="Enable deterministic response caching.",
    )


def byok_from_string(value: str | None):
    """Parse 'provider:key' into a BYOKConfig, or return None."""
    if not value:
        return None
    from axor_openrouter.routing.byok import BYOKConfig
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"--byok must be in 'provider:key' format, got: {value!r}")
    return BYOKConfig(provider=parts[0], api_key=parts[1])
