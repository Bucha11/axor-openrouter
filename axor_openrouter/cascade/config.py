from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from axor_openrouter.cascade.tiers import TierSpec, TierMapper

# Config file searched in order
_CONFIG_NAMES = ["axor.openrouter.toml", ".axor/openrouter.toml", "axor.openrouter.yaml"]


def load_cascade_config(config_path: str | None = None) -> TierMapper:
    """
    Load TierMapper from a TOML/YAML config file.

    Searches for config in:
    1. Explicit config_path argument
    2. AXOR_OPENROUTER_CONFIG env var
    3. Current directory (axor.openrouter.toml, .axor/openrouter.toml)

    If no config is found, returns TierMapper with DEFAULT_TIERS.

    TOML format example:

        [cascade]
        0       = "anthropic/claude-opus-4-7"
        "1-2"   = "anthropic/claude-sonnet-4-6"
        "3-5"   = "openai/gpt-4o-mini"
        "6+"    = "meta-llama/llama-3.3-70b-instruct"
    """
    path = _find_config(config_path)
    if path is None:
        from axor_openrouter.cascade.tiers import DEFAULT_TIERS
        return TierMapper(DEFAULT_TIERS)

    try:
        data = _load_file(path)
        cascade_section = data.get("cascade", {})
        tiers = _parse_cascade_section(cascade_section)
        return TierMapper(tiers)
    except Exception as exc:
        import logging
        logging.getLogger("axor.openrouter.cascade").warning(
            "Failed to load cascade config from %s: %s. Using defaults.", path, exc
        )
        from axor_openrouter.cascade.tiers import DEFAULT_TIERS
        return TierMapper(DEFAULT_TIERS)


def _find_config(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    env = os.environ.get("AXOR_OPENROUTER_CONFIG")
    if env:
        p = Path(env)
        return p if p.exists() else None
    for name in _CONFIG_NAMES:
        p = Path(name)
        if p.exists():
            return p
    return None


def _load_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in (".toml",):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        with open(path, "rb") as f:
            return tomllib.load(f)
    elif suffix in (".yaml", ".yml"):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    raise ValueError(f"Unsupported config format: {suffix}")


def _parse_cascade_section(section: dict[str, Any]) -> list[TierSpec]:
    """
    Parse cascade section into TierSpec list.

    Supported key formats:
        0        → depth == 0
        "1-2"   → min_depth=1, max_depth=2
        "6+"    → min_depth=6, max_depth=None
    """
    specs: list[TierSpec] = []
    for idx, (key, model) in enumerate(section.items()):
        if not isinstance(model, str):
            continue
        key_str = str(key).strip()
        if "-" in key_str:
            parts = key_str.split("-", 1)
            min_d, max_d = int(parts[0]), int(parts[1])
        elif key_str.endswith("+"):
            min_d = int(key_str[:-1])
            max_d = None
        else:
            min_d = max_d = int(key_str)
        specs.append(TierSpec(
            min_depth=min_d,
            max_depth=max_d,
            model=model,
            tier_index=idx,
        ))
    return specs if specs else []
