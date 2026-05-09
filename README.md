# axor-openrouter

OpenRouter adapter for [axor-core](https://github.com/Bucha11/axor-core) — run governed multi-agent sessions across 200+ models through a single API.

```
pip install axor-openrouter
```

## Features

- **Cascade tier routing** — automatically assign cheaper models to child agents (depth ≥ 1) while keeping powerful models at the root
- **Per-call cost ledger** — every API call is recorded with model + token counts for accurate attribution
- **Federated policy support** — `spawn_child` enabled out-of-the-box with `presets.get("federated")`
- **SSE streaming** — chunked responses with tool-call round-trip support
- **Provider preferences** — price caps, fallback control

## Quick start

```python
import asyncio
from axor_openrouter import make_session
from axor_core.policy import presets

session = make_session(
    api_key="sk-or-...",
    model="anthropic/claude-sonnet-4-6",
)

result = asyncio.run(session.run(
    "Write a Python URL shortener with FastAPI.",
    policy=presets.get("federated"),
))
print(result.output)
```

## Cascade model routing

Cascade assigns expensive models to the orchestrator (depth 0) and cheap models to leaf workers (depth ≥ 1). Savings compound with every spawned subtask.

```python
from axor_openrouter import make_session
from axor_openrouter.cascade.tiers import TierSpec, TierMapper
from axor_core.policy import presets

CASCADE_TIERS = [
    TierSpec(min_depth=0, max_depth=0,    model="anthropic/claude-sonnet-4-6", tier_index=0),
    TierSpec(min_depth=1, max_depth=None, model="openai/gpt-4o-mini",          tier_index=1),
]

session = make_session(api_key="sk-or-...", model="anthropic/claude-sonnet-4-6")
session._executor._tier_mapper = TierMapper(CASCADE_TIERS)

result = asyncio.run(session.run(task, policy=presets.get("federated")))
```

## Benchmark — CASCADE vs FLAT

**Task:** 10-component Python REST API microservice built by parallel child agents
(config · models · database · auth · 4× routes · 2× tests · main)

**Pricing:** every API call priced at the model that handled it (via `executor.call_ledger()`).

```
==================================================================
  BENCHMARK RESULTS  —  10-component Python microservice
==================================================================
  Metric                           FLAT    CASCADE        Δ
------------------------------------------------------------------
  Input tokens                  273,665     46,127 -227,538
  Output tokens                  42,931     12,203  -30,728
  API calls                          47         14      -33
  Time                            606.0s     163.4s  -442.6s

  Cost breakdown
    claude-sonnet-4-6        $1.4650      $0.2634
    gpt-4o-mini              $0.0000      $0.0024
    ─────────────────────────────────────────────────────────
    TOTAL                    $1.4650      $0.2659

  💰 Savings                  $1.1991  (81.9%)
  ⚡ Speed gain               442.6s faster  (73%)

  Token volume (input)
    FLAT    [██████████████████████████████] 273,665
    CASCADE [█████░░░░░░░░░░░░░░░░░░░░░░░░░]  46,127

  Token volume (output)
    FLAT    [██████████████████████████████] 42,931
    CASCADE [█████████░░░░░░░░░░░░░░░░░░░░░] 12,203

  Time
    FLAT    [██████████████████████████████] 606.0s
    CASCADE [████████░░░░░░░░░░░░░░░░░░░░░░] 163.4s
==================================================================
```

| | FLAT | CASCADE |
|---|---|---|
| Model (root) | claude-sonnet-4-6 | claude-sonnet-4-6 |
| Model (children) | claude-sonnet-4-6 | gpt-4o-mini |
| Input price | $3.00 / 1M | root $3.00, children $0.15 |
| Output price | $15.00 / 1M | root $15.00, children $0.60 |
| API calls | 47 | 14 |
| Input tokens | 273,665 | 46,127 |
| Output tokens | 42,931 | 12,203 |
| Total cost | **$1.4650** | **$0.2659** |
| Time | 606 s | 163 s |
| **Savings** | — | **$1.20 · 82% cheaper · 3.7× faster** |

Sonnet is 20× more expensive per input token than gpt-4o-mini. With 10 child workers the savings are dramatic: 82% cost reduction and 73% wall-clock speedup.

## Environment variables

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | Primary API key |
| `OPEN_KEY` | Fallback alias (checked if primary not set) |

## Default tier schedule

| Depth | Model |
|---|---|
| 0 | `anthropic/claude-opus-4-7` |
| 1–2 | `anthropic/claude-sonnet-4-6` |
| 3–5 | `openai/gpt-4o-mini` |
| 6+ | `meta-llama/llama-3.3-70b-instruct` |

Override with a custom `TierMapper` as shown above.

## Cost ledger

```python
# After session.run() completes:
for entry in session._executor.call_ledger():
    print(entry)
# {'model': 'anthropic/claude-sonnet-4-6', 'depth': 0, 'in_tokens': 42145, 'out_tokens': 9132}
# {'model': 'openai/gpt-4o-mini', 'depth': 1, 'in_tokens': 398, 'out_tokens': 307}
# ...
```

## License

MIT
