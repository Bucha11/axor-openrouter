# axor-openrouter

[![PyPI](https://img.shields.io/pypi/v/axor-openrouter)](https://pypi.org/project/axor-openrouter/)
[![Python](https://img.shields.io/pypi/pyversions/axor-openrouter)](https://pypi.org/project/axor-openrouter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**OpenRouter backend for axor-core — 200+ models, one governed interface.**

Route any task to the right model at the right cost. Automatically. Under governance.

---

## Why axor-openrouter

Running agents on a single model is a cost trap: you pay frontier prices for work that doesn't need a frontier model. But wiring up fallbacks, model selection, budget controls, and provider routing by hand is boilerplate you shouldn't write.

`axor-openrouter` solves both. It connects the [axor-core](https://github.com/Bucha11/axor-core) governance kernel to OpenRouter's full model catalog, and ships with:

- **Tier Cascade** — best model at the root, cheaper models for child nodes, automatically
- **Adaptive Routing** — shifts tiers down under budget pressure, back up when headroom returns
- **Fallback Chains** — per-tier model fallbacks, OpenRouter tries them in order
- **Provider Preferences** — sort by quality / throughput / latency / price, set a price ceiling
- **Prompt Cache Breakpoints** — maximise cache hit rate with intelligent breakpoint placement
- **BYOK** — route to a specific provider with your own API key
- **Full governance** — every model call runs inside an axor `GovernedNode` with policy, intent loop, budget tracking, and trace

---

## Installation

```bash
pip install axor-openrouter
```

Requires `axor-core` (installed automatically) and Python 3.11+.

---

## Quick Start

```python
import asyncio
from axor_openrouter import make_session

async def main():
    session = make_session(
        api_key="sk-or-...",       # your OpenRouter key
        task="refactor the auth module",
        context_text=open("auth.py").read(),
    )
    result = await session.run()
    print(result.output)

asyncio.run(main())
```

That's it. The session selects the model, manages context, tracks tokens, and returns a governed result.

---

## Tier Cascade — Pay for What You Need

In a federated agent tree, root nodes need the best model. Deep child nodes doing narrow sub-tasks don't. `axor-openrouter` maps execution depth to a model tier automatically:

```
depth 0          anthropic/claude-opus-4-7        ← orchestrator, full reasoning
depth 1–2        anthropic/claude-sonnet-4-6      ← mid-tier agents
depth 3–5        openai/gpt-4o-mini               ← narrow sub-tasks
depth 6+         meta-llama/llama-3.3-70b         ← leaf workers
```

No code changes needed. Governance tracks depth. The right model gets assigned.

### Custom Cascade

Override tiers with a config file — TOML or YAML:

```toml
# axor.openrouter.toml
[cascade]
0     = "anthropic/claude-opus-4-7"
"1-2" = "openai/gpt-4o"
"3+"  = "openai/gpt-4o-mini"
```

```python
session = make_session(
    api_key="sk-or-...",
    task="...",
    tier_config="axor.openrouter.toml",
)
```

Or set `AXOR_OPENROUTER_CONFIG` in the environment — the config is picked up automatically.

---

## Adaptive Routing — Automatic Cost Optimisation

When the `axor-core` budget engine signals pressure (≥80% of token budget), the `AdaptiveRouter` shifts all subsequent calls one tier cheaper. When headroom returns, it shifts back.

```
token spend at 40%  →  nominal tier (claude-opus-4-7 at depth 0)
token spend at 80%  →  shift –1    (claude-sonnet-4-6 at depth 0)
token spend at 95%  →  hard stop   (governance cancels via CancelToken)
```

No polling. No manual tuning. Budget thresholds from axor-core drive routing decisions in real time.

---

## Fallback Chains — Stay Available Under Load

Every tier has an automatic fallback chain. If the primary model is unavailable or errors, OpenRouter tries the next model in the chain seamlessly:

```
tier 0  →  claude-opus-4-7  →  claude-sonnet-4-6
tier 1  →  claude-sonnet-4-6  →  gpt-4o  →  gpt-4o-mini
tier 2  →  gpt-4o  →  gpt-4o-mini  →  gemini-flash-1.5
tier 3  →  gpt-4o-mini  →  gemini-flash-1.5  →  llama-3.3-70b
```

Your agent keeps running even when a provider has an outage.

---

## Provider Preferences — Fine-Grained Routing Control

```python
session = make_session(
    api_key="sk-or-...",
    task="...",
    sort="throughput",          # "quality" | "throughput" | "latency" | "price"
    max_prompt_price=5.0,       # $/M tokens ceiling — skip expensive providers
    fallback=True,              # allow OpenRouter to fall through on errors
)
```

OpenRouter sorts and filters providers for you based on your declared preferences. Set a price ceiling to never pay more than you expect.

---

## Bring Your Own Key (BYOK)

Route to a specific provider with your own API key for direct billing or higher rate limits:

```python
from axor_openrouter.routing.byok import BYOKConfig

session = make_session(
    api_key="sk-or-...",
    task="...",
    byok=BYOKConfig(provider="Anthropic", api_key="sk-ant-..."),
)
```

---

## Prompt Caching — Reduce Repeat Costs

`axor-openrouter` inserts cache breakpoints at the boundaries of stable context sections — system prompt, pinned fragments, prior tool results — so repeated turns reuse cached tokens instead of re-sending them.

Enable caching (on by default):

```python
session = make_session(
    api_key="sk-or-...",
    task="...",
    enable_cache=True,      # prompt cache breakpoints (default: True)
    response_cache=False,   # deterministic response cache (default: False)
)
```

The `CacheHealthMonitor` tracks cache hit rate across tool calls and surfaces it in the session trace.

---

## Full Governance Integration

Every call is a governed execution — not a raw API call:

```
task input
  → TaskAnalyzer     → TaskSignal (complexity × nature)
  → PolicySelector   → ExecutionPolicy
  → ContextManager   → shaped context, compressed, cached
  → EnvelopeBuilder  → ExecutionEnvelope
  → OpenRouterExecutor.stream()     ← this is where axor-openrouter lives
      → TierMapper.resolve(depth)   → model selection
      → AdaptiveRouter              → tier shift from budget signal
      → transport.stream()          → OpenRouter API (SSE)
      → IntentLoop                  → tool calls intercepted by governance
      → ToolResultBus               → results injected back into conversation
  → ExportFilter     → governed ExecutionResult
  → TraceCollector   → model, tier, cache stats, usage recorded
```

The executor never bypasses governance. Tool calls, model selection, and cancellation all flow through axor-core's intent loop.

```python
result = await session.run("refactor auth module")

print(result.output)
print(result.token_usage.total)            # full token accounting
print(result.metadata["policy"])           # which policy was selected

for trace in session.all_traces():
    for event in trace.events:
        if event.kind.value == "tokens_spent":
            print(event.payload)           # input/output/cache breakdown
```

---

## Federated Agents — Correct Cost at Every Depth

When axor-core spawns child nodes, they automatically receive a cheaper tier. The total spend is accounted across the full spawn tree:

```python
result = await session.run(
    "analyze security, performance, and maintainability — spawn subtasks",
    policy=presets.get("federated"),
)

# parent tokens only
print(result.token_usage.total)

# parent + all children + grandchildren
print(session.total_tokens_spent())
```

Root pays frontier prices. Children and grandchildren pay commodity prices. Automatically.

---

## API Reference

### `make_session()`

```python
def make_session(
    *,
    api_key: str,                     # OpenRouter API key (sk-or-...)
    task: str,                        # task to execute
    context_text: str = "",           # initial context
    depth: int = 0,                   # starting node depth (0 = root)
    parent_node_id: str | None = None,
    cache_hints: dict | None = None,
    deterministic: bool = False,      # temperature=0 + response cache
    # Routing
    tier_config: str | None = None,   # path to axor.openrouter.toml
    sort: str = "quality",            # "quality" | "throughput" | "latency" | "price"
    max_prompt_price: float | None = None,  # $/M tokens ceiling
    fallback: bool = True,            # allow OpenRouter model fallbacks
    byok: BYOKConfig | None = None,   # bring your own provider key
    # Cache
    enable_cache: bool = True,        # prompt cache breakpoints
    response_cache: bool = False,     # deterministic response cache
    # Policy
    policy: ExecutionPolicy | None = None,  # override axor policy
) -> GovernedSession
```

### `OpenRouterExecutor`

Low-level executor — use `make_session()` for the full governed setup. Direct instantiation is for advanced adapter composition:

```python
from axor_openrouter import OpenRouterExecutor
from axor_openrouter.cascade.tiers import TierMapper, DEFAULT_TIERS
from axor_openrouter.routing.provider_prefs import ProviderPrefs
from axor_openrouter.transport import OpenRouterTransport

executor = OpenRouterExecutor(
    api_key="sk-or-...",
    model="anthropic/claude-opus-4-7",
    transport=OpenRouterTransport(),
    tier_mapper=TierMapper(DEFAULT_TIERS),
    provider_prefs=ProviderPrefs(sort="throughput"),
)
```

---

## Architecture

```
axor_openrouter/
├── __init__.py          make_session() — one-line entry point
├── executor.py          OpenRouterExecutor — Invokable, ToolResultBus loop
├── transport.py         HTTP/SSE transport, StreamAccumulator
├── envelope_codec.py    ExecutionEnvelope → OpenRouter message format
├── tools.py             Capability-aware tool schema builder
├── usage_codec.py       OpenRouter usage → axor TokenUsage
├── bus.py               ToolResultBus — async tool result injection
├── cascade/
│   ├── tiers.py         TierSpec, TierMapper, DEFAULT_TIERS
│   └── config.py        TOML/YAML cascade config loader
├── routing/
│   ├── provider_prefs.py  ProviderPrefs — sort, price ceiling, BYOK
│   ├── fallbacks.py       DEFAULT_FALLBACKS per tier
│   └── byok.py            BYOKConfig — provider + key passthrough
├── governance/
│   ├── adaptive_router.py   AdaptiveRouter — budget-driven tier shift
│   ├── budget_subscriber.py BudgetSubscriber — listens to axor budget events
│   └── cache_health.py      CacheHealthMonitor — cache hit rate tracking
├── caching/
│   ├── breakpoints.py   Prompt cache breakpoint placement
│   ├── ttl_chooser.py   TTL strategy per context kind
│   └── response_cache.py  Deterministic response cache
└── rates/
    ├── catalog.py       Built-in rate catalog ($/M tokens per model)
    └── refresher.py     Background rate catalog refresh
```

---

## Requirements

- Python 3.11+
- `axor-core` (installed automatically)
- `httpx` for async HTTP transport

---

## Related Packages

| Package | Purpose |
|---|---|
| [`axor-core`](https://github.com/Bucha11/axor-core) | Governance kernel — required |
| `axor-claude` | Anthropic Claude adapter |
| `axor-memory-sqlite` | Cross-session memory |
| `axor-classifier-simple` | ML task classifier + anomaly detection |
| [`axor-cli`](https://github.com/Bucha11/axor-cli) | CLI runner for governed sessions |

---

## License

MIT
