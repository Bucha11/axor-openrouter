# axor-openrouter

[![CI](https://github.com/Bucha11/axor-openrouter/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Bucha11/axor-openrouter/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/axor-openrouter?cacheSeconds=300)](https://pypi.org/project/axor-openrouter/)
[![Python](https://img.shields.io/pypi/pyversions/axor-openrouter?cacheSeconds=300)](https://pypi.org/project/axor-openrouter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OpenRouter backend adapter for [axor-core](https://github.com/Bucha11/axor-core).**

Runs any OpenRouter-hosted model as a governed agent under axor-core's governance
kernel — streaming tool use, cost/rate tracking, and a full audit trail — while
keeping all governance logic in the kernel, never in the adapter.

---

## Installation

```bash
pip install axor-openrouter
```

Requires an [OpenRouter API key](https://openrouter.ai/keys).

Optional behavioral-security observers (axor-probe / axor-sentinel):

```bash
pip install "axor-openrouter[security]"
```

---

## Quick start

```python
import asyncio
from axor_openrouter import make_session

session = make_session(
    api_key="sk-or-...",
    task="explain this code",
    context_text="def foo(): ...",
)

async def main():
    async for event in session.stream():
        print(event)

asyncio.run(main())
```

Enable the optional observers (requires the `[security]` extra):

```python
session = make_session(
    api_key="sk-or-...",
    task="...",
    enable_sentinel=True,        # cross-session reputation (axor-sentinel)
    probe_pipeline=my_pipeline,  # behavioral drift probes (axor-probe)
)
# drain probe-flagged sessions for the sentinel audit cycle:
sessions = session.axor_security.drain_sentinel()
```

---

## What it does

`axor-openrouter` is a thin **adapter** (Layer 1–4 integration): it speaks the
OpenRouter HTTP/streaming API and presents the model to axor-core as a governed
executor. It does **not** contain policy, capability, or intent logic — those
live in axor-core, and CI enforces this separation.

- streaming tool-use interception via axor-core's `ToolResultBus`
- per-model cost rates fetched from OpenRouter (`rates/catalog`)
- model cascade / routing (`cascade`)
- optional `[security]` wiring of axor-probe and axor-sentinel observers

---

## License

MIT.
