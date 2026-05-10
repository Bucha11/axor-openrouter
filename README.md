# axor-openrouter

OpenRouter adapter for [axor-core](https://github.com/Bucha11/axor-core) — run governed multi-agent sessions across 200+ models through a single API.

```bash
pip install axor-openrouter
```

---

## Features

- **Smart cascade routing** — automatically assigns cheaper models to deeper child agents; configurable via `~/.axor/config.toml`
- **Streaming** — SSE chunked responses with tool-call round-trip
- **Full tool suite** — read, write, edit, bash, search, glob, fetch, todo\_write, todo\_read
- **MCP server integration** — connect any stdio MCP server via config
- **Memory provider** — SQLite-backed persistent memory scoped to project
- **Provider preferences** — price caps, fallback control
- **Per-call cost ledger** — every API call recorded with model + token counts

---

## Quick start (Python API)

```python
import asyncio
from axor_openrouter import make_session

session = make_session(
    api_key="sk-or-...",
    model="anthropic/claude-sonnet-4-6",
)

result = asyncio.run(session.run("Write a Python URL shortener with FastAPI."))
print(result.output)
```

For interactive use from the terminal, use [axor-cli](https://github.com/Bucha11/axor-cli):

```bash
pip install axor-cli
axor openrouter "refactor auth module"
```

---

## Routing modes

Configure routing in `~/.axor/config.toml`:

### Smart (default)

Automatically selects model tier based on task complexity and call depth. No config needed.

```toml
[openrouter.routing]
mode = "smart"
prefer_free_at_depth = 3   # use free models at depth ≥ 3 (default: 3)
max_cost_in = 10.0         # reject models costing more than $10/1M input tokens
root_model = "anthropic/claude-sonnet-4-6"  # pin the root model
```

### Cascade

Explicit tier schedule — assign specific models per depth range.

```toml
[openrouter.routing]
mode = "cascade"

[[openrouter.routing.tiers]]
min_depth = 0
max_depth = 0
model     = "anthropic/claude-opus-4-7"

[[openrouter.routing.tiers]]
min_depth = 1
max_depth = 2
model     = "anthropic/claude-sonnet-4-6"

[[openrouter.routing.tiers]]
min_depth = 3
model     = "openai/gpt-4o-mini"
```

### Flat

Always use the same model regardless of depth.

```toml
[openrouter.routing]
mode = "flat"
```

---

## Default smart-tier schedule

| Depth | Default model |
|-------|--------------|
| 0 | `anthropic/claude-opus-4-7` |
| 1–2 | `anthropic/claude-sonnet-4-6` |
| 3–5 | `openai/gpt-4o-mini` |
| 6+ | `meta-llama/llama-3.3-70b-instruct:free` |

---

## MCP servers

Add any stdio MCP server in `~/.axor/config.toml`:

```toml
[[mcp.servers]]
name    = "filesystem"
command = "npx"
args    = ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]

[[mcp.servers]]
name    = "github"
command = "npx"
args    = ["-y", "@modelcontextprotocol/server-github"]

[mcp.servers.env]
GITHUB_PERSONAL_ACCESS_TOKEN = "ghp_..."
```

Tool names are namespaced as `server_name__tool_name`.

---

## make_session() reference

```python
from axor_openrouter import make_session

session = make_session(
    api_key="sk-or-...",              # OPENROUTER_API_KEY env var if omitted
    model="anthropic/claude-sonnet-4-6",
    tools=(                            # enabled tool names
        "read", "write", "edit", "bash",
        "search", "glob", "fetch",
    ),
    soft_token_limit=100_000,          # triggers auto-compact / budget signals
    system_prompt="You are ...",
    load_skills=True,                  # load .claude/skills/*.md
    load_plugins=True,                 # load .claude/plugins/
    memory_provider=my_provider,       # MemoryProvider implementation
    memory_namespace="my_project",
    mcp_servers=[...],                 # list of {name, command, args, env} dicts
    smart_cascade=True,                # smart tier routing
    tiers=[...],                       # explicit TierSpec list (overrides smart)
    prefer_free_at_depth=3,
    max_cost_in=10.0,
    thinking_budget=8000,              # extended thinking token budget
    resume=False,                      # inject last-session history
    telemetry=pipeline,
)
```

---

## Built-in tools

| Tool | Description |
|------|-------------|
| `read` | Read a file (respects `.claudeignore`) |
| `write` | Write/create a file |
| `edit` | Replace exact string in a file |
| `bash` | Execute a shell command (30 s timeout) |
| `search` | Grep content or find files by name; uses `rg` when available |
| `glob` | Find files matching a glob pattern |
| `fetch` | Fetch an HTTP/HTTPS URL (64 KB cap) |
| `todo_write` | Replace the session todo list |
| `todo_read` | Read the current session todo list |

All file operations respect `.claudeignore` in the working directory.

---

## Cost ledger

```python
result = asyncio.run(session.run(task))

for entry in session._executor.call_ledger():
    print(entry)
# {'model': 'anthropic/claude-sonnet-4-6', 'depth': 0, 'in_tokens': 42145, 'out_tokens': 9132}
# {'model': 'openai/gpt-4o-mini',          'depth': 1, 'in_tokens': 398,   'out_tokens': 307}
```

---

## Benchmark — SMART CASCADE vs FLAT

**Task:** 10-component Python REST API microservice built by parallel child agents.

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

  Savings                     $1.1991  (81.9%)
  Speed gain                  442.6s faster  (73%)
==================================================================
```

| | FLAT | CASCADE |
|---|---|---|
| Model (root) | claude-sonnet-4-6 | claude-sonnet-4-6 |
| Model (children) | claude-sonnet-4-6 | gpt-4o-mini |
| Total cost | **$1.4650** | **$0.2659** |
| Time | 606 s | 163 s |
| **Savings** | — | **$1.20 · 82% cheaper · 3.7× faster** |

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | Primary API key |
| `OPEN_KEY` | Fallback alias |

---

## License

MIT
