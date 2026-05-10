from __future__ import annotations

"""
MCP ExtensionLoader + ToolHandler.

MCPLoader connects to all configured MCP servers at session start,
discovers their tools via tools/list, and wires them into axor-core:

  1. Registers MCPToolHandler in CapabilityExecutor (execution path)
  2. Registers OpenAI-format schema in OpenRouterExecutor (model sees tool)
  3. Returns ExtensionBundle with:
       - ExtensionFragment with policy_overrides: extra_allowed_tools
         → PolicyComposer adds tools to allowed set for this session
       - ExtensionTool entries for ExtensionRegistry tracking

Tool names are namespaced: {server_name}__{mcp_tool_name}
e.g. github__create_issue, filesystem__read_file
"""

import logging
from typing import TYPE_CHECKING, Any

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.extension import (
    ExtensionBundle,
    ExtensionFragment,
    ExtensionLoader,
    ExtensionTool,
)

from axor_openrouter.mcp.client import MCPClient

if TYPE_CHECKING:
    from axor_openrouter.executor import OpenRouterExecutor

log = logging.getLogger("axor.mcp.loader")


class MCPToolHandler(ToolHandler):
    """Proxies a single MCP tool call to the MCP server."""

    def __init__(self, client: MCPClient, mcp_tool_name: str, namespaced_name: str) -> None:
        self._client       = client
        self._mcp_name     = mcp_tool_name   # name the MCP server knows
        self._namespaced   = namespaced_name  # name the model uses

    @property
    def name(self) -> str:
        return self._namespaced

    async def execute(self, args: dict[str, Any]) -> Any:
        return await self._client.call_tool(self._mcp_name, args)


class MCPLoader(ExtensionLoader):
    """
    Connects to MCP servers and registers their tools into the session.

    Needs references to both CapabilityExecutor (tool execution) and
    OpenRouterExecutor (tool schema → model prompt).  Both are created
    in make_session() before the session starts, so the references are
    stable by the time GovernedSession.start() calls load().
    """

    def __init__(
        self,
        clients: list[MCPClient],
        cap_executor: CapabilityExecutor,
        executor: "OpenRouterExecutor",
    ) -> None:
        self._clients      = clients
        self._cap_executor = cap_executor
        self._executor     = executor

    async def load(self) -> ExtensionBundle:
        fragments: list[ExtensionFragment] = []
        ext_tools: list[ExtensionTool]     = []

        for client in self._clients:
            try:
                await client.start()
                mcp_tools = await client.list_tools()
            except Exception as exc:
                log.warning("MCP server '%s' failed to start: %s", client.name, exc)
                continue

            if not mcp_tools:
                log.debug("MCP server '%s' returned no tools", client.name)
                continue

            tool_names: list[str] = []
            for t in mcp_tools:
                raw_name   = t["name"]
                namespaced = f"{client.name}__{raw_name}"
                tool_names.append(namespaced)

                # 1. Register execution handler
                self._cap_executor.register(
                    MCPToolHandler(client, raw_name, namespaced)
                )

                # 2. Register schema so model sees the tool
                input_schema = t.get("inputSchema") or {"type": "object", "properties": {}}
                self._executor.register_extension_schema({
                    "type": "function",
                    "function": {
                        "name":        namespaced,
                        "description": t.get("description", ""),
                        "parameters":  input_schema,
                    },
                })

                ext_tools.append(ExtensionTool(
                    name=namespaced,
                    description=t.get("description", ""),
                    parameters=input_schema,
                    source=f"mcp:{client.name}",
                ))

            # 3. Policy fragment: adds tool names to extra_allowed_tools
            tool_list_str = ", ".join(tool_names)
            fragment = ExtensionFragment(
                name=f"mcp:{client.name}",
                context_fragment=(
                    f"MCP server '{client.name}' is available with "
                    f"{len(tool_names)} tool(s): {tool_list_str}"
                ),
                required_tools=tuple(tool_names),
                policy_overrides={"extra_allowed_tools": tool_names},
                source=f"mcp:{client.name}",
            )
            fragments.append(fragment)
            log.info("MCP server '%s': %d tool(s) registered", client.name, len(tool_names))

        return ExtensionBundle(
            fragments=tuple(fragments),
            tools=tuple(ext_tools),
        )
