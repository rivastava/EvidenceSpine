"""EvidenceSpine MCP server.

The ``mcp`` package is an optional dependency (extra ``[mcp]``), so it is
imported lazily inside functions only; importing this module never requires it.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from evidencespine import __version__
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings

_SERVER_DESCRIPTION = (
    "Evidence-bound conversation memory for coding agents: ingest structured "
    "events, build bounded context briefs, emit/import evidence-bound handoff "
    "packets, and query derived agent-state control views."
)


def _build_runtime(
    base_dir: Optional[str],
    storage_format: Optional[str],
    runtime: Optional[AgentMemoryRuntime],
) -> AgentMemoryRuntime:
    if runtime is not None:
        return runtime
    settings = EvidenceSpineSettings.from_env(base_dir=base_dir, storage_format=storage_format)
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def create_server(
    *,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    runtime: Optional[AgentMemoryRuntime] = None,
) -> Any:
    """Create an EvidenceSpine MCPServer instance.

    ``runtime`` may be injected (tests/embedding); otherwise one is built lazily
    on first request from ``base_dir`` / ``storage_format`` (or env defaults).
    """
    from mcp.server.mcpserver import MCPServer

    from evidencespine.mcp_server.prompts import register_prompts
    from evidencespine.mcp_server.resources import register_resources
    from evidencespine.mcp_server.tools import register_tools

    holder: dict[str, Any] = {"runtime": None}
    lock = threading.Lock()

    def get_runtime() -> AgentMemoryRuntime:
        if holder["runtime"] is None:
            with lock:
                if holder["runtime"] is None:
                    holder["runtime"] = _build_runtime(base_dir, storage_format, runtime)
        return holder["runtime"]

    server = MCPServer(
        name="evidencespine",
        title="EvidenceSpine Memory Server",
        description=_SERVER_DESCRIPTION,
        version=__version__,
    )
    register_tools(server, get_runtime)
    register_resources(server, get_runtime)
    register_prompts(server, get_runtime)
    return server


def run_server(
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp",
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    runtime: Optional[AgentMemoryRuntime] = None,
) -> None:
    """Run the EvidenceSpine MCP server (blocking)."""
    if transport not in {"stdio", "streamable-http"}:
        raise ValueError(f"unsupported transport: {transport!r}")
    server = create_server(base_dir=base_dir, storage_format=storage_format, runtime=runtime)
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="streamable-http", host=host, port=int(port), streamable_http_path=path)
