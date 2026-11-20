"""EvidenceSpine MCP server.

The ``mcp`` package is an optional dependency (extra ``[mcp]``), so it is
imported lazily inside functions only; importing this module never requires it.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
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


def _grounding_root() -> str:
    settings = EvidenceSpineSettings.from_env()
    return os.path.realpath(settings.source_root or os.getcwd())


_TRUTHY = {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    raw = os.getenv(name)
    return raw is not None and str(raw).strip().lower() in _TRUTHY


def _is_loopback_host(host: str) -> bool:
    """Return True for loopback bind hosts (127.0.0.1, ::1, localhost)."""
    text = str(host or "").strip().lower()
    if not text:
        return False
    if text == "localhost":
        return True
    try:
        return ipaddress.ip_address(text.strip("[]")).is_loopback
    except ValueError:
        return False


def _allow_unauthenticated() -> bool:
    """Honor the explicit opt-in for unauthenticated non-loopback binds."""
    try:
        settings = EvidenceSpineSettings.from_env()
        if bool(getattr(settings, "mcp_allow_unauthenticated", False)):
            return True
    except Exception:
        pass
    return _env_truthy("EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED")


def _resolve_auth_token(settings: EvidenceSpineSettings) -> str:
    """Resolve the MCP bearer token from settings, then environment."""
    token = str(getattr(settings, "mcp_auth_token", "") or "").strip()
    if not token:
        token = str(os.getenv("EVIDENCESPINE_MCP_AUTH_TOKEN", "") or "").strip()
    return token


def create_server(
    *,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    runtime: Optional[AgentMemoryRuntime] = None,
) -> Any:
    """Create an EvidenceSpine MCPServer instance.

    ``runtime`` may be injected (tests/embedding); otherwise one is built lazily
    on first request from ``base_dir`` / ``storage_format`` (or env defaults).
    Server-side grounding (``ground`` tool, ``ground_refs`` on ingest) is
    confined to ``EVIDENCESPINE_SOURCE_ROOT`` (or the server cwd) and rejects
    absolute paths.
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

    try:
        from evidencespine.usage import mcp_instructions_markdown as _primer

        # Condensed primer (<= 3800 chars) so MCP initialize instructions are
        # self-contained instead of a mid-sentence truncation of the full guide.
        _instructions = _primer()
    except Exception:
        _instructions = _SERVER_DESCRIPTION
    try:
        server = MCPServer(
            name="evidencespine",
            title="EvidenceSpine Memory Server",
            description=_SERVER_DESCRIPTION,
            version=__version__,
            instructions=_instructions,
        )
    except TypeError:
        # Older MCP SDKs lack the instructions field; resources carry the guide.
        server = MCPServer(
            name="evidencespine",
            title="EvidenceSpine Memory Server",
            description=_SERVER_DESCRIPTION,
            version=__version__,
        )
    grounding_root = _grounding_root()
    register_tools(server, get_runtime, source_root=grounding_root)
    register_resources(server, get_runtime)
    register_prompts(server, get_runtime, source_root=grounding_root)
    return server


def _static_bearer_middleware(app: Any, token: str) -> Any:
    """Wrap an ASGI app so every HTTP request must present ``Authorization: Bearer <token>``.

    The token comparison is constant-time (``hmac.compare_digest``) so the
    guard does not leak the expected credential through response timing.
    """

    expected = str(token or "").strip().encode("utf-8")
    if not expected:
        raise ValueError("_static_bearer_middleware requires a non-empty token")

    async def dispatch(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        provided = b""
        for key, value in scope.get("headers") or []:
            if key.lower() != b"authorization":
                continue
            try:
                raw = value.decode("latin-1")
            except Exception:
                raw = ""
            scheme, _, credential = raw.partition(" ")
            if scheme.strip().lower() == "bearer":
                provided = credential.strip().encode("utf-8", "surrogateescape")
            break
        if not hmac.compare_digest(provided, expected):
            body = b'{"error":"invalid_token","error_description":"Authentication required"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"www-authenticate", b'Bearer error="invalid_token"'),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await app(scope, receive, send)

    return dispatch


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
    """Run the EvidenceSpine MCP server (blocking).

    Refuses to start an unauthenticated streamable-http server on a
    non-loopback host unless EVIDENCESPINE_MCP_AUTH_TOKEN is set or
    EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED is truthy.
    """
    if transport not in {"stdio", "streamable-http"}:
        raise ValueError(f"unsupported transport: {transport!r}")
    settings = EvidenceSpineSettings.from_env(base_dir=base_dir, storage_format=storage_format)
    token = _resolve_auth_token(settings)
    if transport == "streamable-http" and not token and not _is_loopback_host(host):
        if not _allow_unauthenticated():
            raise ValueError(
                "refusing unauthenticated non-loopback MCP bind; set "
                "EVIDENCESPINE_MCP_AUTH_TOKEN or "
                "EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED=1"
            )
        print(
            "WARNING: EvidenceSpine MCP streamable-http server is running "
            "WITHOUT authentication on non-loopback host "
            f"{host!r} because EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED is set. "
            "Anyone who can reach this port can read, write and prune memory.",
            flush=True,
        )
    server = create_server(base_dir=base_dir, storage_format=storage_format, runtime=runtime)
    if transport == "stdio":
        server.run(transport="stdio")
        return
    import uvicorn

    if not token:
        print(
            "WARNING: EvidenceSpine MCP streamable-http server is running "
            "WITHOUT authentication. Set EVIDENCESPINE_MCP_AUTH_TOKEN to require "
            "a bearer token. Prefer binding to 127.0.0.1 and avoid 0.0.0.0 on "
            "untrusted networks.",
            flush=True,
        )
    else:
        print(
            "EvidenceSpine MCP streamable-http server requires "
            "Authorization: Bearer <EVIDENCESPINE_MCP_AUTH_TOKEN>.",
            flush=True,
        )
    app = server.streamable_http_app(streamable_http_path=path, host=host)
    if token:
        app = _static_bearer_middleware(app, token)
    uvicorn.run(app, host=host, port=int(port), log_level="info")
