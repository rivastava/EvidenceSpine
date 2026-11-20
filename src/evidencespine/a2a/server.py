"""A2A 1.0 server: FastAPI app assembly and runner.

Requires the [a2a] extra (a2a-sdk + fastapi/uvicorn).

Security: when an auth token is configured (auth_token argument,
EvidenceSpineSettings.a2a_auth_token or EVIDENCESPINE_A2A_AUTH_TOKEN),
every HTTP path except the public Agent Card
(/.well-known/agent-card.json) requires "Authorization: Bearer <token>".
Token comparison is constant-time (hmac.compare_digest). run_server refuses
to bind a non-loopback host without a token unless
EVIDENCESPINE_A2A_ALLOW_UNAUTHENTICATED (or the settings flag) opts in.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
from typing import Any, Callable, Optional

from evidencespine.a2a.card import build_agent_card
from evidencespine.a2a.executor import EvidenceSpineExecutor
from evidencespine.settings import EvidenceSpineSettings

#: The only path that stays publicly readable when auth is enabled (discovery).
PUBLIC_AGENT_CARD_PATH = "/.well-known/agent-card.json"

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


def _grounding_root(source_root: Optional[str] = None) -> str:
    """Resolve the A2A grounding root: explicit arg, env, then cwd."""
    configured = str(source_root or "").strip() or str(os.getenv("EVIDENCESPINE_SOURCE_ROOT", "") or "").strip()
    return os.path.realpath(configured or os.getcwd())


def _resolve_auth_token(auth_token: Optional[str]) -> str:
    """Resolve the A2A bearer token: argument, then settings, then env.

    An explicit empty string disables auth (loopback development only); when
    the argument is None the configured token is looked up.
    """
    if auth_token is not None:
        return str(auth_token).strip()
    token = ""
    try:
        settings = EvidenceSpineSettings.from_env()
        token = str(getattr(settings, "a2a_auth_token", "") or "").strip()
    except Exception:
        token = ""
    if not token:
        token = str(os.getenv("EVIDENCESPINE_A2A_AUTH_TOKEN", "") or "").strip()
    return token


def _allow_unauthenticated() -> bool:
    """Honor the explicit opt-in for unauthenticated non-loopback binds."""
    try:
        settings = EvidenceSpineSettings.from_env()
        if bool(getattr(settings, "a2a_allow_unauthenticated", False)):
            return True
    except Exception:
        pass
    return _env_truthy("EVIDENCESPINE_A2A_ALLOW_UNAUTHENTICATED")


def _is_public_path(path: str) -> bool:
    """Only the Agent Card discovery document is public."""
    return str(path or "").rstrip("/") == PUBLIC_AGENT_CARD_PATH.rstrip("/")


class A2ABearerAuthMiddleware:
    """Pure-ASGI bearer-token guard for the A2A HTTP surface.

    Streaming (SSE) responses pass through untouched because this is an ASGI
    wrapper rather than a BaseHTTPMiddleware. The Agent Card
    (/.well-known/agent-card.json) is public so client discovery still works;
    every other path requires a constant-time-matching bearer token.
    """

    def __init__(self, app: Any, token: str) -> None:
        expected = str(token or "").strip().encode("utf-8")
        if not expected:
            raise ValueError("A2ABearerAuthMiddleware requires a non-empty token")
        self.app = app
        self._expected = expected

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or _is_public_path(str(scope.get("path", ""))):
            await self.app(scope, receive, send)
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
        if hmac.compare_digest(provided, self._expected):
            await self.app(scope, receive, send)
            return
        body = b'{"error":"invalid_token","error_description":"Authentication required","status":401}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b'Bearer realm="evidencespine", error="invalid_token"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _default_get_runtime(base_dir: Optional[str], storage_format: Optional[str]) -> Callable[[], Any]:
    built: list[Any] = []
    base_dir = str(base_dir or os.environ.get("EVIDENCESPINE_BASE_DIR", ".evidencespine"))

    def get_runtime() -> Any:
        from evidencespine.runtime import AgentMemoryRuntime

        if not built:
            settings = EvidenceSpineSettings.from_env(
                base_dir=base_dir,
                storage_format=storage_format or None,
            )
            built.append(AgentMemoryRuntime(config=settings.to_runtime_config()))
        return built[0]

    return get_runtime


def build_fastapi_app(
    *,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    runtime: Any = None,
    name: str = "evidencespine-agent",
    rpc_url: str = "/",
    auth_token: Optional[str] = None,
    source_root: Optional[str] = None,
    card_url: str = "",
) -> Any:
    """Build a FastAPI app exposing the EvidenceSpine A2A agent.

    runtime may be injected (tests/embedding); otherwise one is built lazily on
    first request from base_dir / storage_format (or env defaults). The app
    serves the Agent Card at /.well-known/agent-card.json and the A2A JSON-RPC
    endpoint at rpc_url.

    auth_token enables bearer auth for every path except the Agent Card. None
    (default) resolves the token from settings / environment; an empty string
    explicitly disables auth (loopback development only). source_root is the
    grounding root for handoff packet path imports (defaults to
    EVIDENCESPINE_SOURCE_ROOT or the process cwd) - absolute/escaping packet
    paths are rejected.
    """
    from a2a.server.request_handlers import DefaultRequestHandlerV2
    from a2a.server.routes import agent_card_routes, jsonrpc_routes, rest_routes
    from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
    from a2a.server.tasks import InMemoryTaskStore
    from fastapi import FastAPI

    token = _resolve_auth_token(auth_token)
    grounding_root = _grounding_root(source_root)
    get_runtime = _default_get_runtime(base_dir, storage_format) if runtime is None else (lambda: runtime)

    card = build_agent_card(name=name, url=card_url)
    executor = EvidenceSpineExecutor(get_runtime, source_root=grounding_root)
    handler = DefaultRequestHandlerV2(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    app = FastAPI(title=name, version=str(card.version), docs_url=None, redoc_url=None)
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=agent_card_routes.create_agent_card_routes(card),
        jsonrpc_routes=jsonrpc_routes.create_jsonrpc_routes(handler, rpc_url=rpc_url),
        rest_routes=rest_routes.create_rest_routes(handler),
    )
    app.state.a2a_handler = handler
    if token:
        app.add_middleware(A2ABearerAuthMiddleware, token=token)
    return app


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    auth_token: Optional[str] = None,
    source_root: Optional[str] = None,
    card_url: str = "",
) -> None:
    """Run the A2A server with uvicorn (blocking).

    Refuses to start an unauthenticated server on a non-loopback host unless
    EVIDENCESPINE_A2A_AUTH_TOKEN is set or
    EVIDENCESPINE_A2A_ALLOW_UNAUTHENTICATED is truthy.
    """
    token = _resolve_auth_token(auth_token)
    if token:
        print(
            "EvidenceSpine A2A server requires "
            "Authorization: Bearer <EVIDENCESPINE_A2A_AUTH_TOKEN>.",
            flush=True,
        )
    elif not _is_loopback_host(host):
        if not _allow_unauthenticated():
            raise ValueError(
                "refusing unauthenticated non-loopback A2A bind; set "
                "EVIDENCESPINE_A2A_AUTH_TOKEN or "
                "EVIDENCESPINE_A2A_ALLOW_UNAUTHENTICATED=1"
            )
        print(
            "WARNING: EvidenceSpine A2A server is running WITHOUT authentication "
            f"on non-loopback host {host!r} because "
            "EVIDENCESPINE_A2A_ALLOW_UNAUTHENTICATED is set. Anyone who can reach "
            "this port can read, write and prune memory.",
            flush=True,
        )
    else:
        print(
            "WARNING: EvidenceSpine A2A server is running WITHOUT authentication "
            "on loopback. Set EVIDENCESPINE_A2A_AUTH_TOKEN to require a bearer "
            "token (required before binding to a non-loopback host).",
            flush=True,
        )

    import uvicorn

    public_url = str(card_url or "").strip() or "http://" + str(host) + ":" + str(int(port)) + "/"
    app = build_fastapi_app(
        base_dir=base_dir,
        storage_format=storage_format,
        auth_token=token,
        source_root=source_root,
        card_url=public_url,
    )
    uvicorn.run(app, host=host, port=int(port), log_level="info")
