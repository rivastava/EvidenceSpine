"""A2A 1.0 server: FastAPI app assembly and runner.

Requires the ``[a2a]`` extra (a2a-sdk + fastapi/uvicorn).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from evidencespine.a2a.card import build_agent_card
from evidencespine.a2a.executor import EvidenceSpineExecutor
from evidencespine.settings import EvidenceSpineSettings


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
) -> Any:
    """Build a FastAPI app exposing the EvidenceSpine A2A agent.

    ``runtime`` may be injected (tests/embedding); otherwise one is built
    lazily on first request from ``base_dir`` / ``storage_format`` (or env
    defaults). The app serves the Agent Card at ``/.well-known/agent-card.json``
    and the A2A JSON-RPC endpoint at ``rpc_url``.
    """
    from a2a.server.request_handlers import DefaultRequestHandlerV2
    from a2a.server.routes import agent_card_routes, jsonrpc_routes, rest_routes
    from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
    from a2a.server.tasks import InMemoryTaskStore
    from fastapi import FastAPI

    get_runtime = _default_get_runtime(base_dir, storage_format) if runtime is None else (lambda: runtime)

    card = build_agent_card(name=name)
    executor = EvidenceSpineExecutor(get_runtime)
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
    return app


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
) -> None:
    """Run the A2A server with uvicorn (blocking)."""
    import uvicorn

    app = build_fastapi_app(base_dir=base_dir, storage_format=storage_format)
    uvicorn.run(app, host=host, port=int(port), log_level="info")
