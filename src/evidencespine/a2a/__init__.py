"""A2A (Agent-to-Agent) SDK integration.

Exposes EvidenceSpine as an A2A agent: an Agent Card advertising memory
skills, an executor that maps A2A messages to memory operations, and a
FastAPI/Starlette server implementing the A2A 1.0 protocol.

The ``a2a`` extra is required: ``pip install evidencespine[a2a]``.
"""

from evidencespine.a2a.card import build_agent_card
from evidencespine.a2a.executor import EvidenceSpineExecutor
from evidencespine.a2a.server import build_fastapi_app, run_server

__all__ = [
    "build_agent_card",
    "build_fastapi_app",
    "EvidenceSpineExecutor",
    "run_server",
]
