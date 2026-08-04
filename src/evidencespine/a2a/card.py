"""Agent Card construction for the EvidenceSpine A2A agent."""

from __future__ import annotations

from typing import Any

from evidencespine import __version__


def build_agent_card(
    *,
    name: str = "evidencespine-agent",
    description: str | None = None,
    version: str | None = None,
) -> Any:
    """Build the A2A ``AgentCard`` advertising EvidenceSpine memory skills.

    Returns a protobuf ``AgentCard`` (``a2a.types.a2a_pb2.AgentCard``) as
    required by the a2a-sdk 1.1 server components.
    """
    from a2a.types import a2a_pb2

    description = description or (
        "Evidence-bound working-state memory for coding agents: ingest "
        "structured events and facts, build bounded context briefs, and "
        "emit or import evidence-bound handoff packets."
    )
    card = a2a_pb2.AgentCard(
        name=name,
        description=description,
        provider=a2a_pb2.AgentProvider(organization="evidencespine"),
        version=version or str(__version__),
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=a2a_pb2.AgentCapabilities(
            streaming=False,
            push_notifications=False,
        ),
    )
    card.skills.extend(
        [
            a2a_pb2.AgentSkill(
                id="memory.read",
                name="memory.read",
                description=(
                    "Read evidence-bound memory: build a bounded context brief "
                    "for a thread or query a derived agent-state control view "
                    "(my_work, open_gates, stale_claims, contradictions)."
                ),
                tags=["memory", "retrieval", "brief"],
            ),
            a2a_pb2.AgentSkill(
                id="memory.write",
                name="memory.write",
                description=(
                    "Write evidence-bound memory: ingest a structured event or "
                    "append a verified fact with evidence references."
                ),
                tags=["memory", "ingest", "evidence"],
            ),
            a2a_pb2.AgentSkill(
                id="memory.handoff",
                name="memory.handoff",
                description=(
                    "Hand off working state: emit an evidence-bound handoff "
                    "packet for another agent, or import one received from "
                    "another agent."
                ),
                tags=["handoff", "interop", "A2A"],
            ),
            a2a_pb2.AgentSkill(
                id="memory.health",
                name="memory.health",
                description=(
                    "Inspect memory health: 24h snapshot of volumes, verified "
                    "facts, contradictions, brief success/stale rates, and "
                    "handoff completeness; prune rows older than a TTL."
                ),
                tags=["memory", "health", "ttl"],
            ),
        ]
    )
    return card
