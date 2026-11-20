"""What a Claude Code session does with EvidenceSpine (brief -> verify -> handoff).

Run from the repo root:

    PYTHONPATH=src python examples/claude_code_usage.py

Simulates a bugfix session in a temporary workspace: two grounded claims, a
bounded brief, a handoff packet for the auditor, and the health snapshot. This
is the same runtime the MCP server and harness hooks use; the MCP tool map is
in AGENTS.md.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from evidencespine.grounding import ground_file
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def brief_to_markdown(brief: dict) -> str:
    sections = [
        ("Current Goal", brief.get("current_goal", [])),
        ("Locked Decisions", brief.get("locked_decisions", [])),
        ("Recent Verified Facts", brief.get("recent_verified_facts", [])),
        ("Active Risks", brief.get("active_risks", [])),
        ("Open Items", brief.get("open_items", [])),
        ("Next Actions", brief.get("next_actions", [])),
    ]
    lines = [f"# EvidenceSpine Brief: {brief.get('thread_id', '')}"]
    if brief.get("metadata", {}).get("stale"):
        lines.append("\n_Memory is stale for this thread; verify against the repo._")
    for title, items in sections:
        if not items:
            continue
        lines.append(f"\n## {title}")
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="evidencespine_claude_") as tmp:
        workspace = Path(tmp)
        (workspace / "src" / "middleware").mkdir(parents=True)
        (workspace / "tests").mkdir()
        (workspace / "src" / "middleware" / "request_timeout.py").write_text(
            "# retry guard\n"
            "def request_timeout(request):\n"
            "    # Patch retry path in request middleware.\n"
            "    return request.timeout\n",
            encoding="utf-8",
        )
        (workspace / "tests" / "test_request_timeout.py").write_text(
            "# streaming requests are not covered yet\n"
            "def test_streaming_timeout():\n"
            "    pass\n",
            encoding="utf-8",
        )
        patch_item = ground_file(
            "src/middleware/request_timeout.py", 3, 3, source_root=str(workspace)
        )
        risk_item = ground_file(
            "tests/test_request_timeout.py", 1, 1, source_root=str(workspace)
        )
        assert patch_item is not None and risk_item is not None

        settings = EvidenceSpineSettings.from_env(base_dir=str(workspace / ".evidencespine"))
        runtime = AgentMemoryRuntime(config=settings.to_runtime_config())
        thread_id = "claude_demo_bugfix"

        runtime.ingest_event(
            {
                "thread_id": thread_id,
                "event_type": "decision",
                "role": "implementer",
                "source_agent_id": "claude_code",
                "source_turn_id": "turn_17",
                "payload": {
                    "claim": "Patch retry path in request middleware, additive only",
                    "decision": "Patch retry path in request middleware",
                    "fact_state": "verified",
                    "next_actions": ["auditor should verify timeout edge cases"],
                },
                "evidence_refs": ["src/middleware/request_timeout.py#L3-L3"],
                "evidence_items": [patch_item],
                "confidence": 0.87,
                "salience": 0.74,
            }
        )
        runtime.ingest_event(
            {
                "thread_id": thread_id,
                "event_type": "outcome",
                "role": "auditor",
                "source_agent_id": "claude_code_auditor",
                "source_turn_id": "turn_18",
                "payload": {
                    "claim": "Streaming timeout edge case still unverified",
                    "fact_state": "asserted",
                },
                "evidence_refs": ["tests/test_request_timeout.py#L1-L1"],
                "evidence_items": [risk_item],
                "confidence": 0.68,
                "salience": 0.71,
            }
        )

        brief = runtime.build_brief(
            thread_id,
            "current goal, locked decisions, verified facts, active risks, next actions",
        )
        print("=== BRIEF ===")
        print(brief_to_markdown(brief.to_dict()))

        packet = runtime.emit_handoff(
            role="auditor",
            thread_id=thread_id,
            scope="verify patch and regression risk",
        ).to_dict()
        print("\n=== HANDOFF ===")
        print("packet_id:", packet["packet_id"])
        for claim in packet["claims"]:
            print(f"- [{claim['status']}] {claim['claim']}")

        snapshot = runtime.snapshot()
        print("\n=== SNAPSHOT (governance) ===")
        for key in (
            "agent_memory_verified_facts_24h",
            "agent_claim_span_citation_coverage_24h",
            "agent_handoff_span_grounding_rate_24h",
            "agent_evidence_stale_count_24h",
        ):
            print(f"{key}: {snapshot[key]}")

        runtime.store.close()


if __name__ == "__main__":
    main()
