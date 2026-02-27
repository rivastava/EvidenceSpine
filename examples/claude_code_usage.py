import shutil
from pathlib import Path

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
    for title, items in sections:
        if not items:
            continue
        lines.append(f"\n## {title}")
        for item in items:
            lines.append(f"- {item}")
    return "\n".join(lines)


def main() -> None:
    base_dir = Path(".evidencespine_claude_demo")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    settings = EvidenceSpineSettings.from_env(base_dir=str(base_dir))
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
                "claim": "Use additive patch only. Do not modify auth token rotation.",
                "decision": "Patch retry path in request middleware",
                "fact_state": "verified",
                "next_actions": ["auditor should verify timeout edge cases"],
            },
            "evidence_refs": ["src/middleware/request_timeout.py:42"],
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
                "claim": "Timeout edge case still unverified for streaming requests.",
                "fact_state": "asserted",
            },
            "evidence_refs": ["tests/test_request_timeout.py:88"],
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
    )
    print("\n=== HANDOFF PACKET ===")
    print(packet.to_dict())

    print("\n=== SNAPSHOT ===")
    print(runtime.snapshot())


if __name__ == "__main__":
    main()
