from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def main() -> None:
    settings = EvidenceSpineSettings.from_env(base_dir=".evidencespine_demo")
    runtime = AgentMemoryRuntime(config=settings.to_runtime_config())

    runtime.ingest_event(
        {
            "thread_id": "ci_loop",
            "event_type": "decision",
            "source_agent_id": "implementer_agent",
            "source_turn_id": "t100",
            "payload": {
                "claim": "Implemented objective dedupe guard",
                "fact_state": "verified",
                "next_actions": ["Auditor should verify regression gates"],
            },
            "evidence_refs": ["src/engine.py:42"],
            "confidence": 0.84,
            "salience": 0.76,
        }
    )

    packet = runtime.emit_handoff(role="auditor", thread_id="ci_loop", scope="verify implementation claims")
    print("handoff_packet_id:", packet.to_dict().get("packet_id"))

    imported = runtime.import_handoff(packet.to_dict(), source_agent_id="auditor_agent")
    print("import_status:", imported.get("status"))
    print("snapshot:", runtime.snapshot())


if __name__ == "__main__":
    main()
