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
            "state_context": {
                "scope_id": "objective-dedupe-guard",
                "state_kind": "agent_local_work",
                "status": "active",
                "owner_agent_id": "implementer_agent",
            },
            "confidence": 0.84,
            "salience": 0.76,
        }
    )
    runtime.ingest_event(
        {
            "thread_id": "ci_loop",
            "event_type": "reflection",
            "source_agent_id": "runtime_probe",
            "source_turn_id": "gate_1",
            "payload": {
                "claim": "Regression gate remains ready for auditor verification",
                "fact_state": "verified",
            },
            "state_context": {
                "scope_id": "regression-gate",
                "state_kind": "pending_gate",
                "status": "ready",
                "state_basis": "runtime_validated",
                "validated_at": "2026-03-18T09:30:00Z",
                "validated_by": "pytest-smoke",
                "fresh_until": "2026-03-18T10:30:00Z",
            },
            "evidence_refs": ["reports/pytest_smoke.txt#L1"],
            "confidence": 0.88,
            "salience": 0.72,
        }
    )

    print("active_scopes:", runtime.query_view("active_scopes", thread_id="ci_loop").to_dict())
    print("open_gates:", runtime.query_view("open_gates", thread_id="ci_loop").to_dict())
    packet = runtime.emit_handoff(role="auditor", thread_id="ci_loop", scope="verify implementation claims")
    print("handoff_packet_id:", packet.to_dict().get("packet_id"))

    imported = runtime.import_handoff(packet.to_dict(), source_agent_id="auditor_agent")
    print("import_status:", imported.get("status"))
    print("snapshot:", runtime.snapshot())


if __name__ == "__main__":
    main()
