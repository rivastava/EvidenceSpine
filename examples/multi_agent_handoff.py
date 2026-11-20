"""Multi-agent handoff: implementer emits, auditor imports and orients.

Run from the repo root:

    PYTHONPATH=src python examples/multi_agent_handoff.py

Uses a temporary workspace. Both claims are grounded in real files, so the
handoff is span-grounded end to end and the importer sees the same state.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from evidencespine.grounding import ground_file
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="evidencespine_handoff_") as tmp:
        workspace = Path(tmp)
        (workspace / "src").mkdir()
        (workspace / "reports").mkdir()
        (workspace / "src" / "engine.py").write_text(
            "# engine\n"
            "def apply(plan):\n"
            "    # Objective dedupe guard implemented.\n"
            "    return plan\n",
            encoding="utf-8",
        )
        (workspace / "reports" / "pytest_smoke.txt").write_text(
            "pytest smoke: 12 passed, 0 failed\n", encoding="utf-8"
        )
        implementation = ground_file("src/engine.py", 3, 3, source_root=str(workspace))
        gate = ground_file("reports/pytest_smoke.txt", 1, 1, source_root=str(workspace))
        assert implementation is not None and gate is not None

        settings = EvidenceSpineSettings.from_env(base_dir=str(workspace / ".evidencespine"))
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
                "evidence_refs": ["src/engine.py#L3-L3"],
                "evidence_items": [implementation],
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
                    "claim": "Regression gate passed in the smoke run",
                    "fact_state": "verified",
                },
                "evidence_refs": ["reports/pytest_smoke.txt#L1-L1"],
                "evidence_items": [gate],
                "state_context": {
                    "scope_id": "regression-gate",
                    "state_kind": "pending_gate",
                    "status": "ready",
                    "state_basis": "runtime_validated",
                    "validated_at": "2026-03-18T09:30:00Z",
                    "validated_by": "pytest-smoke",
                    "fresh_until": "2099-01-01T00:00:00Z",
                },
                "confidence": 0.88,
                "salience": 0.72,
            }
        )

        print("active_scopes:", runtime.query_view("active_scopes", thread_id="ci_loop").to_dict()["rows"][:2])
        print("open_gates:", runtime.query_view("open_gates", thread_id="ci_loop").to_dict()["rows"])

        packet = runtime.emit_handoff(
            role="auditor", thread_id="ci_loop", scope="verify implementation claims"
        )
        print("handoff_packet_id:", packet.to_dict().get("packet_id"))

        imported = runtime.import_handoff(packet.to_dict(), source_agent_id="auditor_agent")
        print("import_status:", imported.get("status"), "| failed_ingests:", imported.get("failed_ingests"))
        print("successor_brief:", runtime.build_brief("ci_loop", "verify").to_dict()["open_items"])

        runtime.store.close()


if __name__ == "__main__":
    main()
