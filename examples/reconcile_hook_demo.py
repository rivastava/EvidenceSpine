"""Register a runtime reconciliation hook and derive control state from it.

Run from the repo root:

    PYTHONPATH=src python examples/reconcile_hook_demo.py

The hook reads an external runtime status file and returns control-state
events; EvidenceSpine ingests them and derives the active_scopes view. The
evidence item is grounded from the status file itself.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from evidencespine.grounding import ground_file
from evidencespine.runtime import AgentMemoryRuntime, RuntimeHooks
from evidencespine.settings import EvidenceSpineSettings


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="evidencespine_reconcile_") as tmp:
        workspace = Path(tmp)
        base_dir = workspace / ".evidencespine"
        status_path = workspace / "runtime_status.json"
        status_payload = {
            "scope_id": "release-gate",
            "status": "blocked",
            "validated_at": "2026-03-18T09:30:00Z",
            "validated_by": "toy-runtime-check",
            "fresh_until": "2099-01-01T00:00:00Z",
            "claim": "Release gate is blocked on smoke failures",
            "evidence_ref": "runtime_status.json#L1-L1",
        }
        status_path.write_text(json.dumps(status_payload, indent=2) + "\n", encoding="utf-8")
        gate_item = ground_file("runtime_status.json", 1, 1, source_root=str(workspace))
        assert gate_item is not None

        def reconcile_state(thread_id: str, active_scope_rows: list[dict]) -> list[dict]:
            del active_scope_rows
            status = json.loads(status_path.read_text(encoding="utf-8"))
            return [
                {
                    "thread_id": thread_id,
                    "event_type": "reflection",
                    "role": "operator",
                    "source_agent_id": "toy_runtime_probe",
                    "source_turn_id": "runtime_status_1",
                    "payload": {
                        "claim": status["claim"],
                        "fact_state": "verified",
                    },
                    "evidence_refs": [status["evidence_ref"]],
                    "evidence_items": [gate_item],
                    "state_context": {
                        "scope_id": status["scope_id"],
                        "state_kind": "global_blocker",
                        "status": status["status"],
                        "state_basis": "runtime_validated",
                        "validated_at": status["validated_at"],
                        "validated_by": status["validated_by"],
                        "fresh_until": status["fresh_until"],
                    },
                }
            ]

        settings = EvidenceSpineSettings.from_env(base_dir=str(base_dir))
        runtime = AgentMemoryRuntime(
            config=settings.to_runtime_config(),
            hooks=RuntimeHooks(reconcile_state=reconcile_state),
        )

        result = runtime.reconcile("demo")
        print("reconcile:", result)
        view = runtime.query_view("active_scopes", thread_id="demo").to_dict()
        print("active_scopes:", [(row["scope_id"], row["status"]) for row in view["rows"]])
        print("open_gates:", runtime.query_view("open_gates", thread_id="demo").to_dict()["rows"])

        runtime.store.close()


if __name__ == "__main__":
    main()
