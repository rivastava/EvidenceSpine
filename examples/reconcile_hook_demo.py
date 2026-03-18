from __future__ import annotations

import json
from pathlib import Path

from evidencespine.runtime import AgentMemoryRuntime, RuntimeHooks
from evidencespine.settings import EvidenceSpineSettings


def _read_runtime_status(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    base_dir = ".evidencespine_demo"
    status_path = Path(base_dir) / "runtime_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "scope_id": "release-gate",
                "status": "blocked",
                "validated_at": "2026-03-18T09:30:00Z",
                "validated_by": "toy-runtime-check",
                "fresh_until": "2026-03-18T10:30:00Z",
                "claim": "Release gate is blocked on smoke failures",
                "evidence_ref": "runtime_status.json#L1",
            }
        ),
        encoding="utf-8",
    )

    def reconcile_state(thread_id: str, active_scope_rows: list[dict]) -> list[dict]:
        del active_scope_rows
        status = _read_runtime_status(status_path)
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

    settings = EvidenceSpineSettings.from_env(base_dir=base_dir)
    runtime = AgentMemoryRuntime(
        config=settings.to_runtime_config(),
        hooks=RuntimeHooks(reconcile_state=reconcile_state),
    )

    result = runtime.reconcile("demo")
    print("reconcile:", result)
    print("active_scopes:", runtime.query_view("active_scopes", thread_id="demo").to_dict())


if __name__ == "__main__":
    main()
