from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def main() -> None:
    settings = EvidenceSpineSettings.from_env(base_dir=".evidencespine_demo")
    runtime = AgentMemoryRuntime(config=settings.to_runtime_config())

    runtime.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "source_agent_id": "implementer",
            "source_turn_id": "1",
            "payload": {
                "claim": "Switch to additive patch strategy",
                "decision": "Apply patch set A",
                "fact_state": "verified",
            },
            "evidence_refs": ["reports/decision.md#L1"],
            "confidence": 0.9,
            "salience": 0.7,
        }
    )

    brief = runtime.build_brief("demo", "current status and next actions")
    print(brief.to_dict())


if __name__ == "__main__":
    main()
