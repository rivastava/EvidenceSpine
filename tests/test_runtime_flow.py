import hashlib
import json
from pathlib import Path

from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _runtime(tmp_path: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _jsonl_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


def test_runtime_ingest_brief_handoff_roundtrip_with_evidence_items(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    excerpt = "deploy patch"
    out = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t1",
            "payload": {
                "claim": excerpt,
                "fact_state": "verified",
                "next_actions": ["auditor verify"],
            },
            "evidence_items": [
                {
                    "source_id": "patch.diff",
                    "line_start": 7,
                    "line_end": 9,
                    "excerpt": excerpt,
                    "checksum": f"sha256:{hashlib.sha256(excerpt.encode('utf-8')).hexdigest()}",
                }
            ],
            "confidence": 0.8,
            "salience": 0.7,
        }
    )
    assert out["status"] == "ok"

    facts = _jsonl_rows(tmp_path / ".es" / "facts.jsonl")
    assert facts[-1]["evidence_items"][0]["source_id"] == "patch.diff"

    brief = rt.build_brief("demo", "status")
    b = brief.to_dict()
    assert b["thread_id"] == "demo"
    assert b["schema_version"] == "v2"
    claim = b["recent_verified_facts"][0]
    assert b["citations"][claim]["span_grounded"] is True
    assert b["citation_refs"][claim] == ["patch.diff#L7-L9"]

    packet = rt.emit_handoff("auditor", "demo", "verify")
    p = packet.to_dict()
    assert p["thread_id"] == "demo"
    assert p["checksum"]
    assert p["evidence_items"][0]["source_id"] == "patch.diff"
    assert p["claims"][0]["span_grounded"] is True

    imported = rt.import_handoff(p, source_agent_id="auditor")
    assert imported["status"] == "ok"

    snap = rt.snapshot()
    assert snap["agent_memory_events_24h"] >= 1
    assert snap["agent_handoff_packets_emitted_24h"] >= 1
    assert snap["agent_claim_span_citation_coverage_24h"] >= 1.0


def test_runtime_import_handoff_supports_v1_ref_only_packets(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.import_handoff(
        {
            "schema_version": "v1",
            "packet_id": "legacy_packet",
            "role": "auditor",
            "thread_id": "demo",
            "scope": "verify",
            "claims": [{"claim": "legacy claim", "evidence_refs": ["legacy.md#L1"], "status": "asserted"}],
            "locked_decisions": [],
            "required_validations": ["check legacy flow"],
            "evidence_refs": ["legacy.md#L1"],
            "checksum": "legacy",
        },
        source_agent_id="auditor",
    )
    assert out["status"] == "ok"


def test_runtime_import_handoff_v2_flattens_claim_and_packet_evidence_items(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    packet = {
        "schema_version": "v2",
        "packet_id": "packet_v2",
        "role": "auditor",
        "thread_id": "demo",
        "scope": "verify",
        "claims": [
            {
                "claim": "verify patch",
                "status": "verified",
                "evidence_items": [{"source_id": "src/file.py", "line_start": 10, "line_end": 12}],
            }
        ],
        "unresolved_contradictions": [
            {
                "claim": "CONTRADICTION: drift",
                "reason": "unresolved_contradiction",
                "evidence_items": [{"source_id": "notes.md", "line_start": 3, "line_end": 4}],
            }
        ],
        "required_validations": ["validate v2"],
        "evidence_items": [{"source_id": "packet.md", "line_start": 1, "line_end": 1}],
        "checksum": "placeholder",
    }
    out = rt.import_handoff(packet, source_agent_id="auditor")
    assert out["status"] == "ok"

    events = _jsonl_rows(tmp_path / ".es" / "events.jsonl")
    latest = events[-1]
    assert {item["source_id"] for item in latest["evidence_items"]} == {"packet.md", "src/file.py", "notes.md"}


def test_snapshot_distinguishes_ref_and_span_coverage(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t1",
            "payload": {"claim": "ref only claim", "fact_state": "verified"},
            "evidence_refs": ["ref_only.md#L1"],
        }
    )
    excerpt = "span grounded claim"
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t2",
            "payload": {"claim": excerpt, "fact_state": "verified"},
            "evidence_items": [
                {
                    "source_id": "span.md",
                    "line_start": 4,
                    "line_end": 5,
                    "excerpt": excerpt,
                    "checksum": f"sha256:{hashlib.sha256(excerpt.encode('utf-8')).hexdigest()}",
                }
            ],
        }
    )

    rt.build_brief("demo", "status")
    snap = rt.snapshot()
    assert snap["agent_claim_ref_citation_coverage_24h"] >= snap["agent_claim_span_citation_coverage_24h"]
    assert snap["agent_claim_span_citation_coverage_24h"] < snap["agent_claim_ref_citation_coverage_24h"]
    assert snap["agent_claim_excerpt_fidelity_24h"] == 1.0


def test_runtime_dedupe_hits(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    event = {
        "thread_id": "demo",
        "event_type": "intent",
        "source_agent_id": "impl",
        "source_turn_id": "t1",
        "payload": {"claim": "same"},
    }
    first = rt.ingest_event(event)
    second = rt.ingest_event(event)
    assert first["status"] == "ok"
    assert second["status"] in {"ok", "deduped"}
