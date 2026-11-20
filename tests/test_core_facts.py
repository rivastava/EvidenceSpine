"""Core hardening: verified policy, redaction order, fact identity, supersession."""

from __future__ import annotations

import hashlib
from pathlib import Path

from evidencespine.protocol import AgentMemoryFact, event_to_fact_candidates
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings
from grounding_utils import grounded_item


def _runtime(tmp_path: Path, *, storage_format: str = "sqlite", **overrides) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"), storage_format=storage_format)
    for key, value in overrides.items():
        setattr(settings, key, value)
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _sha_item(excerpt: str = "sha grounded excerpt") -> dict:
    return {
        "source_id": "src/evidence.py",
        "line_start": 1,
        "line_end": 1,
        "excerpt": excerpt,
        "checksum": "sha256:" + hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
    }


def _event(claim: str, **extra) -> dict:
    row = {
        "thread_id": "demo",
        "event_type": "outcome",
        "role": "auditor",
        "source_agent_id": "auditor",
        "source_turn_id": "turn",
        "payload": {"claim": claim},
    }
    row.update(extra)
    return row


# --- item 3: shared verified policy on append_fact ------------------------


def test_append_fact_ungrounded_verified_is_downgraded(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.append_fact({"thread_id": "demo", "claim": "ungrounded assertion", "state": "verified"})
    assert out["status"] == "ok"
    assert out["policy_downgrades"] == 1
    fact = list(rt.store.iter_facts())[0]
    assert fact["state"] == "asserted"
    assert fact["metadata"]["policy"] == "verified_requires_span"
    rt.store.close()


def test_append_fact_grounded_excerpt_stays_verified(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.append_fact(
        {"thread_id": "demo", "claim": "grounded assertion", "state": "verified", "evidence_items": [_sha_item()]}
    )
    assert out["policy_downgrades"] == 0
    assert list(rt.store.iter_facts())[0]["state"] == "verified"
    rt.store.close()


def test_append_fact_metadata_verification_stays_verified(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.append_fact(
        {
            "thread_id": "demo",
            "claim": "provenanced assertion",
            "state": "verified",
            "metadata": {"verification": {"method": "test", "reference": "pytest tests/test_core_facts.py::t"}},
        }
    )
    assert out["policy_downgrades"] == 0
    fact = list(rt.store.iter_facts())[0]
    assert fact["state"] == "verified"
    assert fact["metadata"]["verification"]["reference"] == "pytest tests/test_core_facts.py::t"
    rt.store.close()


def test_ingest_and_append_share_verified_policy(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    ingest_out = rt.ingest_event(_event("ungrounded ingest", payload={"claim": "ungrounded ingest", "fact_state": "verified"}))
    assert ingest_out["status"] == "ok"
    assert ingest_out.get("policy_downgrades") == 1
    append_out = rt.append_fact({"thread_id": "demo", "claim": "ungrounded append", "state": "verified"})
    assert append_out["policy_downgrades"] == 1
    assert all(f["state"] == "asserted" for f in rt.store.iter_facts())
    rt.store.close()


def test_verified_policy_can_be_disabled(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, verified_requires_span=False)
    out = rt.append_fact({"thread_id": "demo", "claim": "policy off", "state": "verified"})
    assert out["policy_downgrades"] == 0
    assert list(rt.store.iter_facts())[0]["state"] == "verified"
    rt.store.close()


# --- item 10: redaction before identity/derivation ------------------------


def test_redaction_happens_before_hashing_and_fact_derivation(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.ingest_event(_event("account 123456789012 leaked"))
    assert out["status"] == "ok"
    event = list(rt.store.iter_events())[0]
    assert event["payload"]["claim"] == "account [REDACTED] leaked"
    assert not any(ch.isdigit() and ch in "123456789012" for ch in "")  # sanity: digits gone
    assert "123456789012" not in str(event)
    assert not event["event_id"].startswith("ame_") or event["event_hash"] == out["event_hash"]
    fact = list(rt.store.iter_facts())[0]
    assert fact["claim"] == "account [REDACTED] leaked"
    # Store-level redaction is a no-op on an already redacted row.
    assert rt.store.state["redactions_total"] == 1
    # The recomputed event hash binds the redacted payload.
    from evidencespine.protocol import _canonical_hash, merge_evidence_refs, normalize_evidence_items

    expected = _canonical_hash(
        {
            "thread_id": event["thread_id"],
            "event_type": event["event_type"],
            "source_agent_id": event["source_agent_id"],
            "source_turn_id": event["source_turn_id"],
            "payload": event["payload"],
            "evidence_refs": merge_evidence_refs(event.get("evidence_refs"), normalize_evidence_items(event.get("evidence_items"))),
            "evidence_items": normalize_evidence_items(event.get("evidence_items")),
            "state_context": event.get("state_context", {}),
        }
    )
    assert event["event_hash"] == expected
    rt.store.close()


def test_redacted_claims_supersede_their_twin(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    assert rt.ingest_event(_event("ticket 123456789012 open", payload={"claim": "ticket 123456789012 open", "fact_state": "asserted"}))["status"] == "ok"
    assert rt.ingest_event(
        _event(
            "ticket 123456789012 open",
            event_type="outcome",
            payload={"claim": "ticket 123456789012 open", "fact_state": "verified"},
            evidence_items=[grounded_item(tmp_path, name="twin/evidence.py")],
        )
    )["status"] == "ok"
    facts = [f for f in rt.store.iter_facts() if f["state"] in {"asserted", "verified"}]
    asserted = [f for f in facts if f["state"] == "asserted"]
    verified = [f for f in facts if f["state"] == "verified"]
    assert asserted and verified
    assert asserted[0]["claim"] == "ticket [REDACTED] open"
    assert verified[0]["claim"] == "ticket [REDACTED] open"
    assert verified[0]["supersedes_fact_id"] == asserted[0]["fact_id"], "twin matching must use the redacted claim"
    rt.store.close()


def test_redaction_preserves_evidence_checksums(tmp_path: Path) -> None:
    item = grounded_item(tmp_path, name="checksum/evidence.py")
    rt = _runtime(tmp_path)
    out = rt.ingest_event(
        _event(
            "checksum 123456789012 claim",
            payload={"claim": "checksum 123456789012 claim", "fact_state": "verified"},
            evidence_items=[item],
        )
    )
    assert out["status"] == "ok"
    assert not out.get("policy_downgrades")
    stored = list(rt.store.iter_facts())[0]["evidence_items"][0]
    assert stored["excerpt"] == item["excerpt"]
    assert stored["checksum"] == item["checksum"]
    rt.store.close()


# --- item 11: case-insensitive fact identity ------------------------------


def test_fact_identity_is_case_insensitive_but_text_preserved() -> None:
    upper = AgentMemoryFact(fact_id="", thread_id="demo", claim="Fix The Bug", source_turn_id="t1").to_dict()
    lower = AgentMemoryFact(fact_id="", thread_id="demo", claim="fix the bug", source_turn_id="t1").to_dict()
    assert upper["fact_id"] == lower["fact_id"]
    assert upper["claim"] == "Fix The Bug"
    assert lower["claim"] == "fix the bug"


def test_x_and_capital_x_share_fact_id(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(_event("X", payload={"claim": "X", "fact_state": "asserted"}, source_turn_id="same-turn"))
    rt.ingest_event(_event("x", payload={"claim": "x", "fact_state": "asserted"}, source_turn_id="same-turn"))
    facts = list(rt.store.iter_facts())
    assert len(facts) == 2
    assert facts[0]["fact_id"] == facts[1]["fact_id"]
    assert {facts[0]["claim"], facts[1]["claim"]} == {"X", "x"}
    brief = rt.build_brief("demo", "status").to_dict()
    assert sum(1 for item in brief["open_items"] if item.startswith("X") or item.startswith("x")) == 1
    rt.store.close()


# --- item 12: state_context.supersedes -> fact supersedes ------------------


def test_state_context_supersedes_sets_fact_supersedes_fact_id(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(_event("old claim", payload={"claim": "old claim", "fact_state": "asserted"}, source_turn_id="t1"))
    old_fact = list(rt.store.iter_facts())[0]
    rt.ingest_event(
        _event(
            "new claim",
            payload={"claim": "new claim", "fact_state": "asserted"},
            source_turn_id="t2",
            state_context={
                "scope_id": "scope-a",
                "state_kind": "agent_local_work",
                "status": "superseded",
                "supersedes": old_fact["fact_id"],
                "fresh_until": "2099-01-01T00:00:00Z",
            },
        )
    )
    new_fact = [f for f in rt.store.iter_facts() if f["claim"] == "new claim"][0]
    assert new_fact["supersedes_fact_id"] == old_fact["fact_id"]


def test_event_to_fact_candidates_promotes_state_context_supersedes() -> None:
    event = {
        "event_id": "e1",
        "thread_id": "demo",
        "event_type": "reflection",
        "source_agent_id": "a",
        "source_turn_id": "t1",
        "payload": {"claim": "replacement"},
        "state_context": {
            "scope_id": "s",
            "state_kind": "agent_local_work",
            "status": "superseded",
            "supersedes": "amf_target",
            "fresh_until": "2099-01-01T00:00:00Z",
        },
    }
    candidates = event_to_fact_candidates(event)
    assert candidates and candidates[0]["supersedes_fact_id"] == "amf_target"
