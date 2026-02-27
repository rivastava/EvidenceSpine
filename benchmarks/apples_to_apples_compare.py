#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import socket
import string
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import requests

from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower())


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return float(len(sa.intersection(sb)) / max(1, len(sa.union(sb))))


def _q(samples_ms: List[float], qv: float) -> float:
    if not samples_ms:
        return 0.0
    ordered = sorted(samples_ms)
    idx = int(round((len(ordered) - 1) * qv))
    idx = max(0, min(len(ordered) - 1, idx))
    return float(ordered[idx])


def _rand_token(n: int = 8) -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(max(3, n)))


def _mk_event(i: int) -> Dict[str, Any]:
    segment = f"segment_{i % 13}"
    return {
        "thread_id": "bench_thread",
        "event_type": "decision" if i % 3 else "outcome",
        "role": "implementer" if i % 2 else "auditor",
        "source_agent_id": "bench_runner",
        "source_turn_id": f"turn_{i}",
        "payload": {
            "claim": f"decision {i} about {_rand_token(6)} {_rand_token(5)} {segment} status {_rand_token(4)}",
            "fact_state": "verified" if i % 7 == 0 else "asserted",
            "decision": f"apply_patch_{i % 7}",
            "outcome": "ok" if i % 4 else "needs_review",
            "next_actions": [f"validate_{i % 9}", f"audit_{i % 6}"],
            "target": f"target_{i % 11}",
        },
        "evidence_refs": [f"bench/file_{i % 17}.md#L{i % 100 + 1}"],
        "confidence": 0.7,
        "salience": 0.6,
    }


def _mk_query(i: int) -> str:
    return f"what changed in segment_{i % 13} and target_{i % 11}?"


def _event_to_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
    refs = event.get("evidence_refs", []) if isinstance(event.get("evidence_refs", []), list) else []
    return {
        "thread_id": str(event.get("thread_id", "bench_thread")),
        "event_type": str(event.get("event_type", "reflection")),
        "role": str(event.get("role", "unknown")),
        "source_turn_id": str(event.get("source_turn_id", "")),
        "claim": str(payload.get("claim", "")),
        "decision": str(payload.get("decision", "")),
        "outcome": str(payload.get("outcome", "")),
        "fact_state": str(payload.get("fact_state", "asserted")),
        "evidence_ref": str(refs[0]) if refs else "",
    }


def _build_brief_from_records(query: str, records: Sequence[Dict[str, Any]], *, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    qtok = _tokenize(query)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in records:
        text = " ".join(
            [
                str(row.get("event_type", "")),
                str(row.get("claim", "")),
                str(row.get("decision", "")),
                str(row.get("outcome", "")),
                str(row.get("fact_state", "")),
            ]
        )
        score = _jaccard(qtok, _tokenize(text))
        scored.append((score, row))
    scored.sort(key=lambda it: it[0], reverse=True)
    top = [row for _, row in scored[:24]]

    locked_decisions: List[str] = []
    recent_verified_facts: List[str] = []
    active_risks: List[str] = []
    open_items: List[str] = []
    citations: Dict[str, List[str]] = {}

    for row in top:
        claim = str(row.get("claim", "")).strip()
        decision = str(row.get("decision", "")).strip()
        fact_state = str(row.get("fact_state", "asserted")).strip().lower()
        ref = str(row.get("evidence_ref", "")).strip() or f"row:{row.get('id', 'na')}"

        if decision:
            line = f"{decision} [ref:{ref}]"
            locked_decisions.append(line)
            citations[line] = [ref]

        if claim:
            line = f"{claim} [ref:{ref}]"
            if fact_state == "verified":
                recent_verified_facts.append(line)
            elif fact_state == "contradicted":
                active_risks.append(line)
            else:
                open_items.append(line)
            citations[line] = [ref]

    return {
        "schema_version": "v1",
        "thread_id": "bench_thread",
        "query": query,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "token_budget": 1800,
        "current_goal": [],
        "locked_decisions": locked_decisions[:24],
        "recent_verified_facts": recent_verified_facts[:24],
        "active_risks": active_risks[:24],
        "open_items": open_items[:24],
        "next_actions": [],
        "citations": citations,
        "metadata": dict(metadata or {}),
    }


def _handoff_from_brief(brief: Dict[str, Any], scope: str, *, packet_id_prefix: str, checksum_prefix: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    claims = [
        {"claim": c, "evidence_refs": brief.get("citations", {}).get(c, []), "status": "asserted"}
        for c in brief.get("open_items", [])[:24]
    ]
    payload = {
        "schema_version": "v1",
        "packet_id": f"{packet_id_prefix}_{int(time.time() * 1000)}",
        "role": "auditor",
        "thread_id": "bench_thread",
        "scope": scope,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "locked_decisions": brief.get("locked_decisions", []),
        "claims": claims,
        "unresolved_contradictions": brief.get("active_risks", []),
        "required_validations": [f"Validate scope: {scope}"],
        "evidence_refs": sorted({r for c in claims for r in c.get("evidence_refs", [])}),
        "source_snapshot": {"runner": checksum_prefix},
        "metadata": dict(metadata or {}),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    payload["checksum"] = f"{checksum_prefix}_{digest}"
    return payload


def _extract_claim_rows(section_values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for row in section_values:
        if isinstance(row, str) and row.strip():
            out.append(row.strip())
    return out


def _brief_claim_citation_coverage(brief: Dict[str, Any]) -> Tuple[int, int]:
    citations = brief.get("citations", {}) if isinstance(brief.get("citations", {}), dict) else {}
    claims: List[str] = []
    for key in [
        "current_goal",
        "locked_decisions",
        "recent_verified_facts",
        "active_risks",
        "open_items",
        "next_actions",
    ]:
        claims.extend(_extract_claim_rows(brief.get(key, []) if isinstance(brief.get(key, []), list) else []))
    total = len(claims)
    covered = 0
    for claim in claims:
        refs = citations.get(claim, []) if isinstance(citations, dict) else []
        if isinstance(refs, list) and len(refs) > 0:
            covered += 1
    return total, covered


def _handoff_complete(packet: Dict[str, Any]) -> bool:
    if not isinstance(packet, dict):
        return False
    has_scope = bool(str(packet.get("scope", "")).strip())
    has_role = bool(str(packet.get("role", "")).strip())
    has_thread = bool(str(packet.get("thread_id", "")).strip())
    claims = packet.get("claims", []) if isinstance(packet.get("claims", []), list) else []
    locked_decisions = packet.get("locked_decisions", []) if isinstance(packet.get("locked_decisions", []), list) else []
    required_validations = packet.get("required_validations", []) if isinstance(packet.get("required_validations", []), list) else []
    unresolved = packet.get("unresolved_contradictions", []) if isinstance(packet.get("unresolved_contradictions", []), list) else []
    has_substance = bool(len(claims) > 0 or len(locked_decisions) > 0 or len(required_validations) > 0 or len(unresolved) > 0)
    return bool(has_scope and has_role and has_thread and has_substance)


class Runner(Protocol):
    name: str

    def ingest(self, event: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def brief(self, query: str) -> Dict[str, Any]:
        ...

    def handoff(self, scope: str) -> Dict[str, Any]:
        ...

    def snapshot(self) -> Dict[str, Any]:
        ...

    def close(self) -> None:
        ...


class EvidenceSpineRunner:
    def __init__(self, *, base_dir: Path, mode: str) -> None:
        self.name = f"evidencespine_{mode}"
        settings = EvidenceSpineSettings.from_env(base_dir=str(base_dir))
        settings.retrieval_mode = mode
        settings.retrieval_lexical_weight = 1.0
        settings.retrieval_vector_weight = 0.35
        self.runtime = AgentMemoryRuntime(config=settings.to_runtime_config())

    def ingest(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self.runtime.ingest_event(event))

    def brief(self, query: str) -> Dict[str, Any]:
        return self.runtime.build_brief("bench_thread", query).to_dict()

    def handoff(self, scope: str) -> Dict[str, Any]:
        return self.runtime.emit_handoff("auditor", "bench_thread", scope=scope).to_dict()

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.runtime.snapshot())

    def close(self) -> None:
        self.runtime.flush()


class SQLiteBaselineRunner:
    def __init__(self, *, db_path: Path) -> None:
        self.name = "baseline_sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                event_type TEXT,
                role TEXT,
                source_agent_id TEXT,
                source_turn_id TEXT,
                claim TEXT,
                decision TEXT,
                outcome TEXT,
                evidence_ref TEXT,
                ts REAL
            )
            """
        )
        self.conn.commit()

    def ingest(self, event: Dict[str, Any]) -> Dict[str, Any]:
        payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
        refs = event.get("evidence_refs", []) if isinstance(event.get("evidence_refs", []), list) else []
        self.conn.execute(
            "INSERT INTO events(thread_id,event_type,role,source_agent_id,source_turn_id,claim,decision,outcome,evidence_ref,ts) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                str(event.get("thread_id", "bench_thread")),
                str(event.get("event_type", "reflection")),
                str(event.get("role", "unknown")),
                str(event.get("source_agent_id", "baseline")),
                str(event.get("source_turn_id", "")),
                str(payload.get("claim", "")),
                str(payload.get("decision", "")),
                str(payload.get("outcome", "")),
                str(refs[0]) if refs else "",
                float(time.time()),
            ),
        )
        self.conn.commit()
        return {"status": "ok"}

    def _top_rows(self, query: str, k: int = 24) -> List[Dict[str, Any]]:
        qtok = _tokenize(query)
        rows = self.conn.execute(
            "SELECT id,event_type,claim,decision,outcome,evidence_ref,ts FROM events WHERE thread_id=? ORDER BY id DESC LIMIT 500",
            ("bench_thread",),
        ).fetchall()
        scored: List[Tuple[float, sqlite3.Row]] = []
        for row in rows:
            text = f"{row[1]} {row[2]} {row[3]} {row[4]}"
            score = _jaccard(qtok, _tokenize(text))
            scored.append((score, row))
        scored.sort(key=lambda it: (it[0], it[1][0]), reverse=True)
        out: List[Dict[str, Any]] = []
        for _, row in scored[:k]:
            out.append(
                {
                    "id": row[0],
                    "event_type": row[1],
                    "claim": row[2],
                    "decision": row[3],
                    "outcome": row[4],
                    "evidence_ref": row[5],
                    "ts": row[6],
                }
            )
        return out

    def brief(self, query: str) -> Dict[str, Any]:
        top = self._top_rows(query)
        locked_decisions: List[str] = []
        open_items: List[str] = []
        citations: Dict[str, List[str]] = {}
        for row in top:
            decision = str(row.get("decision", "")).strip()
            claim = str(row.get("claim", "")).strip()
            ref = str(row.get("evidence_ref", "")).strip() or f"sqlite:{row.get('id')}"
            if decision:
                line = f"{decision} [ref:{ref}]"
                locked_decisions.append(line)
                citations[line] = [ref]
            if claim:
                line = f"{claim} [ref:{ref}]"
                open_items.append(line)
                citations[line] = [ref]

        return {
            "schema_version": "v1",
            "thread_id": "bench_thread",
            "query": query,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "token_budget": 1800,
            "current_goal": [],
            "locked_decisions": locked_decisions[:24],
            "recent_verified_facts": [],
            "active_risks": [],
            "open_items": open_items[:24],
            "next_actions": [],
            "citations": citations,
            "metadata": {"baseline": "sqlite"},
        }

    def handoff(self, scope: str) -> Dict[str, Any]:
        brief = self.brief(f"handoff {scope}")
        return _handoff_from_brief(
            brief,
            scope,
            packet_id_prefix="sqlite",
            checksum_prefix="baseline_sqlite",
            metadata={"baseline": "sqlite"},
        )

    def snapshot(self) -> Dict[str, Any]:
        events_total = int(self.conn.execute("SELECT COUNT(1) FROM events").fetchone()[0])
        return {
            "enabled": True,
            "events_total": events_total,
            "facts_total": 0,
            "agent_memory_events_24h": events_total,
            "agent_memory_verified_facts_24h": 0,
            "agent_memory_contradiction_count_24h": 0,
            "agent_memory_unresolved_contradiction_ratio_24h": 0.0,
            "agent_brief_generation_success_rate_24h": 1.0,
            "agent_brief_stale_rate_24h": 0.0,
            "agent_handoff_packets_emitted_24h": 0,
            "agent_handoff_packet_completeness_24h": 0.0,
            "agent_claim_citation_coverage_24h": 0.0,
            "agent_memory_fail_open_events_24h": 0,
        }

    def close(self) -> None:
        self.conn.close()


class SkippedRunner:
    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    def ingest(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "skipped", "reason": self.reason}

    def brief(self, query: str) -> Dict[str, Any]:
        return {}

    def handoff(self, scope: str) -> Dict[str, Any]:
        return {}

    def snapshot(self) -> Dict[str, Any]:
        return {}

    def close(self) -> None:
        return None


class _DeterministicEmbedder:
    def __init__(self, dims: int = 1536) -> None:
        self.dims = int(max(64, dims))

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dims
        tokens = _tokenize(text)
        if not tokens:
            return vec
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self.dims
            vec[idx] += 1.0
        scale = max(1.0, float(len(tokens)))
        return [v / scale for v in vec]


class Mem0Runner:
    def __init__(self, *, base_dir: Path) -> None:
        self.name = "mem0"
        os.environ.setdefault("OPENAI_API_KEY", "benchmark_dummy_key")
        from mem0 import Memory

        cfg = {
            "vector_store": {
                "provider": "qdrant",
                "config": {"path": str(base_dir / "qdrant")},
            },
            "collection_name": "bench_mem0",
            "history_db_path": str(base_dir / "mem0_history.db"),
        }
        self.memory = Memory.from_config(cfg)
        self.memory.embedding_model = _DeterministicEmbedder(dims=1536)
        self._rows: List[Dict[str, Any]] = []
        self._search_fallback_total = 0
        self._events_total = 0
        self._verified_total = 0
        self._contradiction_total = 0

    def ingest(self, event: Dict[str, Any]) -> Dict[str, Any]:
        row = _event_to_payload(event)
        self._rows.append(dict(row))
        doc = json.dumps(row, sort_keys=True, ensure_ascii=True)
        meta = {
            "thread_id": row["thread_id"],
            "event_type": row["event_type"],
            "role": row["role"],
            "source_turn_id": row["source_turn_id"],
            "fact_state": row["fact_state"],
            "evidence_ref": row["evidence_ref"],
        }
        memory_id = self.memory._create_memory_tool(doc, metadata=meta)
        self._events_total += 1
        if row["fact_state"] == "verified":
            self._verified_total += 1
        if row["fact_state"] == "contradicted":
            self._contradiction_total += 1
        return {"status": "ok", "id": memory_id}

    def brief(self, query: str) -> Dict[str, Any]:
        try:
            found = self.memory.search(query, limit=24)
            rows: List[Dict[str, Any]] = []
            for item in found:
                text = str(item.get("text", ""))
                meta = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
                row: Dict[str, Any]
                try:
                    row = json.loads(text)
                    if not isinstance(row, dict):
                        row = {}
                except Exception:
                    row = {}
                row.setdefault("claim", row.get("claim", text))
                row.setdefault("decision", row.get("decision", ""))
                row.setdefault("outcome", row.get("outcome", ""))
                row.setdefault("event_type", row.get("event_type", str(meta.get("event_type", ""))))
                row.setdefault("fact_state", str(meta.get("fact_state", row.get("fact_state", "asserted"))))
                row.setdefault("evidence_ref", str(meta.get("evidence_ref", row.get("evidence_ref", ""))))
                rows.append(row)
            return _build_brief_from_records(query, rows, metadata={"baseline": "mem0_local"})
        except Exception:
            # Compatibility fallback for external API drift.
            self._search_fallback_total += 1
            return _build_brief_from_records(
                query,
                self._rows,
                metadata={"baseline": "mem0_local", "search_fallback": True},
            )

    def handoff(self, scope: str) -> Dict[str, Any]:
        brief = self.brief(f"handoff {scope}")
        return _handoff_from_brief(
            brief,
            scope,
            packet_id_prefix="mem0",
            checksum_prefix="mem0",
            metadata={"baseline": "mem0_local"},
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "events_total": self._events_total,
            "facts_total": self._events_total,
            "agent_memory_events_24h": self._events_total,
            "agent_memory_verified_facts_24h": self._verified_total,
            "agent_memory_contradiction_count_24h": self._contradiction_total,
            "agent_memory_unresolved_contradiction_ratio_24h": float(
                self._contradiction_total / max(1, self._events_total)
            ),
            "agent_brief_generation_success_rate_24h": 1.0,
            "agent_brief_stale_rate_24h": 0.0,
            "agent_handoff_packets_emitted_24h": 0,
            "agent_handoff_packet_completeness_24h": 0.0,
            "agent_claim_citation_coverage_24h": 0.0,
            "agent_memory_fail_open_events_24h": 0,
            "mem0_search_fallback_total": self._search_fallback_total,
        }

    def close(self) -> None:
        try:
            self.memory.reset()
        except Exception:
            pass


class LettaRunner:
    def __init__(self, *, base_dir: Path) -> None:
        self.name = "letta"
        self._events_total = 0
        self._verified_total = 0
        self._contradiction_total = 0
        self._home = base_dir / "letta_home"
        self._home.mkdir(parents=True, exist_ok=True)
        self._port = self._find_free_port()
        self._proc = self._start_server()
        from letta import RESTClient

        self.client = RESTClient(base_url=f"http://127.0.0.1:{self._port}")
        self._wait_ready()

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def _start_server(self) -> subprocess.Popen:
        env = os.environ.copy()
        env["LETTA_DIR"] = str(self._home)
        letta_bin = Path(sys.executable).with_name("letta")
        if letta_bin.exists():
            cmd = [str(letta_bin), "server", "--host", "127.0.0.1", "--port", str(self._port)]
        else:
            cmd = [sys.executable, "-m", "letta.cli.cli", "server", "--host", "127.0.0.1", "--port", str(self._port)]
        return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _wait_ready(self, timeout_sec: float = 25.0) -> None:
        t0 = time.time()
        last_err: Optional[Exception] = None
        while time.time() - t0 < timeout_sec:
            if self._proc.poll() is not None:
                raise RuntimeError(f"letta_server_exited_{self._proc.returncode}")
            try:
                self.client.list_blocks(templates_only=False)
                return
            except Exception as exc:  # pragma: no cover - startup race
                last_err = exc
                time.sleep(0.25)
        raise RuntimeError(f"letta_server_not_ready: {last_err}")

    def _list_blocks(self, *, limit: int = 5000, label: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "templates_only": "false",
            "limit": int(max(1, limit)),
        }
        if label:
            params["label"] = str(label)
        resp = requests.get(
            f"{self.client.base_url}/{self.client.api_prefix}/blocks/",
            params=params,
            headers=getattr(self.client, "headers", {}),
            timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"letta_blocks_http_{resp.status_code}")
        data = resp.json()
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    def ingest(self, event: Dict[str, Any]) -> Dict[str, Any]:
        row = _event_to_payload(event)
        label = f"bench_evt_{row.get('source_turn_id', _rand_token(6))[:64]}"
        value = json.dumps(row, sort_keys=True, ensure_ascii=True)
        block = self.client.create_block(label=label, value=value, limit=4000)
        self._events_total += 1
        if row["fact_state"] == "verified":
            self._verified_total += 1
        if row["fact_state"] == "contradicted":
            self._contradiction_total += 1
        return {"status": "ok", "id": getattr(block, "id", "")}

    def brief(self, query: str) -> Dict[str, Any]:
        try:
            blocks = self._list_blocks(limit=max(5000, self._events_total + 32))
        except Exception:
            blocks = []
            try:
                # Fallback to legacy client behavior (defaults to limit=50).
                legacy = self.client.list_blocks(templates_only=False)
                for b in legacy:
                    blocks.append(
                        {
                            "id": getattr(b, "id", ""),
                            "label": getattr(b, "label", ""),
                            "value": getattr(b, "value", ""),
                        }
                    )
            except Exception:
                pass
        rows: List[Dict[str, Any]] = []
        for b in blocks:
            label = str(b.get("label", "") if isinstance(b, dict) else "")
            if not label.startswith("bench_evt_"):
                continue
            raw = str(b.get("value", "") if isinstance(b, dict) else "")
            try:
                row = json.loads(raw)
            except Exception:
                row = {"claim": raw}
            if not isinstance(row, dict):
                row = {"claim": raw}
            row.setdefault("evidence_ref", f"letta:block:{str(b.get('id', 'na'))}")
            rows.append(row)
        return _build_brief_from_records(query, rows, metadata={"baseline": "letta_local"})

    def handoff(self, scope: str) -> Dict[str, Any]:
        brief = self.brief(f"handoff {scope}")
        return _handoff_from_brief(
            brief,
            scope,
            packet_id_prefix="letta",
            checksum_prefix="letta",
            metadata={"baseline": "letta_local"},
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "events_total": self._events_total,
            "facts_total": self._events_total,
            "agent_memory_events_24h": self._events_total,
            "agent_memory_verified_facts_24h": self._verified_total,
            "agent_memory_contradiction_count_24h": self._contradiction_total,
            "agent_memory_unresolved_contradiction_ratio_24h": float(
                self._contradiction_total / max(1, self._events_total)
            ),
            "agent_brief_generation_success_rate_24h": 1.0,
            "agent_brief_stale_rate_24h": 0.0,
            "agent_handoff_packets_emitted_24h": 0,
            "agent_handoff_packet_completeness_24h": 0.0,
            "agent_claim_citation_coverage_24h": 0.0,
            "agent_memory_fail_open_events_24h": 0,
        }

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=3.0)


@dataclass
class RunnerMetrics:
    runner: str
    status: str
    reason: str = ""
    ingest_total_ms: float = 0.0
    ingest_eps: float = 0.0
    ingest_p50_ms: float = 0.0
    ingest_p95_ms: float = 0.0
    brief_total_ms: float = 0.0
    brief_qps: float = 0.0
    brief_p50_ms: float = 0.0
    brief_p95_ms: float = 0.0
    handoff_total_ms: float = 0.0
    handoff_qps: float = 0.0
    handoff_p50_ms: float = 0.0
    handoff_p95_ms: float = 0.0
    brief_claim_citation_coverage: float = 0.0
    handoff_completeness_rate: float = 0.0
    handoff_checksum_rate: float = 0.0
    verified_probe_hit: float = 0.0
    contradiction_probe_hit: float = 0.0
    governance_score: float = 0.0
    snapshot: Dict[str, Any] = field(default_factory=dict)


def _build_runner(name: str, root: Path) -> Runner:
    n = name.strip().lower()
    if n == "evidencespine_lexical":
        return EvidenceSpineRunner(base_dir=root / n, mode="lexical")
    if n == "evidencespine_hybrid":
        return EvidenceSpineRunner(base_dir=root / n, mode="hybrid")
    if n == "evidencespine_vector":
        return EvidenceSpineRunner(base_dir=root / n, mode="vector")
    if n == "baseline_sqlite":
        return SQLiteBaselineRunner(db_path=root / n / "baseline.db")
    if n == "mem0":
        try:
            __import__("mem0")
            return Mem0Runner(base_dir=root / n)
        except Exception as exc:
            reason = str(exc).strip().lower().replace(" ", "_")
            return SkippedRunner("mem0", f"mem0_init_failed:{reason[:160]}")
    if n == "letta":
        try:
            __import__("letta")
            return LettaRunner(base_dir=root / n)
        except Exception as exc:
            reason = str(exc).strip().lower().replace(" ", "_")
            return SkippedRunner("letta", f"letta_init_failed:{reason[:160]}")
    return SkippedRunner(n, "unknown_runner")


def _evaluate_runner(runner: Runner, *, events: int, queries: int, handoffs: int) -> RunnerMetrics:
    if isinstance(runner, SkippedRunner):
        return RunnerMetrics(runner=runner.name, status="skipped", reason=runner.reason)

    ingest_lat: List[float] = []
    t0 = time.perf_counter()
    for i in range(events):
        event = _mk_event(i)
        s = time.perf_counter()
        runner.ingest(event)
        ingest_lat.append((time.perf_counter() - s) * 1000.0)
    ingest_total_ms = (time.perf_counter() - t0) * 1000.0

    brief_lat: List[float] = []
    brief_claim_total = 0
    brief_claim_covered = 0
    t1 = time.perf_counter()
    for i in range(queries):
        s = time.perf_counter()
        b = runner.brief(_mk_query(i))
        brief_lat.append((time.perf_counter() - s) * 1000.0)
        total, covered = _brief_claim_citation_coverage(b if isinstance(b, dict) else {})
        brief_claim_total += total
        brief_claim_covered += covered
    brief_total_ms = (time.perf_counter() - t1) * 1000.0

    handoff_lat: List[float] = []
    handoff_complete = 0
    handoff_checksum = 0
    t2 = time.perf_counter()
    for i in range(handoffs):
        s = time.perf_counter()
        p = runner.handoff(f"verify wave {i}")
        handoff_lat.append((time.perf_counter() - s) * 1000.0)
        p_dict = p if isinstance(p, dict) else {}
        if _handoff_complete(p_dict):
            handoff_complete += 1
        checksum = str(p_dict.get("checksum", "")).strip()
        if checksum and checksum != "baseline_sqlite_no_checksum":
            handoff_checksum += 1
    handoff_total_ms = (time.perf_counter() - t2) * 1000.0

    # Probe checks: verify whether runners preserve semantic state quality.
    verified_probe = "probe_verified_claim_token"
    contradicted_probe = "probe_contradicted_claim_token"
    runner.ingest(
        {
            "thread_id": "bench_thread",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "bench_probe",
            "source_turn_id": "probe_verified",
            "payload": {
                "claim": verified_probe,
                "fact_state": "verified",
            },
            "evidence_refs": ["bench/probe_verified.md#L1"],
            "confidence": 0.8,
            "salience": 0.8,
        }
    )
    runner.ingest(
        {
            "thread_id": "bench_thread",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "bench_probe",
            "source_turn_id": "probe_contradicted",
            "payload": {
                "claim": contradicted_probe,
                "fact_state": "contradicted",
            },
            "evidence_refs": ["bench/probe_contradicted.md#L1"],
            "confidence": 0.8,
            "salience": 0.8,
        }
    )
    probe_brief = runner.brief(f"{verified_probe} {contradicted_probe}")
    p_recent_verified = probe_brief.get("recent_verified_facts", []) if isinstance(probe_brief, dict) else []
    p_active_risks = probe_brief.get("active_risks", []) if isinstance(probe_brief, dict) else []
    verified_hit = 1.0 if any(verified_probe in str(x) for x in p_recent_verified) else 0.0
    contradicted_hit = 1.0 if any(contradicted_probe in str(x) for x in p_active_risks) else 0.0

    snap = runner.snapshot() if hasattr(runner, "snapshot") else {}
    runner.close()

    citation_cov = (float(brief_claim_covered / max(1, brief_claim_total)) if brief_claim_total > 0 else 0.0)
    handoff_comp = float(handoff_complete / max(1, handoffs))
    checksum_rate = float(handoff_checksum / max(1, handoffs))
    governance_score = float(
        (citation_cov + handoff_comp + checksum_rate + verified_hit + contradicted_hit) / 5.0
    )

    return RunnerMetrics(
        runner=getattr(runner, "name", "runner"),
        status="ok",
        ingest_total_ms=ingest_total_ms,
        ingest_eps=float(events / max(0.001, ingest_total_ms / 1000.0)),
        ingest_p50_ms=_q(ingest_lat, 0.50),
        ingest_p95_ms=_q(ingest_lat, 0.95),
        brief_total_ms=brief_total_ms,
        brief_qps=float(queries / max(0.001, brief_total_ms / 1000.0)),
        brief_p50_ms=_q(brief_lat, 0.50),
        brief_p95_ms=_q(brief_lat, 0.95),
        handoff_total_ms=handoff_total_ms,
        handoff_qps=float(handoffs / max(0.001, handoff_total_ms / 1000.0)),
        handoff_p50_ms=_q(handoff_lat, 0.50),
        handoff_p95_ms=_q(handoff_lat, 0.95),
        brief_claim_citation_coverage=citation_cov,
        handoff_completeness_rate=handoff_comp,
        handoff_checksum_rate=checksum_rate,
        verified_probe_hit=verified_hit,
        contradiction_probe_hit=contradicted_hit,
        governance_score=governance_score,
        snapshot=dict(snap) if isinstance(snap, dict) else {},
    )


def _write_markdown(payload: Dict[str, Any], path: Path) -> None:
    rows = payload.get("results", []) if isinstance(payload.get("results", []), list) else []
    lines: List[str] = []
    lines.append("# Apples-to-Apples Comparison")
    lines.append("")
    params = payload.get("params", {}) if isinstance(payload.get("params", {}), dict) else {}
    lines.append(f"- events: {params.get('events')}")
    lines.append(f"- queries: {params.get('queries')}")
    lines.append(f"- handoffs: {params.get('handoffs')}")
    lines.append(f"- seed: {params.get('seed')}")
    lines.append("")
    lines.append("| runner | status | ingest eps | brief qps | handoff qps | brief citation coverage | handoff completeness |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {runner} | {status} | {ingest_eps:.2f} | {brief_qps:.2f} | {handoff_qps:.2f} | {bcc:.3f} | {hcr:.3f} |".format(
                runner=str(row.get("runner", "")),
                status=str(row.get("status", "")),
                ingest_eps=float(row.get("ingest_eps", 0.0) or 0.0),
                brief_qps=float(row.get("brief_qps", 0.0) or 0.0),
                handoff_qps=float(row.get("handoff_qps", 0.0) or 0.0),
                bcc=float(row.get("brief_claim_citation_coverage", 0.0) or 0.0),
                hcr=float(row.get("handoff_completeness_rate", 0.0) or 0.0),
            )
        )
    lines.append("")
    lines.append("| runner | checksum rate | verified probe hit | contradiction probe hit | governance score |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {runner} | {csr:.3f} | {vph:.3f} | {cph:.3f} | {gs:.3f} |".format(
                runner=str(row.get("runner", "")),
                csr=float(row.get("handoff_checksum_rate", 0.0) or 0.0),
                vph=float(row.get("verified_probe_hit", 0.0) or 0.0),
                cph=float(row.get("contradiction_probe_hit", 0.0) or 0.0),
                gs=float(row.get("governance_score", 0.0) or 0.0),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apples-to-apples benchmark comparison")
    parser.add_argument("--events", type=int, default=1200)
    parser.add_argument("--queries", type=int, default=80)
    parser.add_argument("--handoffs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--runners",
        default="evidencespine_lexical,evidencespine_hybrid,baseline_sqlite,mem0,letta",
        help="comma-separated runners",
    )
    parser.add_argument("--out-json", default="benchmarks/results/apples_to_apples.json")
    parser.add_argument("--out-md", default="benchmarks/results/apples_to_apples.md")
    args = parser.parse_args()

    random.seed(int(args.seed))
    run_ts = int(time.time())
    root = Path("benchmarks") / "runs" / f"apples_{run_ts}"
    root.mkdir(parents=True, exist_ok=True)

    requested = [x.strip().lower() for x in str(args.runners).split(",") if x.strip()]

    results: List[RunnerMetrics] = []
    for name in requested:
        runner = _build_runner(name, root)
        results.append(
            _evaluate_runner(
                runner,
                events=int(max(100, args.events)),
                queries=int(max(10, args.queries)),
                handoffs=int(max(5, args.handoffs)),
            )
        )

    payload: Dict[str, Any] = {
        "benchmark": "apples_to_apples_v1",
        "timestamp": run_ts,
        "params": {
            "events": int(args.events),
            "queries": int(args.queries),
            "handoffs": int(args.handoffs),
            "seed": int(args.seed),
            "runners": requested,
        },
        "results": [asdict(x) for x in results],
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    out_md = Path(args.out_md)
    _write_markdown(payload, out_md)

    print(json.dumps(payload, indent=2, ensure_ascii=True))
    print(f"markdown_report={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
