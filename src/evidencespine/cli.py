from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _build_runtime(base_dir: str | None) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=base_dir)
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _coerce_evidence_item_payload(raw: Any, *, source: str) -> list[Dict[str, Any]]:
    if isinstance(raw, dict):
        return [dict(raw)]
    if isinstance(raw, list):
        out: list[Dict[str, Any]] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"{source}[{idx}] must be a JSON object")
            out.append(dict(item))
        return out
    raise ValueError(f"{source} must be a JSON object or array of objects")


def _load_evidence_items(args: argparse.Namespace) -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    for idx, raw in enumerate(list(args.evidence_item_json or [])):
        try:
            parsed = json.loads(str(raw))
        except Exception as exc:
            raise ValueError(f"--evidence-item-json[{idx}] is not valid JSON: {exc}") from exc
        items.extend(_coerce_evidence_item_payload(parsed, source=f"--evidence-item-json[{idx}]"))
    for path in list(args.evidence_item_file or []):
        with open(str(path), "r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        items.extend(_coerce_evidence_item_payload(parsed, source=str(path)))
    return items


def _build_state_context(args: argparse.Namespace) -> Dict[str, Any] | None:
    payload: Dict[str, Any] = {}
    mapping = {
        "scope_id": "scope_id",
        "scope_kind": "scope_kind",
        "state_kind": "state_kind",
        "status": "status",
        "owner_agent_id": "owner_agent_id",
        "state_basis": "state_basis",
        "validated_at": "validated_at",
        "validated_by": "validated_by",
        "fresh_until": "fresh_until",
        "lease_expires_at": "lease_expires_at",
        "supersedes": "supersedes",
    }
    for arg_name, key in mapping.items():
        value = getattr(args, arg_name, "")
        text = str(value).strip()
        if text:
            payload[key] = text
    return (payload or None)


def _cmd_ingest(args: argparse.Namespace) -> int:
    runtime = _build_runtime(args.base_dir)
    payload: Dict[str, Any] = {}
    if str(args.objective_id).strip():
        payload["objective_id"] = str(args.objective_id).strip()
    if str(args.claim).strip():
        payload["claim"] = str(args.claim).strip()
    if str(args.decision).strip():
        payload["decision"] = str(args.decision).strip()
    if str(args.outcome).strip():
        payload["outcome"] = str(args.outcome).strip()
    if str(args.target).strip():
        payload["target"] = str(args.target).strip()
    if str(args.fact_state).strip():
        payload["fact_state"] = str(args.fact_state).strip().lower()
    next_actions = [str(x).strip() for x in list(args.next_action or []) if str(x).strip()]
    if next_actions:
        payload["next_actions"] = next_actions
    try:
        evidence_items = _load_evidence_items(args)
    except ValueError as exc:
        if bool(args.json):
            print(json.dumps({"status": "invalid_input", "reason": str(exc)}, indent=2, ensure_ascii=True))
        else:
            print(str(exc))
        return 2

    out = runtime.ingest_event(
        {
            "thread_id": str(args.thread_id),
            "event_type": str(args.event_type),
            "role": str(args.role),
            "source_agent_id": str(args.source_agent_id),
            "source_turn_id": str(args.source_turn_id),
            "payload": payload,
            "evidence_refs": [str(x).strip() for x in list(args.evidence_ref or []) if str(x).strip()],
            "evidence_items": evidence_items,
            "state_context": _build_state_context(args),
            "confidence": float(args.confidence),
            "salience": float(args.salience),
        }
    )
    if bool(args.json):
        print(json.dumps(out, indent=2, ensure_ascii=True))
    else:
        print(out)
    return 0


def _print_brief(payload: Dict[str, Any]) -> None:
    print("Agent Context Brief")
    print("===================")
    print(f"thread_id: {payload.get('thread_id', '')}")
    print(f"query: {payload.get('query', '')}")
    print(f"generated_at: {payload.get('generated_at', '')}")
    print(f"token_budget: {payload.get('token_budget', 0)}")
    print("")

    for key in [
        "current_goal",
        "locked_decisions",
        "recent_verified_facts",
        "active_risks",
        "open_items",
        "next_actions",
    ]:
        print(f"[{key}]")
        rows = payload.get(key, []) if isinstance(payload.get(key, []), list) else []
        if not rows:
            print("- none")
        for row in rows:
            print(f"- {row}")
        print("")


def _cmd_brief(args: argparse.Namespace) -> int:
    runtime = _build_runtime(args.base_dir)
    budget = int(args.token_budget) if int(args.token_budget) > 0 else None
    brief = runtime.build_brief(thread_id=str(args.thread_id), query=str(args.query), token_budget=budget)
    payload = brief.to_dict()
    if bool(args.json):
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        _print_brief(payload)
    return 0


def _cmd_handoff(args: argparse.Namespace) -> int:
    runtime = _build_runtime(args.base_dir)
    if str(args.import_path or "").strip():
        out = runtime.import_handoff(str(args.import_path), source_agent_id="external_agent")
        print(json.dumps(out, indent=2, ensure_ascii=True))
        return 0

    packet = runtime.emit_handoff(role=str(args.role), thread_id=str(args.thread_id), scope=str(args.scope))
    payload = packet.to_dict()
    if args.output:
        with open(str(args.output), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)

    if bool(args.json) or (not args.output):
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(args.output)
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    runtime = _build_runtime(args.base_dir)
    payload = runtime.snapshot()
    if bool(args.json):
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        for key in sorted(payload.keys()):
            print(f"{key}={payload[key]}")
    return 0


def _print_view(payload: Dict[str, Any]) -> None:
    print("scope_id\tstate_kind\tstatus\towner\tfreshness\tconflict\tclaim")
    for row in payload.get("rows", []) if isinstance(payload.get("rows", []), list) else []:
        if not isinstance(row, dict):
            continue
        print(
            "\t".join(
                [
                    str(row.get("scope_id", "")),
                    str(row.get("state_kind", "")),
                    str(row.get("status", "")),
                    str(row.get("owner_agent_id", "")),
                    str(row.get("freshness_state", "")),
                    str(bool(row.get("conflict", False))).lower(),
                    str(row.get("claim", "")),
                ]
            )
        )


def _cmd_view(args: argparse.Namespace) -> int:
    runtime = _build_runtime(args.base_dir)
    payload = runtime.query_view(
        str(args.view_name).replace("-", "_"),
        thread_id=str(args.thread_id or ""),
        owner_agent_id=str(args.owner_agent_id or ""),
        include_closed=bool(args.include_closed),
        limit=int(args.limit),
    ).to_dict()
    if bool(args.json):
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        _print_view(payload)
    return 0


def _cmd_reconcile(args: argparse.Namespace) -> int:
    runtime = _build_runtime(args.base_dir)
    payload = runtime.reconcile(thread_id=str(args.thread_id), limit=int(args.limit))
    if bool(args.json):
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        for key in sorted(payload.keys()):
            print(f"{key}={payload[key]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EvidenceSpine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest one structured memory event")
    ingest.add_argument("--base-dir", default=None)
    ingest.add_argument("--thread-id", required=True)
    ingest.add_argument("--event-type", required=True, choices=["intent", "decision", "action", "outcome", "reflection"])
    ingest.add_argument("--role", default="operator")
    ingest.add_argument("--source-agent-id", default="external_agent")
    ingest.add_argument("--source-turn-id", default="")
    ingest.add_argument("--objective-id", default="")
    ingest.add_argument("--claim", default="")
    ingest.add_argument("--decision", default="")
    ingest.add_argument("--outcome", default="")
    ingest.add_argument("--target", default="")
    ingest.add_argument("--fact-state", default="", choices=["", "asserted", "verified", "contradicted", "superseded"])
    ingest.add_argument("--next-action", action="append", default=[])
    ingest.add_argument("--evidence-ref", action="append", default=[])
    ingest.add_argument("--evidence-item-json", action="append", default=[])
    ingest.add_argument("--evidence-item-file", action="append", default=[])
    ingest.add_argument("--scope-id", default="")
    ingest.add_argument("--scope-kind", default="", choices=["", "task", "gate", "blocker", "runtime_state", "thread"])
    ingest.add_argument(
        "--state-kind",
        default="",
        choices=["", "agent_local_work", "global_blocker", "pending_gate", "runtime_validated_state"],
    )
    ingest.add_argument("--status", default="", choices=["", "active", "blocked", "ready", "closed", "superseded"])
    ingest.add_argument("--owner-agent-id", default="")
    ingest.add_argument("--state-basis", default="", choices=["", "reported", "runtime_validated", "derived", "imported"])
    ingest.add_argument("--validated-at", default="")
    ingest.add_argument("--validated-by", default="")
    ingest.add_argument("--fresh-until", default="")
    ingest.add_argument("--lease-expires-at", default="")
    ingest.add_argument("--supersedes", default="")
    ingest.add_argument("--confidence", type=float, default=0.6)
    ingest.add_argument("--salience", type=float, default=0.5)
    ingest.add_argument("--json", action="store_true")
    ingest.set_defaults(func=_cmd_ingest)

    brief = sub.add_parser("brief", help="Build bounded context brief")
    brief.add_argument("--base-dir", default=None)
    brief.add_argument("--thread-id", required=True)
    brief.add_argument("--query", default="")
    brief.add_argument("--token-budget", type=int, default=0)
    brief.add_argument("--json", action="store_true")
    brief.set_defaults(func=_cmd_brief)

    handoff = sub.add_parser("handoff", help="Emit/import handoff packet")
    handoff.add_argument("--base-dir", default=None)
    handoff.add_argument("--role", default="auditor")
    handoff.add_argument("--thread-id", default="default")
    handoff.add_argument("--scope", default="cross-agent coordination")
    handoff.add_argument("--output", default="")
    handoff.add_argument("--import", dest="import_path", default="")
    handoff.add_argument("--json", action="store_true")
    handoff.set_defaults(func=_cmd_handoff)

    snapshot = sub.add_parser("snapshot", help="Show memory health snapshot")
    snapshot.add_argument("--base-dir", default=None)
    snapshot.add_argument("--json", action="store_true")
    snapshot.set_defaults(func=_cmd_snapshot)

    view = sub.add_parser("view", help="Show derived agent-state control views")
    view.add_argument(
        "view_name",
        choices=["active-scopes", "my-work", "open-gates", "stale-claims", "contradictions"],
    )
    view.add_argument("--base-dir", default=None)
    view.add_argument("--thread-id", default="")
    view.add_argument("--owner-agent-id", default="")
    view.add_argument("--include-closed", action="store_true")
    view.add_argument("--limit", type=int, default=50)
    view.add_argument("--json", action="store_true")
    view.set_defaults(func=_cmd_view)

    reconcile = sub.add_parser("reconcile", help="Run optional state reconciliation hook")
    reconcile.add_argument("--base-dir", default=None)
    reconcile.add_argument("--thread-id", required=True)
    reconcile.add_argument("--limit", type=int, default=50)
    reconcile.add_argument("--json", action="store_true")
    reconcile.set_defaults(func=_cmd_reconcile)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
