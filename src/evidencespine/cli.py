from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings

#: Process-status helper statuses: payload "status" -> shell exit code.
_OK_STATUSES = {"ok", "deduped", "noop", "verified", "unchanged", "passed", "dry_run"}
_ERROR_STATUSES = {"invalid", "missing", "ungroundable", "ungroundable_refs", "invalid_input", "mismatch", "skipped", "broken", "unknown", "unsupported"}
_FAIL_OPEN_STATUSES = {"fail_open", "disabled", "partial"}


def _status_exit(payload: Any, *, soft: tuple[str, ...] = ()) -> int:
    """Map a command payload status to a process exit code (2 = invalid/missing)."""
    if not isinstance(payload, dict):
        return 0
    status = str(payload.get("status") or "").strip().lower()
    if not status or status in _OK_STATUSES or status in soft:
        return 0
    if status in _ERROR_STATUSES:
        return 2
    if status in _FAIL_OPEN_STATUSES:
        return 3
    return 0


def _resolve_thread(value: Any) -> str:
    """Resolve an optional --thread-id to the workspace thread when omitted."""
    from evidencespine.threads import resolve_thread_id

    return resolve_thread_id(str(value or ""))

def _build_runtime(base_dir: str | None, storage_format: str | None = None) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=base_dir, storage_format=storage_format)
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
        try:
            with open(str(path), "r", encoding="utf-8") as handle:
                parsed = json.load(handle)
        except OSError as exc:
            raise ValueError(f"--evidence-item-file {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"--evidence-item-file {path} is not valid JSON: {exc}") from exc
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


def _resolve_cli_thread(thread_id: Any, session_id: Any = "") -> tuple[str, Dict[str, Any]]:
    """Resolve a CLI thread id; provider session ids travel in metadata only."""
    explicit = str(thread_id or "").strip()
    metadata: Dict[str, Any] = {}
    sid = str(session_id or "").strip()
    if sid:
        metadata["session_id"] = sid
    if explicit:
        return explicit, metadata
    from evidencespine.threads import resolve_thread_id

    return resolve_thread_id("", cwd=os.getcwd()), metadata


def _cmd_verify(args: argparse.Namespace) -> int:
    runtime = _build_runtime(args.base_dir, getattr(args, "storage_format", None))
    out = runtime.verify_fact(
        str(args.fact_id),
        method=str(args.method),
        reference=str(args.reference),
        verified_by=str(args.verified_by),
        thread_id=_resolve_thread(args.thread_id),
    )
    if bool(args.json):
        print(json.dumps(out, indent=2, ensure_ascii=True))
    else:
        print(out)
    return _status_exit(out)


def _cmd_drift_check(args: argparse.Namespace) -> int:
    runtime = _build_runtime(args.base_dir, getattr(args, "storage_format", None))
    out = runtime.check_evidence_stale(
        thread_id=_resolve_thread(args.thread_id),
        source_root=str(args.source_root),
        dry_run=not bool(args.apply),
    )
    if bool(args.json):
        print(json.dumps(out, indent=2, ensure_ascii=True))
    else:
        print(
            f"checked {out.get('checked_items', 0)} grounded items, "
            f"{out.get('stale_facts', 0)} stale facts "
            f"({'dry-run' if out.get('dry_run') else 'applied'})"
        )
        for row in out.get("results", [])[:20]:
            print(f"  [{row.get('reason')}] {row.get('source_id')}: {row.get('claim', '')[:60]}")
    return _status_exit(out)


def _cmd_ground(args: argparse.Namespace) -> int:
    from evidencespine.grounding import ground_ref

    refs = [str(r).strip() for r in list(args.ref or []) if str(r).strip()]
    if not refs:
        print("usage: evidencespine ground <ref> [<ref> ...]  (ref like path#L10-L20)")
        return 2
    items = []
    seen: set[str] = set()
    for ref in refs:
        item = ground_ref(ref, source_root=str(args.source_root))
        if item is None:
            print(json.dumps({"status": "ungroundable", "ref": ref}, ensure_ascii=True))
            return 1
        key = f"{item['source_id']}#L{item['line_start']}-{item['line_end']}"
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    if bool(args.json):
        print(json.dumps({"status": "ok", "items": items, "count": len(items)}, indent=2, ensure_ascii=True))
    else:
        for item in items:
            print(
                f"{item['source_id']}#L{item['line_start']}-{item['line_end']} "
                f"({len(item['excerpt'])} chars, {item['checksum'][:12]}...)")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    runtime = _build_runtime(args.base_dir, getattr(args, "storage_format", None))
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

    ground_refs: List[str] = []
    for raw_ref in list(getattr(args, "ground_ref", None) or []):
        ref = str(raw_ref).strip()
        if ref and ref not in ground_refs:
            ground_refs.append(ref)
    if ground_refs:
        from evidencespine.grounding import ground_ref as _ground_one

        grounded: List[Dict[str, Any]] = []
        ungroundable: List[str] = []
        seen_keys: set[str] = set()
        for ref in ground_refs:
            item = _ground_one(ref, source_root=str(getattr(args, "source_root", ".")))
            if item is None:
                ungroundable.append(ref)
                continue
            key = f"{item['source_id']}#L{item['line_start']}-{item['line_end']}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            grounded.append(item)
        grounded_source_ids = {str(item.get("source_id")) for item in grounded}
        evidence_items = [x for x in evidence_items if str(x.get("source_id")) not in grounded_source_ids]
        evidence_items.extend(grounded)
        if ungroundable:
            if bool(args.json):
                print(json.dumps({"status": "ungroundable_refs", "refs": ungroundable}, indent=2, ensure_ascii=True))
            else:
                print(f"ungroundable refs: {ungroundable}")
            return 1

    thread_id, metadata = _resolve_cli_thread(
        getattr(args, "thread_id", ""),
        getattr(args, "session_id", ""),
    )
    source_turn_id = str(getattr(args, "source_turn_id", "") or "").strip()
    if not source_turn_id and metadata.get("session_id"):
        source_turn_id = str(metadata["session_id"])
    event: Dict[str, Any] = {
        "thread_id": thread_id,
        "event_type": str(args.event_type),
        "role": str(args.role),
        "source_agent_id": str(args.source_agent_id),
        "source_turn_id": source_turn_id,
        "payload": payload,
        "evidence_refs": [str(x).strip() for x in list(args.evidence_ref or []) if str(x).strip()],
        "evidence_items": evidence_items,
        "state_context": _build_state_context(args),
        "confidence": float(args.confidence),
        "salience": float(args.salience),
    }
    if metadata:
        event["metadata"] = metadata
    out = runtime.ingest_event(event)
    if bool(args.json):
        print(json.dumps(out, indent=2, ensure_ascii=True))
    else:
        print(out)
    return _status_exit(out)


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


def _cmd_chat(args: argparse.Namespace) -> int:
    from evidencespine.chat import run_chat

    roles = [r.strip() for r in str(args.roles).split(",") if r.strip()]
    # None (not the string "None") lets settings resolve env/default base dir.
    base_dir = str(args.base_dir) if getattr(args, "base_dir", None) else None
    out = run_chat(
        topic=str(args.topic),
        roles=roles or ["developer", "engineer", "scientist", "skeptic"],
        thread_id=_resolve_thread(args.thread_id),
        room_id=str(args.room_id) or _resolve_thread(args.thread_id),
        base_dir=base_dir,
        storage_format=getattr(args, "storage_format", None),
        llm=str(args.llm),
        directory=str(args.directory),
        interval=float(args.interval),
        quiet_secs=float(args.quiet_secs),
        max_messages=int(args.max_messages),
        max_reply_words=int(args.max_reply_words),
        window_size=int(args.window_size),
        minutes=float(args.minutes),
        facilitate=bool(args.facilitate),
        facilitator_budget=int(args.facilitator_budget),
        summarize=not bool(args.no_summarize),
        timeout=int(args.timeout),
        max_wall_secs=float(getattr(args, "max_wall_secs", 0.0) or 0.0),
    )
    if bool(args.json):
        print(json.dumps(out, indent=2, ensure_ascii=True))
    stopped_by = str(out.get("stopped_by", "") if isinstance(out, dict) else "")
    if stopped_by in {"backend_failure", "stalled", "error"}:
        return 3
    return 0


def _cmd_brief(args: argparse.Namespace) -> int:
    runtime = _build_runtime(args.base_dir, getattr(args, "storage_format", None))
    budget = int(args.token_budget) if int(args.token_budget) > 0 else None
    brief = runtime.build_brief(thread_id=_resolve_thread(args.thread_id), query=str(args.query), token_budget=budget)
    payload = brief.to_dict()
    if bool(args.json):
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        _print_brief(payload)
    return 0


def _cmd_handoff(args: argparse.Namespace) -> int:
    runtime = _build_runtime(args.base_dir, getattr(args, "storage_format", None))
    if str(args.import_path or "").strip():
        out = runtime.import_handoff(
            str(args.import_path),
            source_agent_id="external_agent",
            thread_id=str(args.thread_id or "") if str(args.thread_id or "") not in {"", "default"} else "",
        )
        print(json.dumps(out, indent=2, ensure_ascii=True))
        return _status_exit(out)

    packet = runtime.emit_handoff(role=str(args.role), thread_id=_resolve_thread(args.thread_id), scope=str(args.scope))
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
    runtime = _build_runtime(args.base_dir, getattr(args, "storage_format", None))
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
    runtime = _build_runtime(args.base_dir, getattr(args, "storage_format", None))
    view_name = str(args.view_name).replace("-", "_")
    allowed = {"active_scopes", "my_work", "open_gates", "stale_claims", "contradictions"}
    if view_name not in allowed:
        message = {"status": "invalid", "reason": f"unknown view: {args.view_name}", "views": sorted(allowed)}
        if bool(args.json):
            print(json.dumps(message, indent=2, ensure_ascii=True))
        else:
            print(f"unknown view: {args.view_name} (expected one of {sorted(allowed)})")
        return 2
    payload = runtime.query_view(
        view_name,
        thread_id=_resolve_thread(args.thread_id),
        owner_agent_id=str(args.owner_agent_id or ""),
        include_closed=bool(args.include_closed),
        limit=int(args.limit),
    ).to_dict()
    if bool(args.json):
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        _print_view(payload)
    return 0


def _cmd_prune(args: argparse.Namespace) -> int:
    runtime = _build_runtime(args.base_dir, getattr(args, "storage_format", None))
    payload = runtime.prune(
        thread_id=str(args.thread_id or ""),
        ttl_hours=float(args.ttl_hours),
        ttl_hours_facts=float(args.ttl_hours_facts) if float(args.ttl_hours_facts) > 0 else None,
        dry_run=bool(args.dry_run),
    )
    if bool(args.json):
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        for key in sorted(payload.keys()):
            print(f"{key}={payload[key]}")
    return 0


def _cmd_reconcile(args: argparse.Namespace) -> int:
    runtime = _build_runtime(args.base_dir, getattr(args, "storage_format", None))
    payload = runtime.reconcile(thread_id=_resolve_thread(args.thread_id), limit=int(args.limit))
    if bool(args.json):
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        for key in sorted(payload.keys()):
            print(f"{key}={payload[key]}")
    return 0


def _add_base_args(parser: argparse.ArgumentParser, *, include_json: bool = True) -> None:
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--storage-format", default=None, choices=["sqlite", "jsonl"])
    if include_json:
        parser.add_argument("--json", action="store_true")


def _print_config_payload(payload: Dict[str, Any]) -> None:
    if "clients" in payload:
        for client_payload in payload.get("clients", []):
            _print_config_payload(client_payload)
            print("")
        return
    print(f"client: {payload.get('client', '')}")
    print(f"path: {payload.get('path', '')}")
    snippet = str(payload.get("snippet", "") or "").rstrip("\n")
    print("snippet:")
    for line in snippet.splitlines():
        print(f"  {line}")
    verify = str(payload.get("verify", "") or "")
    if verify:
        print(f"verify: {verify}")
    grade = str(payload.get("grade", "") or "")
    if grade:
        print(f"grade: {grade}")
    note = str(payload.get("note", "") or "")
    if note:
        print(f"note: {note}")


def _cmd_harness(args: argparse.Namespace) -> int:
    from evidencespine.harness import (
        cmd_debug,
        cmd_doctor,
        cmd_install,
        cmd_precompact,
        cmd_print_config,
        cmd_session_start,
        cmd_session_stop,
    )

    action = str(getattr(args, "harness_action", "")).replace("-", "_")
    base_dir = getattr(args, "base_dir", None)
    storage_format = getattr(args, "storage_format", None)
    json_out = bool(getattr(args, "json", False))

    if action == "install":
        harness_target = getattr(args, "harness_target", None) or getattr(args, "harness", "all")
        payload = cmd_install(
            harness=str(harness_target),
            target_dir=str(getattr(args, "target_dir", ".")),
            base_dir=str(getattr(args, "harness_base_dir", ".evidencespine")),
            executable=getattr(args, "executable", None),
            scope=str(getattr(args, "scope", "project")),
            dry_run=bool(getattr(args, "dry_run", False)),
            mcp_url=getattr(args, "mcp_url", None),
            with_settings_hooks=bool(getattr(args, "with_settings_hooks", False)),
            with_agents_md=bool(getattr(args, "with_agents_md", False)),
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        status = str(payload.get("status", "")).lower()
        if status in {"invalid", "skipped", "fail_open"}:
            return 2
        if status == "partial" or payload.get("warnings"):
            return 3
        return 0

    if action == "debug":
        payload = cmd_debug(
            base_dir=base_dir,
            storage_format=storage_format,
            thread_id=getattr(args, "thread_id", None),
            target_dir=str(getattr(args, "target_dir", ".")),
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return _status_exit(payload, soft=("fail_open",))

    if action == "doctor":
        payload = cmd_doctor(
            base_dir=base_dir,
            storage_format=storage_format,
            thread_id=getattr(args, "thread_id", None),
            target_dir=str(getattr(args, "target_dir", ".")),
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 2 if str(payload.get("status", "")).lower() == "broken" else 0

    if action == "print_config":
        client_name = getattr(args, "client", None) or getattr(args, "harness", "generic")
        payload = cmd_print_config(
            client=str(client_name),
            scope=str(getattr(args, "scope", "project")),
            transport=str(getattr(args, "transport", "stdio")),
            executable=getattr(args, "executable", None),
            base_dir=str(getattr(args, "harness_base_dir", ".evidencespine")),
            project_dir=str(getattr(args, "target_dir", ".")),
        )
        if json_out:
            print(json.dumps(payload, indent=2, ensure_ascii=True))
        elif str(payload.get("status", "ok")) == "ok":
            _print_config_payload(payload)
        else:
            print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if str(payload.get("status", "ok")).lower() == "ok" else 2

    if action in {"session_start", "session_stop", "precompact", "compaction"}:
        session_id = str(getattr(args, "session_id", "") or "")
        if action == "session_start":
            print(
                cmd_session_start(
                    thread_id=getattr(args, "thread_id", None),
                    objective=str(getattr(args, "objective", "")),
                    token_budget=int(getattr(args, "token_budget", 0)) or None,
                    base_dir=base_dir,
                    storage_format=storage_format,
                    json_out=json_out,
                    session_id=session_id,
                )
            )
        elif action == "session_stop":
            print(
                cmd_session_stop(
                    thread_id=getattr(args, "thread_id", None),
                    summary=getattr(args, "summary", None),
                    auto_handoff=(bool(getattr(args, "auto_handoff", False)) or None),
                    reason=str(getattr(args, "reason", "session_stop")),
                    base_dir=base_dir,
                    storage_format=storage_format,
                    json_out=json_out,
                    session_id=session_id,
                )
            )
        else:
            # Plain-text stdout: Claude Code uses it as compaction-summary
            # instructions; Codex ignores it (ingest side effect only).
            print(
                cmd_precompact(
                    thread_id=getattr(args, "thread_id", None),
                    summary=getattr(args, "summary", None),
                    token_budget=int(getattr(args, "token_budget", 0)) or None,
                    base_dir=base_dir,
                    storage_format=storage_format,
                    json_out=json_out,
                    session_id=session_id,
                )
            )
        return 0

    if action in {"git_hook", "test_record"}:
        from evidencespine.harness.git import record_commit, record_test_result

        if action == "git_hook":
            payload = record_commit(
                str(getattr(args, "sha", "")),
                repo_dir=str(getattr(args, "repo_dir", ".")),
                base_dir=base_dir,
                storage_format=storage_format,
            )
        else:
            payload = record_test_result(
                str(getattr(args, "status", "passed")),
                str(getattr(args, "command", "")),
                base_dir=base_dir,
                storage_format=storage_format,
            )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return _status_exit(payload)

    if action == "install_hook":
        from evidencespine.harness.git import install_git_hooks

        payload = install_git_hooks(
            repo_dir=str(getattr(args, "target_dir", ".")),
            executable=str(getattr(args, "executable", "evidencespine")),
            base_dir=str(getattr(args, "git_base_dir", ".evidencespine")),
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return _status_exit(payload)

    print(json.dumps({"status": "invalid", "reason": f"unknown harness action: {action}"}))
    return 2


def _cmd_mcp(args: argparse.Namespace) -> int:
    try:
        import mcp  # noqa: F401
    except Exception as exc:
        print(f"error: the [mcp] extra is not installed (pip install evidencespine[mcp]): {exc}")
        return 2
    from evidencespine.mcp_server import run_server

    try:
        run_server(
            transport=str(getattr(args, "transport", "stdio")),
            host=str(getattr(args, "host", "127.0.0.1")),
            port=int(getattr(args, "port", 8000)),
            path=str(getattr(args, "path", "/mcp")),
            base_dir=args.base_dir,
            storage_format=getattr(args, "storage_format", None),
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    return 0


def _cmd_a2a(args: argparse.Namespace) -> int:
    try:
        import a2a  # noqa: F401
    except Exception as exc:
        print(f"error: the [a2a] extra is not installed (pip install evidencespine[a2a]): {exc}")
        return 2
    from evidencespine.a2a import run_server

    try:
        run_server(
            host=str(getattr(args, "host", "127.0.0.1")),
            port=int(getattr(args, "port", 8765)),
            base_dir=args.base_dir,
            storage_format=getattr(args, "storage_format", None),
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    return 0


def _cmd_migrate(args: argparse.Namespace) -> int:
    settings = EvidenceSpineSettings.from_env(base_dir=args.base_dir)
    config = settings.to_runtime_config()
    from evidencespine.store import AgentMemoryStoreConfig

    store_config = AgentMemoryStoreConfig(
        storage_format=str(config.storage_format),
        db_path=str(config.db_path),
        events_path=str(config.events_path),
        facts_path=str(config.facts_path),
        state_path=str(config.state_path),
        briefs_dir=str(config.briefs_dir),
        handoffs_dir=str(config.handoffs_dir),
        max_event_tail=int(config.max_event_tail),
        dedupe_window_sec=float(config.dedupe_window_sec),
        redaction_enable=bool(config.redaction_enable),
        fail_open=bool(config.fail_open),
    )

    from evidencespine.migrate import migrate_source_to_target, verify_migration

    source_format = str(getattr(args, "source_format", "") or "jsonl").strip().lower()
    target_format = str(getattr(args, "target_format", "") or "sqlite").strip().lower()
    if source_format not in {"sqlite", "jsonl"}:
        source_format = "jsonl"
    if target_format not in {"sqlite", "jsonl"}:
        target_format = "sqlite"
    if source_format == target_format:
        print(json.dumps({"status": "noop", "reason": "source_format_equals_target_format"}, indent=2, ensure_ascii=True))
        return 0

    verify_enabled = bool(getattr(args, "verify", True))
    result = migrate_source_to_target(
        store_config,
        source_format=source_format,
        target_format=target_format,
    )
    payload: Dict[str, Any] = dict(result)
    payload["verify_requested"] = verify_enabled
    payload["report"] = {
        "copied": {
            "events": int(result.get("events_copied", 0)),
            "facts": int(result.get("facts_copied", 0)),
        },
        "skipped": {
            "events": int(result.get("events_skipped", 0)),
            "facts": int(result.get("facts_skipped", 0)),
        },
    }
    exit_code = 0
    if verify_enabled:
        verified = verify_migration(
            store_config,
            source_format=source_format,
            target_format=target_format,
        )
        payload["verify"] = verified
        payload["report"]["verified"] = str(verified.get("status", "unknown"))
        matched = bool(verified.get("events_match")) and bool(verified.get("facts_match"))
        if not matched:
            payload["status"] = "mismatch"
            exit_code = 1
    else:
        payload["report"]["verified"] = "skipped"
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EvidenceSpine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest one structured memory event")
    _add_base_args(ingest)
    ingest.add_argument("--thread-id", default="", help="memory thread id (default: resolve from cwd/env)")
    ingest.add_argument("--session-id", default="", help="provider session id (metadata only)")
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
    ingest.add_argument("--ground-ref", action="append", default=[], help="auto-ground file:line refs into evidence items (any count, deduped)")
    ingest.add_argument("--source-root", default=".", help="root for grounding file refs")
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
    ingest.set_defaults(func=_cmd_ingest)

    ground = sub.add_parser("ground", help="Ground file:line refs into checksummed evidence items")
    ground.add_argument("ref", nargs="+", help="refs like path#L10-L20")
    ground.add_argument("--source-root", default=".")
    ground.add_argument("--json", action="store_true")
    ground.set_defaults(func=_cmd_ground)

    drift = sub.add_parser("drift-check", help="Re-verify grounded evidence against live files")
    _add_base_args(drift)
    drift.add_argument("--thread-id", default="")
    drift.add_argument("--source-root", default=".")
    drift.add_argument("--apply", action="store_true", help="write evidence_stale flags (default: dry-run)")
    drift.set_defaults(func=_cmd_drift_check)

    verify = sub.add_parser("verify", help="Record verification provenance for a fact")
    _add_base_args(verify)
    verify.add_argument("--fact-id", required=True)
    verify.add_argument("--method", required=True, choices=["test", "gate", "tool", "manual"])
    verify.add_argument("--reference", required=True)
    verify.add_argument("--verified-by", default="external_agent")
    verify.add_argument("--thread-id", default="")
    verify.set_defaults(func=_cmd_verify)

    brief = sub.add_parser("brief", help="Build bounded context brief")
    _add_base_args(brief)
    brief.add_argument("--thread-id", default="", help="memory thread id (default: resolve from cwd/env)")
    brief.add_argument("--query", default="")
    brief.add_argument("--token-budget", type=int, default=0)
    brief.set_defaults(func=_cmd_brief)

    chat = sub.add_parser("chat", help="Realtime multi-agent chat over the spine")
    _add_base_args(chat)
    chat.add_argument("--topic", default="Debate the usefulness of EvidenceSpine")
    chat.add_argument("--roles", default="developer,engineer,scientist,skeptic")
    chat.add_argument("--thread-id", default="", help="chat room thread (default: resolve from cwd/env)")
    chat.add_argument("--room-id", default="")
    chat.add_argument("--llm", default="opencode", choices=["opencode", "scripted", "echo"])
    chat.add_argument("--directory", default=".")
    chat.add_argument("--interval", type=float, default=1.5)
    chat.add_argument("--quiet-secs", type=float, default=12.0)
    chat.add_argument("--max-messages", type=int, default=24)
    chat.add_argument("--max-reply-words", type=int, default=60)
    chat.add_argument("--window-size", type=int, default=12)
    chat.add_argument("--minutes", type=float, default=0.0, help="max run duration in minutes (0 = derive a <=180s cap)")
    chat.add_argument("--max-wall-secs", type=float, default=0.0, help="hard wall-clock cap in seconds (0 = derive from minutes)")
    chat.add_argument("--facilitate", action="store_true", help="pump follow-up questions when the room stalls")
    chat.add_argument("--facilitator-budget", type=int, default=3)
    chat.add_argument("--no-summarize", action="store_true", help="disable rolling summary of old messages")
    chat.add_argument("--timeout", type=int, default=180)
    chat.set_defaults(func=_cmd_chat)

    handoff = sub.add_parser("handoff", help="Emit/import handoff packet")
    _add_base_args(handoff)
    handoff.add_argument("--role", default="auditor")
    handoff.add_argument("--thread-id", default="", help="memory thread id (default: resolve from cwd/env; imports inherit the packet thread)")
    handoff.add_argument("--scope", default="cross-agent coordination")
    handoff.add_argument("--output", default="")
    handoff.add_argument("--import", dest="import_path", default="")
    handoff.set_defaults(func=_cmd_handoff)

    snapshot = sub.add_parser("snapshot", help="Show memory health snapshot")
    _add_base_args(snapshot)
    snapshot.set_defaults(func=_cmd_snapshot)

    view = sub.add_parser("view", help="Show derived agent-state control views")
    view.add_argument(
        "view_name",
        choices=["active-scopes", "my-work", "open-gates", "stale-claims", "contradictions"],
    )
    _add_base_args(view)
    view.add_argument("--thread-id", default="")
    view.add_argument("--owner-agent-id", default="")
    view.add_argument("--include-closed", action="store_true")
    view.add_argument("--limit", type=int, default=50)
    view.set_defaults(func=_cmd_view)

    reconcile = sub.add_parser("reconcile", help="Run optional state reconciliation hook")
    _add_base_args(reconcile)
    reconcile.add_argument("--thread-id", default="", help="memory thread id (default: resolve from cwd/env)")
    reconcile.add_argument("--limit", type=int, default=50)
    reconcile.set_defaults(func=_cmd_reconcile)

    prune_cmd = sub.add_parser("prune", help="TTL archival: delete rows older than the TTL")
    _add_base_args(prune_cmd)
    prune_cmd.add_argument("--thread-id", default="")
    prune_cmd.add_argument("--ttl-hours", type=float, default=720.0)
    prune_cmd.add_argument("--ttl-hours-facts", type=float, default=0.0)
    prune_cmd.add_argument("--dry-run", action="store_true")
    prune_cmd.set_defaults(func=_cmd_prune)

    migrate = sub.add_parser("migrate", help="Migrate store between JSONL and SQLite backends")
    _add_base_args(migrate)
    migrate.add_argument("--source-format", default="", choices=["", "sqlite", "jsonl"])
    migrate.add_argument("--target-format", default="", choices=["", "sqlite", "jsonl"])
    migrate.add_argument("--no-verify", dest="verify", action="store_false", help="copy only; skip verification")
    migrate.set_defaults(func=_cmd_migrate)

    mcp_cmd = sub.add_parser(
        "mcp",
        help="Run the MCP server (stdio or streamable-http)",
        epilog=(
            "Auth: set EVIDENCESPINE_MCP_AUTH_TOKEN for streamable-http; non-loopback binds "
            "are refused without auth. Escape hatch (trusted dev only): "
            "EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED=1."
        ),
    )
    _add_base_args(mcp_cmd, include_json=False)
    mcp_cmd.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    mcp_cmd.add_argument("--host", default="127.0.0.1")
    mcp_cmd.add_argument("--port", type=int, default=8000)
    mcp_cmd.add_argument("--path", default="/mcp")
    mcp_cmd.set_defaults(func=_cmd_mcp)

    a2a_cmd = sub.add_parser(
        "a2a",
        help="Run the A2A (Agent-to-Agent) protocol server",
        epilog=(
            "Auth: set EVIDENCESPINE_A2A_AUTH_TOKEN to require bearer auth; non-loopback binds "
            "are refused without auth. Escape hatch (trusted dev only): "
            "EVIDENCESPINE_A2A_ALLOW_UNAUTHENTICATED=1."
        ),
    )
    a2a_cmd.add_argument("--host", default="127.0.0.1")
    a2a_cmd.add_argument("--port", type=int, default=8765)
    a2a_cmd.add_argument("--base-dir", default=None)
    a2a_cmd.add_argument("--storage-format", default=None, choices=["sqlite", "jsonl"])
    a2a_cmd.set_defaults(func=_cmd_a2a)

    harness_cmd = sub.add_parser("harness", help="Harness delivery hooks and installation")
    harness_sub = harness_cmd.add_subparsers(dest="harness_group", required=True)

    install_p = harness_sub.add_parser("install", help="Install the delivery layer into a harness")
    install_p.add_argument(
        "--harness",
        dest="harness_target",
        default="all",
        choices=[
            "claude-code",
            "opencode",
            "cursor",
            "vscode",
            "vscode-copilot",
            "codex",
            "git",
            "agents-md",
            "dsh",
            "generic",
            "instructions",
            "all",
        ],
    )
    install_p.add_argument("--target-dir", default=".")
    install_p.add_argument("--harness-base-dir", default=".evidencespine")
    install_p.add_argument("--executable", default=None)
    install_p.add_argument(
        "--scope",
        default="project",
        choices=["project", "global"],
        help=(
            "project writes under target-dir (.claude/skills, .opencode/plugins, .cursor/, "
            ".vscode/, .codex/); global writes directly into target-dir (e.g. ~/.claude, "
            "~/.codex, ~/.config/opencode/plugins)"
        ),
    )
    install_p.add_argument("--mcp-url", default=None, help="register HTTP MCP transport (Codex: url + bearer env var)")
    install_p.add_argument(
        "--with-settings-hooks",
        action="store_true",
        help="Claude Code: also append plugin hooks to .claude/settings.json (opt-in; both copies run)",
    )
    install_p.add_argument("--with-agents-md", action="store_true", help="also write the AGENTS.md managed block for lazy harnesses")
    install_p.add_argument("--dry-run", action="store_true", help="report planned writes without writing")
    install_p.add_argument("--json", action="store_true", help="JSON output (install always prints JSON)")
    install_p.set_defaults(harness_action="install")
    install_p.set_defaults(func=_cmd_harness)

    debug_p = harness_sub.add_parser("debug", help="Check harness delivery health")
    _add_base_args(debug_p)
    debug_p.add_argument("--thread-id", default="")
    debug_p.add_argument("--target-dir", default=".")
    debug_p.set_defaults(harness_action="debug")
    debug_p.set_defaults(func=_cmd_harness)

    doctor_p = harness_sub.add_parser("doctor", help="Validate installed harness files + JSON/TOML validity")
    _add_base_args(doctor_p)
    doctor_p.add_argument("--thread-id", default="")
    doctor_p.add_argument("--target-dir", default=".")
    doctor_p.set_defaults(harness_action="doctor")
    doctor_p.set_defaults(func=_cmd_harness)

    print_p = harness_sub.add_parser("print-config", help="Print a copy-paste MCP config (never writes)")
    print_p.add_argument(
        "--harness",
        "--client",
        dest="client",
        default="generic",
        choices=sorted(
            [
                "claude-code",
                "claude-desktop",
                "codex",
                "cursor",
                "vscode",
                "vscode-copilot",
                "windsurf",
                "zed",
                "gemini",
                "antigravity",
                "cline",
                "roo",
                "continue",
                "junie",
                "warp",
                "amp",
                "jetbrains",
                "opencode",
                "dsh",
                "generic",
                "all",
            ]
        ),
    )
    print_p.add_argument("--scope", default="project", choices=["project", "global"])
    print_p.add_argument("--transport", default="stdio", choices=["stdio", "http"])
    print_p.add_argument("--target-dir", default=".")
    print_p.add_argument("--harness-base-dir", default=".evidencespine")
    print_p.add_argument("--executable", default=None)
    print_p.add_argument("--json", action="store_true")
    print_p.set_defaults(harness_action="print-config")
    print_p.set_defaults(func=_cmd_harness)

    for provider, actions in (
        ("claude-code", ("session-start", "session-stop", "precompact")),
        ("opencode", ("session-start", "session-stop", "compaction")),
        ("codex", ("session-start", "session-stop", "precompact")),
        ("cursor", ("session-start", "session-stop", "precompact")),
        ("vscode", ("session-start", "session-stop", "precompact")),
        ("git", ("install-hook", "git-hook", "test-record")),
    ):
        provider_p = harness_sub.add_parser(provider, help=f"{provider} delivery hooks")
        provider_sub = provider_p.add_subparsers(dest="harness_action", required=True)
        for action in actions:
            action_p = provider_sub.add_parser(action)
            _add_base_args(action_p)
            if action == "install-hook":
                action_p.add_argument("--target-dir", default=".")
                action_p.add_argument("--executable", default="evidencespine")
                action_p.add_argument("--git-base-dir", dest="git_base_dir", default=".evidencespine")
            elif action == "session-start":
                action_p.add_argument("--thread-id", default="")
                action_p.add_argument("--session-id", default="")
                action_p.add_argument("--objective", default="")
                action_p.add_argument("--token-budget", type=int, default=0)
            elif action == "session-stop":
                action_p.add_argument("--thread-id", default="")
                action_p.add_argument("--session-id", default="")
                action_p.add_argument("--summary", default="")
                action_p.add_argument("--auto-handoff", action="store_true")
                action_p.add_argument("--reason", default="session_stop")
            elif action == "git-hook":
                action_p.add_argument("--sha", required=True)
                action_p.add_argument("--repo-dir", default=".")
            elif action == "test-record":
                action_p.add_argument("--status", default="passed", choices=["passed", "failed", "ok", "green", "error"])
                action_p.add_argument("--command", required=True)
            else:
                action_p.add_argument("--thread-id", default="")
                action_p.add_argument("--session-id", default="")
                action_p.add_argument("--summary", default="")
                action_p.add_argument("--token-budget", type=int, default=0)
            action_p.set_defaults(func=_cmd_harness)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
