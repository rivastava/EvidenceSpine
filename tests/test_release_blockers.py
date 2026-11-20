from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from evidencespine.a2a.card import build_agent_card
from evidencespine.harness.install import install_cursor
from evidencespine.migrate import migrate_source_to_target
from evidencespine.store import AgentMemoryStoreConfig


def _git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def test_cursor_hook_install_is_idempotent(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    first = install_cursor(target_dir=str(repo), base_dir=".evidencespine", executable="evidencespine")
    assert first["status"] == "ok"
    hooks_path = repo / ".cursor" / "hooks.json"
    first_payload = json.loads(hooks_path.read_text(encoding="utf-8"))
    counts_first = {event: len(groups) for event, groups in first_payload["hooks"].items()}
    for _ in range(3):
        out = install_cursor(target_dir=str(repo), base_dir=".evidencespine", executable="evidencespine")
        assert out["status"] == "ok"
    payload = json.loads(hooks_path.read_text(encoding="utf-8"))
    counts = {event: len(groups) for event, groups in payload["hooks"].items()}
    assert counts == counts_first, f"duplicate hook groups after reinstall: {counts}"


def test_agent_card_advertises_supported_interface() -> None:
    card = build_agent_card(url="http://127.0.0.1:8765/")
    interfaces = list(card.supported_interfaces)
    assert interfaces and interfaces[0].url == "http://127.0.0.1:8765/"
    assert interfaces[0].protocol_binding == "JSONRPC"


def test_explicit_migrate_marks_jsonl_and_prevents_resurrection(tmp_path: Path) -> None:
    base = tmp_path / ".evidencespine"
    base.mkdir(parents=True)
    config = AgentMemoryStoreConfig(
        storage_format="sqlite",
        db_path=str(base / "evidencespine.db"),
        events_path=str(base / "events.jsonl"),
        facts_path=str(base / "facts.jsonl"),
        state_path=str(base / "state.json"),
        briefs_dir=str(base / "briefs"),
        handoffs_dir=str(base / "handoffs"),
    )
    with open(config.events_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"thread_id": "t", "event_type": "decision", "source_agent_id": "a", "source_turn_id": "s1", "ts_utc": "2000-01-01T00:00:00Z", "payload": {"claim": "legacy"}}) + "\n")
    with open(config.facts_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"thread_id": "t", "claim": "legacy fact", "state": "asserted", "fact_id": "f1", "ts_utc": "2000-01-01T00:00:00Z"}) + "\n")
    result = migrate_source_to_target(config, source_format="jsonl", target_format="sqlite")
    assert result["status"] == "ok"
    assert result.get("jsonl_migration_marked") is True
    state = json.loads(Path(config.state_path).read_text(encoding="utf-8"))
    assert state["jsonl_migration"]["source"] == "explicit_migrate"

    # emptying the target must not resurrect the legacy rows on reopen
    from evidencespine.store import AgentMemoryStore

    store = AgentMemoryStore(config)
    store.prune(ttl_hours=0.001)
    assert store.count_events() == 0 and store.count_facts() == 0
    store.close()
    reopened = AgentMemoryStore(config)
    assert reopened.count_events() == 0, "pruned rows were resurrected from legacy JSONL"
    assert reopened.count_facts() == 0
    reopened.close()


def test_cli_ingest_counters_survive_process_exit(tmp_path: Path) -> None:
    base_dir = tmp_path / ".evidencespine"
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    cmd = [sys.executable, "-m", "evidencespine.cli", "ingest", "--base-dir", str(base_dir),
           "--thread-id", "t", "--event-type", "decision", "--source-agent-id", "a",
           "--source-turn-id", "s1", "--claim", "hello", "--json"]
    for turn in ("s1", "s2"):
        run_cmd = [x if x != "s1" else turn for x in cmd]
        proc = subprocess.run(run_cmd, capture_output=True, text=True, env=env)
        assert proc.returncode == 0, proc.stderr
    snap = subprocess.run([sys.executable, "-m", "evidencespine.cli", "snapshot", "--base-dir", str(base_dir), "--json"],
                          capture_output=True, text=True, env=env)
    assert snap.returncode == 0, snap.stderr
    payload = json.loads(snap.stdout)
    assert payload["events_total"] == 2
    assert payload["facts_total"] >= 1, "fact counter did not survive process exit"


def test_brief_and_view_resolve_workspace_thread(tmp_path: Path) -> None:
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    env.pop("EVIDENCESPINE_THREAD_ID", None)
    base = str(tmp_path / ".evidencespine")
    ingest = subprocess.run([sys.executable, "-m", "evidencespine.cli", "ingest", "--base-dir", base,
                             "--event-type", "decision", "--source-agent-id", "a", "--source-turn-id", "s1",
                             "--claim", "workspace claim", "--json"], capture_output=True, text=True, env=env)
    assert ingest.returncode == 0, ingest.stderr
    brief = subprocess.run([sys.executable, "-m", "evidencespine.cli", "brief", "--base-dir", base, "--json"],
                           capture_output=True, text=True, env=env)
    assert brief.returncode == 0, brief.stderr
    payload = json.loads(brief.stdout)
    assert payload["thread_id"].startswith("proj_")
    assert payload["recent_verified_facts"] or payload["locked_decisions"] or payload["open_items"]

def test_agent_card_interface_absent_without_url() -> None:
    card = build_agent_card()
    assert len(card.supported_interfaces) == 0

def test_cli_view_and_prune_thread_scope(tmp_path: Path, monkeypatch) -> None:
    """view resolves the workspace thread; prune without --thread-id stays global."""
    import argparse

    from evidencespine import cli

    captured = {}

    class FakeRuntime:
        def query_view(self, view, **kwargs):
            captured["view_thread"] = kwargs.get("thread_id")

            class _View:
                def to_dict(self):
                    return {"view": view, "rows": []}

            return _View()

        def prune(self, **kwargs):
            captured["prune_thread"] = kwargs.get("thread_id")
            return {"status": "ok"}

    monkeypatch.setattr(cli, "_build_runtime", lambda *a, **k: FakeRuntime())
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EVIDENCESPINE_THREAD_ID", raising=False)
    rc = cli._cmd_view(argparse.Namespace(view_name="active-scopes", thread_id="", owner_agent_id="",
                                          include_closed=False, limit=10, json=True, base_dir=None,
                                          storage_format=None))
    assert rc == 0 and captured["view_thread"].startswith("proj_")
    assert cli._cmd_prune(argparse.Namespace(thread_id="", ttl_hours=720.0, ttl_hours_facts=0.0,
                                            dry_run=True, json=True, base_dir=None, storage_format=None)) == 0
    assert captured["prune_thread"] == "", "prune without --thread-id must stay global"


def test_mcp_build_brief_allows_omitted_thread(tmp_path: Path) -> None:
    from evidencespine.mcp_server.server import create_server
    from evidencespine.settings import EvidenceSpineSettings
    from evidencespine.runtime import AgentMemoryRuntime

    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    server = create_server(runtime=AgentMemoryRuntime(config=settings.to_runtime_config()))

    async def go() -> None:
        result = await server.call_tool("build_brief", {"query": "anything"})
        assert '"thread_id": "proj_' in result.content[0].text

    import asyncio

    asyncio.run(go())