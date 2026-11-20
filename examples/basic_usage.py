"""Minimal end-to-end example: ground a decision, then brief and query state.

Run from the repo root:

    PYTHONPATH=src python examples/basic_usage.py

Uses a temporary workspace/store, so it never writes into your checkout. The
evidence file is created first, then grounded through the same helper the CLI
and MCP server use, which is what makes the claim "verified".
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from evidencespine.grounding import ground_file
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="evidencespine_basic_") as tmp:
        workspace = Path(tmp)
        (workspace / "reports").mkdir()
        (workspace / "reports" / "decision.md").write_text(
            "Decision log\n"
            "============\n"
            "\n"
            "Context: the release branch must keep a clean history.\n"
            "\n"
            "Options considered:\n"
            "1. Rebase the feature branch.\n"
            "2. Squash the patch set.\n"
            "3. Apply patch set A (additive only).\n"
            "\n"
            "Decision: apply patch set A (additive only).\n"
            "Owner: implementer\n"
            "Reviewer: auditor\n",
            encoding="utf-8",
        )
        item = ground_file("reports/decision.md", 11, 11, source_root=str(workspace))
        assert item is not None, "evidence span must be readable"

        settings = EvidenceSpineSettings.from_env(base_dir=str(workspace / ".evidencespine"))
        runtime = AgentMemoryRuntime(config=settings.to_runtime_config())

        ingest = runtime.ingest_event(
            {
                "thread_id": "demo",
                "event_type": "decision",
                "role": "implementer",
                "source_agent_id": "implementer",
                "source_turn_id": "turn-1",
                "payload": {
                    "claim": "Apply patch set A (additive only)",
                    "decision": "Apply patch set A",
                    "fact_state": "verified",
                },
                "evidence_refs": ["reports/decision.md#L11-L11"],
                "evidence_items": [item],
                "state_context": {
                    "scope_id": "patch-set-a",
                    "state_kind": "agent_local_work",
                    "status": "active",
                    "owner_agent_id": "implementer",
                },
                "confidence": 0.9,
                "salience": 0.7,
            }
        )
        print("ingest:", ingest)

        brief = runtime.build_brief("demo", "current status and next actions").to_dict()
        print("locked_decisions:", brief["locked_decisions"])
        print(
            "span_grounded:",
            [citation["span_grounded"] for citation in brief["citations"].values()],
        )
        print("stale:", brief["metadata"]["stale"])

        view = runtime.query_view("active_scopes", thread_id="demo").to_dict()
        print("active_scopes:", [(row["scope_id"], row["status"]) for row in view["rows"]])

        drift = runtime.check_evidence_stale(source_root=str(workspace), dry_run=True)
        print("drift (clean):", drift["stale_facts"], "stale /", drift["checked_items"], "checked")

        # Change the cited line: drift-check now flags the claim instead of
        # silently believing it.
        revised = (workspace / "reports" / "decision.md").read_text(encoding="utf-8").splitlines()
        revised[10] = "Decision: apply patch set B instead."
        (workspace / "reports" / "decision.md").write_text("\n".join(revised) + "\n", encoding="utf-8")
        drift = runtime.check_evidence_stale(source_root=str(workspace), dry_run=True)
        print("drift (edited):", drift["results"][0]["reason"] if drift["results"] else "none")

        runtime.store.close()


if __name__ == "__main__":
    main()
