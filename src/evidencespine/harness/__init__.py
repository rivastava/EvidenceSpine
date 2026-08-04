"""Harness delivery layer: memory recall/retain as a property of the harness.

Research ground ("Delivery, Not Storage"): agents make almost no voluntary
memory calls, so v0.5.0 delivers memory through harness hooks — auto-recall of
a bounded context brief at session start, retain-through-compaction, and
auto-retain of session end (optionally as an evidence-bound handoff packet).
"""

from evidencespine.harness.commands import (
    cmd_debug,
    cmd_install,
    cmd_precompact,
    cmd_session_start,
    cmd_session_stop,
)
from evidencespine.harness.hooks import (
    handle_precompact,
    handle_session_start,
    handle_session_stop,
)

__all__ = [
    "cmd_debug",
    "cmd_install",
    "cmd_precompact",
    "cmd_session_start",
    "cmd_session_stop",
    "handle_precompact",
    "handle_session_start",
    "handle_session_stop",
]
