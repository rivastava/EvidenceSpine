"""Harness delivery layer: memory recall/retain as a property of the harness.

Research ground ("Delivery, Not Storage"): agents make almost no voluntary
memory calls, so v0.5.0 delivers memory through harness hooks — auto-recall of
a bounded context brief at session start, retain-through-compaction, and
auto-retain of session end (optionally as an evidence-bound handoff packet).

Installation, verification (harness debug/doctor/print-config) and the
print-only universal MCP client matrix also live here.
"""

from evidencespine.harness.commands import (
    cmd_debug,
    cmd_doctor,
    cmd_install,
    cmd_precompact,
    cmd_print_config,
    cmd_session_start,
    cmd_session_stop,
)
from evidencespine.harness.hooks import (
    handle_precompact,
    handle_session_start,
    handle_session_stop,
)
from evidencespine.harness.install import (
    debug_harness,
    doctor_harness,
    install_harness,
    render_print_config,
    resolve_executable,
)

__all__ = [
    "cmd_debug",
    "cmd_doctor",
    "cmd_install",
    "cmd_precompact",
    "cmd_print_config",
    "cmd_session_start",
    "cmd_session_stop",
    "handle_precompact",
    "handle_session_start",
    "handle_session_stop",
    "debug_harness",
    "doctor_harness",
    "install_harness",
    "render_print_config",
    "resolve_executable",
]
