"""DeepSeek Harness (DSH) MCP delivery: `$DSH_HOME/cordis.patch.yml`.

DSH composes one machine-wide *user patch layer* above every profile's own
`cordis.patch.yml`. A static MCP client row inserted there is therefore
available to `web`, `headless`, `acp`, `sdk` and desktop sessions
without profile detection:

.. code-block:: yaml

    # BEGIN EvidenceSpine (managed)
    - insert:
        - id: evidencespine-mcp
          name: '@deepseek-ai/dsh-mcp-client'
          config:
            serverName: evidencespine
            transport: stdio
            command: /absolute/path/to/evidencespine
            args:
              - mcp
              - '--base-dir'
              - /absolute/base
            cwd: /absolute/project/root
            failOnStartupError: false
            startupTimeoutMs: 20000
    # END EvidenceSpine (managed)

Why this shape (DeepSeek Harness 0.2.2-rc.5):

* The `@deepseek-ai/dsh-mcp-client` row is mounted by every profile. The
  `mcp-manager` route exists only in web/desktop and a profile patch
  *replaces* the entry's complete `servers` map, so the manager route is
  only used for `--scope profile:<name>` and only when its target file is
  ours or empty (never merged blindly into user-owned YAML).
* `cwd` and `--base-dir` MUST be absolute: the stdio child inherits the DSH
  host process cwd (the Electron app usually runs from `/`), not the session
  workspace, so a relative `.evidencespine` would land in the wrong place.
* `failOnStartupError: false` keeps sessions alive when the side-car breaks.
* `serverName` must match `[A-Za-z0-9_-]{1,32}`; tools surface as
  `mcp__evidencespine__<tool>`.

Write strategy (stdlib only, never corrupt user YAML): a textual merge around
sentinel comments (`# BEGIN/END EvidenceSpine (managed)`) that preserves the
rest of the file verbatim, atomic temp + `os.replace` writes, mode `0600`
and a timestamped `.bak` before a modification. An existing file is only
touched when it is empty/comments-only or a top-level YAML sequence (first
non-comment line starts with `-`); anything else is reported `invalid` and
left untouched. Verification (when the `dsh` CLI is available) composes the
real config with `dsh --dump-config --profile headless|web` instead of
grepping our own file; it never fails the install.

Known limits: the home patch applies to *every* profile and a malformed home
patch breaks boot everywhere (we refuse to write one); the desktop profile is
managed exclusively by the Electron app; Windows has no shell in the stdio
spawn path, so the command must be an executable (not a `.cmd` shim).
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

HARNESS = "dsh"
MANAGED_BEGIN = "# BEGIN EvidenceSpine (managed)"
MANAGED_END = "# END EvidenceSpine (managed)"
ROW_ID = "evidencespine-mcp"
SERVER_NAME = "evidencespine"
MCP_CLIENT_PACKAGE = "@deepseek-ai/dsh-mcp-client"
DEFAULT_STARTUP_TIMEOUT_MS = 20000
PATCH_FILENAME = "cordis.patch.yml"
PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_PLAIN_SCALAR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_YAML_RESERVED = {"true", "false", "null", "yes", "no", "on", "off", "~"}

_DSH_CLI_CANDIDATES = (
    "/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh",
    os.path.expanduser("~/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh"),
    "/usr/local/bin/dsh",
    "/opt/homebrew/bin/dsh",
)


# ---------------------------------------------------------------------------
# paths / scope
# ---------------------------------------------------------------------------


def dsh_home() -> str:
    """Return `$DSH_HOME` or `~/.dsh` as an absolute path (blank env = unset)."""
    raw = str(os.environ.get("DSH_HOME", "") or "").strip()
    if not raw:
        raw = "~/.dsh"
    return os.path.abspath(os.path.expanduser(raw))


def patch_path(home: str = "") -> str:
    """Absolute path of the machine-wide user patch layer."""
    return os.path.join(os.path.abspath(os.path.expanduser(home or dsh_home())), PATCH_FILENAME)


def profile_patch_path(home: str, profile: str) -> str:
    """Absolute path of one profile's patch layer (`profiles/<name>/`)."""
    return os.path.join(
        os.path.abspath(os.path.expanduser(home or dsh_home())),
        "profiles",
        str(profile),
        PATCH_FILENAME,
    )


def _parse_scope(scope: str) -> Tuple[str, str, str]:
    """Return `(kind, profile_name, error)`; kind is `home` or `profile`."""
    raw = str(scope or "project").strip().lower()
    if raw in {"", "project", "global", "user", "machine", "all"}:
        return "home", "", ""
    if raw.startswith("profile:"):
        name = raw.split(":", 1)[1].strip()
        if not PROFILE_NAME_RE.match(name):
            return "", "", "invalid profile scope: " + repr(scope)
        return "profile", name, ""
    return "", "", "unsupported dsh scope: " + repr(scope) + " (expected project|global|profile:<name>)"


def _absolute_path(path: str, base: str = "") -> str:
    expanded = os.path.expanduser(str(path or "").strip() or ".")
    if os.path.isabs(expanded):
        return expanded
    anchor = os.path.abspath(os.path.expanduser(base or os.getcwd()))
    return os.path.abspath(os.path.join(anchor, expanded))


def _project_root(path: str) -> str:
    from evidencespine.threads import project_root

    return project_root(path or os.getcwd())


# ---------------------------------------------------------------------------
# YAML rendering (textual; stdlib only, no PyYAML dependency)
# ---------------------------------------------------------------------------


def _yaml_scalar(value: str) -> str:
    """Render one YAML single-quoted scalar (paths may contain spaces)."""
    text = str(value).replace("\r", " ").replace("\n", " ")
    return "'" + text.replace("'", "''") + "'"


def _yaml_token(value: str) -> str:
    """Render a plain scalar when safe, else a quoted scalar."""
    text = str(value)
    if _PLAIN_SCALAR_RE.match(text) and text.lower() not in _YAML_RESERVED:
        return text
    return _yaml_scalar(text)


def render_patch_block(row: Dict[str, Any], *, note: str = "") -> str:
    """Render the managed client-row block (sentinel comments included)."""
    config = row["config"]
    lines: List[str] = [
        MANAGED_BEGIN,
        "# EvidenceSpine - DSH MCP client row (generated by `evidencespine harness install --harness dsh`).",
        "# Machine-wide user patch layer: applies to every DSH profile, above each profile's cordis.patch.yml.",
        "# The row is merged by id; uninstall removes only this managed block.",
    ]
    if note:
        lines.append("# " + str(note).replace("\n", " "))
    lines.extend(
        [
            "- insert:",
            "    - id: " + _yaml_token(str(row["id"])),
            "      name: " + _yaml_scalar(str(row["name"])),
            "      config:",
            "        serverName: " + _yaml_token(str(config["serverName"])),
            "        transport: stdio",
            "        command: " + _yaml_scalar(str(config["command"])),
            "        args:",
        ]
    )
    for arg in config.get("args") or []:
        lines.append("          - " + _yaml_scalar(str(arg)))
    lines.extend(
        [
            "        cwd: " + _yaml_scalar(str(config["cwd"])),
            "        failOnStartupError: " + ("true" if config.get("failOnStartupError") else "false"),
            "        startupTimeoutMs: " + str(int(config.get("startupTimeoutMs") or DEFAULT_STARTUP_TIMEOUT_MS)),
            MANAGED_END,
        ]
    )
    return "\n".join(lines) + "\n"


def render_manager_block(profile: str, row: Dict[str, Any]) -> str:
    """Render the GUI-visible `mcp-manager` row for a single profile."""
    config = row["config"]
    lines: List[str] = [
        MANAGED_BEGIN,
        "# EvidenceSpine - DSH mcp-manager row for profile "
        + repr(profile)
        + " (written by `evidencespine harness install --harness dsh`).",
        "# A manager row replaces that profile's whole servers map; keep Settings-managed servers out of this block.",
        "- id: mcp-manager",
        "  config:",
        "    servers:",
        "      " + _yaml_token(SERVER_NAME) + ":",
        "        transport: stdio",
        "        command: " + _yaml_scalar(str(config["command"])),
        "        args:",
    ]
    for arg in config.get("args") or []:
        lines.append("          - " + _yaml_scalar(str(arg)))
    lines.extend(
        [
            "        cwd: " + _yaml_scalar(str(config["cwd"])),
            "        enabled: true",
            MANAGED_END,
        ]
    )
    return "\n".join(lines) + "\n"


def render_http_patch_block(
    url: str = "http://127.0.0.1:8000/mcp",
    *,
    auth_env: str = "EVIDENCESPINE_MCP_AUTH_TOKEN",
    fail_on_startup_error: bool = True,
) -> str:
    """Render the managed streamable-HTTP client row (DSH Loader !!js header).

    DSH resolves !!js expressions at composition time, so the bearer token is
    read from the host environment at startup. A plain string header is NOT
    interpolated, which is why this variant exists.
    """
    safe_url = str(url or "").strip() or "http://127.0.0.1:8000/mcp"
    safe_env = str(auth_env or "EVIDENCESPINE_MCP_AUTH_TOKEN").strip()
    header_line = "          Authorization: !!js '\"Bearer \" + process.env." + safe_env + "'"
    lines: List[str] = [
        MANAGED_BEGIN,
        "# EvidenceSpine - DSH MCP client row (streamable-HTTP; generated by evidencespine harness print-config).",
        "# Machine-wide user patch layer: applies to every DSH profile.",
        "- insert:",
        "    - id: " + _yaml_token(ROW_ID),
        "      name: " + _yaml_scalar(MCP_CLIENT_PACKAGE),
        "      config:",
        "        serverName: " + _yaml_token(SERVER_NAME),
        "        transport: streamable-http",
        "        url: " + _yaml_scalar(safe_url),
        "        headers:",
        header_line,
        "        failOnStartupError: " + ("true" if fail_on_startup_error else "false"),
        MANAGED_END,
    ]
    return "\n".join(lines) + "\n"


def _resolve_row(
    *,
    executable: str,
    base_dir: str,
    project_dir: str = "",
    target_dir: str = "",
) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    argv = _executable_argv(executable, warnings)
    anchor = os.path.abspath(os.path.expanduser(target_dir or os.getcwd()))
    abs_base = _absolute_path(base_dir, anchor)
    cwd = _project_root(project_dir or target_dir or os.getcwd())
    row: Dict[str, Any] = {
        "id": ROW_ID,
        "name": MCP_CLIENT_PACKAGE,
        "config": {
            "serverName": SERVER_NAME,
            "transport": "stdio",
            "command": argv[0],
            "args": [*argv[1:], "mcp", "--base-dir", abs_base],
            "cwd": cwd,
            "failOnStartupError": False,
            "startupTimeoutMs": DEFAULT_STARTUP_TIMEOUT_MS,
        },
    }
    return row, warnings


def build_insert_row(
    *,
    executable: str = "",
    base_dir: str = ".evidencespine",
    project_dir: str = "",
    target_dir: str = "",
) -> Dict[str, Any]:
    """Build the DSH client row with absolute command/args/cwd (public helper)."""
    row, _ = _resolve_row(
        executable=executable,
        base_dir=base_dir,
        project_dir=project_dir,
        target_dir=target_dir,
    )
    return row


def _executable_argv(executable: str, warnings: List[str]) -> List[str]:
    """Split the executable into argv, resolving a bare name to an absolute path."""
    try:
        argv = shlex.split(str(executable or ""))
    except ValueError:
        argv = []
    if not argv:
        try:
            from evidencespine.harness.install import resolve_executable

            argv = shlex.split(resolve_executable())
        except Exception:
            argv = ["evidencespine"]
    if not argv:
        argv = ["evidencespine"]
    command = argv[0]
    if not os.path.isabs(command):
        found = shutil.which(command)
        if found:
            argv[0] = os.path.abspath(found)
        else:
            warnings.append(
                "command "
                + repr(command)
                + " is not absolute and was not found on PATH; DSH spawns stdio servers "
                "without a shell (use --executable with an absolute path)"
            )
    return argv


def _platform_warnings(command: str) -> List[str]:
    warnings: List[str] = []
    lowered = str(command or "").lower()
    if os.name == "nt" or lowered.endswith(".cmd") or lowered.endswith(".bat"):
        warnings.append(
            "Windows: DSH spawns stdio servers without a shell; point --executable at "
            "evidencespine.exe (or a .exe launcher), not a .cmd/.bat shim."
        )
    return warnings


# ---------------------------------------------------------------------------
# textual merge primitives
# ---------------------------------------------------------------------------


def _managed_span(text: str) -> Optional[Tuple[int, int]]:
    """Return `(start, stop)` offsets of the managed block, or `None`."""
    begin = text.find(MANAGED_BEGIN)
    if begin < 0:
        return None
    line_start = text.rfind("\n", 0, begin) + 1
    if text[line_start:begin].strip():
        return None
    end = text.find(MANAGED_END, begin)
    if end < 0:
        return None
    end_newline = text.find("\n", end)
    stop = len(text) if end_newline < 0 else end_newline + 1
    return line_start, stop


def _managed_state(text: str) -> str:
    """Return `none`, `ok` or `unterminated` (BEGIN without END)."""
    if MANAGED_BEGIN not in text:
        return "none"
    return "ok" if _managed_span(text) is not None else "unterminated"


def _text_outside_block(text: str) -> str:
    span = _managed_span(text)
    if span is None:
        return text
    return text[: span[0]] + text[span[1] :]


def _first_content_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped in {"---", "..."}:
            continue
        return stripped
    return ""


def _yaml_sequence_ok(text: str) -> bool:
    """True when the file is empty/comments-only or a top-level YAML sequence."""
    first = _first_content_line(text)
    return not first or first.startswith("-")


def merge_patch_text(existing: str, block: str) -> Tuple[str, str]:
    """Merge *block* into *existing*; return `(new_text, action)`."""
    span = _managed_span(existing)
    if span is not None:
        new_text = existing[: span[0]] + block + existing[span[1] :]
        return new_text, ("unchanged" if new_text == existing else "updated")
    if not existing.strip():
        return block, "added"
    return existing.rstrip("\n") + "\n\n" + block, "added"


def removal_text(existing: str) -> Tuple[str, bool]:
    """Remove the managed block; return `(new_text, delete_file)`."""
    span = _managed_span(existing)
    if span is None:
        return existing, False
    head, tail = existing[: span[0]], existing[span[1] :]
    if head.endswith("\n\n"):
        # Undo the blank separator line appended by merge_patch_text().
        head = head[:-1]
    new_text = head + tail
    return new_text, not new_text.strip()


# ---------------------------------------------------------------------------
# file IO helpers
# ---------------------------------------------------------------------------


def _read_text(path: str) -> Tuple[str, str]:
    """Read a file; return `(text, error)` (a missing file is `("", "")`)."""
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return handle.read(), ""
    except FileNotFoundError:
        return "", ""
    except UnicodeDecodeError:
        return "", "file is not valid UTF-8"
    except OSError as exc:
        return "", str(exc)


def _atomic_write(path: str, text: str, *, mode: int = 0o600) -> None:
    """Write *text* atomically through a same-directory temp file."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".evidencespine-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _backup(path: str, original: str) -> str:
    """Write a timestamped `.bak-<utc>` copy of the pre-modification content."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    candidate = path + ".bak-" + stamp
    counter = 1
    while os.path.exists(candidate):
        candidate = path + ".bak-" + stamp + "-" + str(counter)
        counter += 1
    _atomic_write(candidate, original)
    return candidate


def _profile_coexistence_warnings(home: str) -> List[str]:
    """Warn when a profile patch already registers mcp-manager evidencespine."""
    warnings: List[str] = []
    profiles_dir = os.path.join(home, "profiles")
    try:
        names = sorted(os.listdir(profiles_dir))
    except OSError:
        return warnings
    for name in names:
        patch = os.path.join(profiles_dir, name, PATCH_FILENAME)
        text, _ = _read_text(patch)
        if not text:
            continue
        outside = _text_outside_block(text)
        if "mcp-manager" in outside and SERVER_NAME in outside:
            warnings.append(
                patch
                + " configures mcp-manager servers."
                + SERVER_NAME
                + " for profile "
                + repr(name)
                + "; two registrations of the same server name conflict - remove one "
                "(the GUI Settings editor writes this same profile file)."
            )
    return warnings


# ---------------------------------------------------------------------------
# composition verification (never fails the install)
# ---------------------------------------------------------------------------


def find_dsh_cli() -> str:
    """Locate the dsh launcher: `$DSH_CLI`, PATH, then the app bundle path."""
    env = str(os.environ.get("DSH_CLI", "") or "").strip()
    if env:
        return env
    which = shutil.which("dsh")
    if which:
        return which
    for candidate in _DSH_CLI_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return ""


def verify_composition(home: str, *, profiles: Tuple[str, ...] = ("headless", "web")) -> Dict[str, Any]:
    """Compose DSH config for *profiles* and report whether our row appears."""
    result: Dict[str, Any] = {
        "composed_headless": None,
        "composed_web": None,
        "error": "",
    }
    cli = find_dsh_cli()
    result["cli"] = cli
    if not cli:
        result["error"] = "dsh CLI not found (set DSH_CLI or install DeepSeek Harness)"
        return result
    env = dict(os.environ)
    env["DSH_HOME"] = os.path.abspath(os.path.expanduser(home or dsh_home()))
    errors: List[str] = []
    for name in profiles:
        key = "composed_" + name.replace("-", "_")
        try:
            proc = subprocess.run(
                [cli, "--dump-config", "--profile", name],
                capture_output=True,
                text=True,
                timeout=90,
                env=env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(name + ": " + str(exc))
            result[key] = None
            continue
        result[key] = bool(proc.returncode == 0 and ROW_ID in (proc.stdout or ""))
        if proc.returncode != 0:
            errors.append(name + ": dump-config failed with rc=" + str(proc.returncode))
    result["error"] = "; ".join(errors)
    return result


def _verification(home: str, *, verify: bool, profiles: Tuple[str, ...]) -> Dict[str, Any]:
    if not verify:
        return {
            "composed_headless": None,
            "composed_web": None,
            "error": "verification disabled",
        }
    if not profiles:
        return {
            "composed_headless": None,
            "composed_web": None,
            "error": "profile cannot be composed from the CLI",
        }
    return verify_composition(home, profiles=profiles)


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------


def _base_result(*, scope: str, home: str, path: str) -> Dict[str, Any]:
    return {
        "status": "ok",
        "harness": HARNESS,
        "scope": scope,
        "home": home,
        "path": path,
        "wrote": [],
        "warnings": [],
    }


def _fail(
    *,
    status: str,
    scope: str,
    home: str,
    path: str,
    reason: str,
    warnings: List[str],
    action: str = "",
    dry_run: bool = False,
) -> Dict[str, Any]:
    result = _base_result(scope=scope, home=home, path=path)
    result.update(
        {
            "status": status,
            "action": action,
            "reason": reason,
            "warnings": [*warnings, reason],
            "verification": {
                "composed_headless": None,
                "composed_web": None,
                "error": "verification not run: target is not safe to modify",
            },
        }
    )
    if dry_run:
        result["dry_run"] = True
    return result


def _maybe_materialize_skill(
    result: Dict[str, Any],
    *,
    target_dir: str,
    scope: str,
    executable: str,
    with_skill: bool,
) -> None:
    if not with_skill:
        return
    try:
        from evidencespine.harness import skills as skills_module

        skills_result = skills_module.materialize(
            target_dir=target_dir,
            harnesses=[HARNESS],
            executable=executable,
            scope=scope,
        )
        result["skill"] = skills_result
        for path in skills_result.get("wrote", []) or []:
            if path not in result["wrote"]:
                result["wrote"].append(path)
        for warning in skills_result.get("warnings", []) or []:
            if warning not in result["warnings"]:
                result["warnings"].append(warning)
    except Exception as exc:  # never fail the install on skill delivery
        result["warnings"].append("skill materialization failed: " + str(exc))


def _maybe_install_instructions(
    result: Dict[str, Any],
    *,
    target_dir: str,
    base_dir: str,
    scope: str,
    executable: str,
    dry_run: bool,
    enabled: bool,
) -> None:
    """Opt-in AGENTS.md managed block (--with-agents-md); never fails the install."""
    if not enabled:
        return
    try:
        from evidencespine.harness import skills as skills_module

        instructions = skills_module.install_instructions(
            target_dir=target_dir,
            base_dir=base_dir,
            scope=scope,
            executable=executable,
            dry_run=dry_run,
        )
        result["instructions"] = instructions
        for path in instructions.get("wrote", []) or []:
            if path not in result["wrote"]:
                result["wrote"].append(path)
        for warning in instructions.get("warnings", []) or []:
            if warning not in result["warnings"]:
                result["warnings"].append(warning)
    except Exception as exc:  # never fail the install on instructions delivery
        result["warnings"].append("AGENTS.md instructions failed: " + str(exc))


def install_dsh(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    scope: str = "project",
    executable: str = "",
    dry_run: bool = False,
    **kw: Any,
) -> Dict[str, Any]:
    """Install the EvidenceSpine MCP row into DSH's patch layer."""
    home = os.path.abspath(os.path.expanduser(str(kw.get("home") or dsh_home())))
    verify = bool(kw.get("verify", True))
    with_skill = bool(kw.get("with_skill", False))
    with_agents_md = bool(kw.get("with_agents_md", False))
    project_dir = str(kw.get("project_dir") or target_dir or os.getcwd())
    row, warnings = _resolve_row(
        executable=executable,
        base_dir=base_dir,
        project_dir=project_dir,
        target_dir=target_dir,
    )
    warnings.extend(_platform_warnings(str(row["config"]["command"])))
    kind, profile, scope_error = _parse_scope(scope)
    if scope_error:
        return _fail(
            status="invalid",
            scope=scope,
            home=home,
            path=patch_path(home),
            reason=scope_error,
            warnings=warnings,
            dry_run=dry_run,
        )
    if kind == "profile":
        profile_result = _install_profile(
            home=home,
            scope=scope,
            profile=profile,
            row=row,
            warnings=warnings,
            dry_run=dry_run,
            verify=verify,
            target_dir=target_dir,
            executable=executable,
            with_skill=with_skill,
        )
        _maybe_install_instructions(
            profile_result,
            target_dir=target_dir,
            base_dir=base_dir,
            scope=scope,
            executable=executable,
            dry_run=dry_run,
            enabled=with_agents_md,
        )
        return profile_result

    path = patch_path(home)
    existing, read_error = _read_text(path)
    if read_error:
        return _fail(
            status="error",
            scope=scope,
            home=home,
            path=path,
            reason="cannot read " + path + ": " + read_error,
            warnings=warnings,
            dry_run=dry_run,
        )
    state = _managed_state(existing)
    if state == "unterminated":
        return _fail(
            status="invalid",
            scope=scope,
            home=home,
            path=path,
            reason=path + " contains an unterminated EvidenceSpine managed block (missing END marker)",
            warnings=warnings,
            dry_run=dry_run,
        )
    if not _yaml_sequence_ok(existing):
        return _fail(
            status="invalid",
            scope=scope,
            home=home,
            path=path,
            reason=(
                path
                + " is not a top-level YAML sequence (first non-comment line: "
                + repr(_first_content_line(existing)[:60])
                + "); refusing to modify user YAML"
            ),
            warnings=warnings,
            dry_run=dry_run,
        )
    if state == "none" and ROW_ID in existing:
        return _fail(
            status="conflict",
            scope=scope,
            home=home,
            path=path,
            reason=(
                path
                + " already contains an unmanaged "
                + repr(ROW_ID)
                + " row; remove it (or wrap it in the EvidenceSpine sentinel block) "
                "instead of duplicating the registration"
            ),
            warnings=warnings,
            dry_run=dry_run,
        )

    warnings.extend(_profile_coexistence_warnings(home))
    block = render_patch_block(row)
    new_text, action = merge_patch_text(existing, block)
    result = _base_result(scope=scope, home=home, path=path)
    result["warnings"] = warnings
    result["action"] = action
    result["managed_block"] = {"begin": MANAGED_BEGIN, "end": MANAGED_END}
    if dry_run:
        result["dry_run"] = True
        result["content"] = new_text
        result["verification"] = {
            "composed_headless": None,
            "composed_web": None,
            "error": "skipped (dry_run)",
        }
        _maybe_install_instructions(
            result,
            target_dir=target_dir,
            base_dir=base_dir,
            scope=scope,
            executable=executable,
            dry_run=dry_run,
            enabled=with_agents_md,
        )
        return result

    backup = ""
    if action != "unchanged":
        if os.path.exists(path):
            backup = _backup(path, existing)
        _atomic_write(path, new_text)
        result["wrote"] = [path]
    result["backup"] = backup
    result["verification"] = _verification(home, verify=verify, profiles=("headless", "web"))
    _maybe_materialize_skill(
        result,
        target_dir=target_dir,
        scope=scope,
        executable=executable,
        with_skill=with_skill,
    )
    _maybe_install_instructions(
        result,
        target_dir=target_dir,
        base_dir=base_dir,
        scope=scope,
        executable=executable,
        dry_run=dry_run,
        enabled=with_agents_md,
    )
    return result


def _install_profile(
    *,
    home: str,
    scope: str,
    profile: str,
    row: Dict[str, Any],
    warnings: List[str],
    dry_run: bool,
    verify: bool,
    target_dir: str,
    executable: str,
    with_skill: bool,
) -> Dict[str, Any]:
    path = profile_patch_path(home, profile)
    if not os.path.isdir(os.path.dirname(path)):
        return _fail(
            status="invalid",
            scope=scope,
            home=home,
            path=path,
            reason=(
                "profile "
                + repr(profile)
                + " is not provisioned under "
                + os.path.dirname(path)
                + "; create it first (e.g. dsh plugin --profile "
                + profile
                + " add <package>)"
            ),
            warnings=warnings,
            dry_run=dry_run,
        )
    existing, read_error = _read_text(path)
    if read_error:
        return _fail(
            status="error",
            scope=scope,
            home=home,
            path=path,
            reason="cannot read " + path + ": " + read_error,
            warnings=warnings,
            dry_run=dry_run,
        )
    state = _managed_state(existing)
    if state == "unterminated":
        return _fail(
            status="invalid",
            scope=scope,
            home=home,
            path=path,
            reason=path + " contains an unterminated EvidenceSpine managed block",
            warnings=warnings,
            dry_run=dry_run,
        )
    if not _yaml_sequence_ok(existing):
        return _fail(
            status="invalid",
            scope=scope,
            home=home,
            path=path,
            reason=(
                path
                + " is not a top-level YAML sequence (first non-comment line: "
                + repr(_first_content_line(existing)[:60])
                + "); refusing to modify user YAML"
            ),
            warnings=warnings,
            dry_run=dry_run,
        )
    if state == "none" and "mcp-manager" in existing:
        warnings.append(
            "a user-managed mcp-manager row already exists in this profile; a second row would "
            "replace its servers map, so nothing was written - merge the snippet by hand or use "
            "the GUI Settings editor"
        )
        failure = _fail(
            status="conflict",
            scope=scope,
            home=home,
            path=path,
            reason=path + " already owns an mcp-manager row; refusing to erase Settings-managed servers",
            warnings=warnings,
            dry_run=dry_run,
        )
        failure["snippet"] = render_manager_block(profile, row)
        return failure

    block = render_manager_block(profile, row)
    new_text, action = merge_patch_text(existing, block)
    result = _base_result(scope=scope, home=home, path=path)
    result["profile"] = profile
    result["warnings"] = warnings
    result["action"] = action
    if dry_run:
        result["dry_run"] = True
        result["content"] = new_text
        result["verification"] = {
            "composed_headless": None,
            "composed_web": None,
            "error": "skipped (dry_run)",
        }
        return result
    backup = ""
    if action != "unchanged":
        if os.path.exists(path):
            backup = _backup(path, existing)
        _atomic_write(path, new_text)
        result["wrote"] = [path]
    result["backup"] = backup
    profiles: Tuple[str, ...] = (profile,) if profile in {"headless", "web"} else ()
    result["verification"] = _verification(home, verify=verify, profiles=profiles)
    _maybe_materialize_skill(
        result,
        target_dir=target_dir,
        scope=scope,
        executable=executable,
        with_skill=with_skill,
    )
    return result


def uninstall_dsh(
    *,
    target_dir: str = ".",
    base_dir: str = ".evidencespine",
    scope: str = "project",
    executable: str = "",
    dry_run: bool = False,
    **kw: Any,
) -> Dict[str, Any]:
    """Remove only the EvidenceSpine managed block from the DSH patch layer."""
    home = os.path.abspath(os.path.expanduser(str(kw.get("home") or dsh_home())))
    verify = bool(kw.get("verify", True))
    kind, profile, scope_error = _parse_scope(scope)
    if scope_error:
        return _fail(
            status="invalid",
            scope=scope,
            home=home,
            path=patch_path(home),
            reason=scope_error,
            warnings=[],
            dry_run=dry_run,
        )
    path = profile_patch_path(home, profile) if kind == "profile" else patch_path(home)
    existing, read_error = _read_text(path)
    if read_error:
        return _fail(
            status="error",
            scope=scope,
            home=home,
            path=path,
            reason="cannot read " + path + ": " + read_error,
            warnings=[],
            dry_run=dry_run,
        )
    state = _managed_state(existing)
    if state == "unterminated":
        return _fail(
            status="invalid",
            scope=scope,
            home=home,
            path=path,
            reason=path + " contains an unterminated EvidenceSpine managed block",
            warnings=[],
            dry_run=dry_run,
        )
    if state == "none":
        result = _base_result(scope=scope, home=home, path=path)
        result.update(
            {
                "action": "absent",
                "note": "no EvidenceSpine managed block found",
                "verification": {
                    "composed_headless": None,
                    "composed_web": None,
                    "error": "not run: nothing installed",
                },
            }
        )
        return result
    new_text, delete_file = removal_text(existing)
    result = _base_result(scope=scope, home=home, path=path)
    result["action"] = "removed"
    if kind == "profile":
        result["profile"] = profile
    if dry_run:
        result["dry_run"] = True
        result["content"] = new_text
        result["would_remove_file"] = delete_file
        result["verification"] = {
            "composed_headless": None,
            "composed_web": None,
            "error": "skipped (dry_run)",
        }
        return result
    backup = _backup(path, existing)
    if delete_file:
        try:
            os.unlink(path)
        except OSError:
            _atomic_write(path, new_text)
        result["removed"] = [path]
    else:
        _atomic_write(path, new_text)
    result["backup"] = backup
    result["wrote"] = [path]
    profiles: Tuple[str, ...] = (
        ("headless", "web") if kind == "home" else ((profile,) if profile in {"headless", "web"} else ())
    )
    verification = _verification(home, verify=verify, profiles=profiles)
    verification["expected"] = "absent"
    result["verification"] = verification
    return result
