"""Universal MCP client delivery: canonical file, projections, print-config renderer.

The 2026 MCP client landscape has exactly one portable shape: stdio plus the
`{"mcpServers": {<name>: {"command": ..., "args": [...]}}}` object. This module
owns the lowest-common-denominator delivery:

* `install_generic` writes the canonical `<target_dir>/evidencespine.mcp.json`
  (merge-preserving) and non-destructive project-scope projections that the
  surveyed clients read directly: `.mcp.json` (Claude Code project / VS Code
  Copilot portable / Warp), `.cursor/mcp.json`, `.agents/mcp_config.json`
  (Antigravity), `.agents/.mcp.json` (Warp), `.continue/mcpServers/`,
  `.junie/mcp/mcp.json` and `.roo/mcp.json`. Windsurf/Devin is *not*
  written by default (user-global config, no documented project file).
* `render_client_config` returns a copy-paste snippet, the target path(s), a
  verification command and a confidence grade for every surveyed client; the
  CLI's `harness print-config` prints it and never writes.
* `uninstall_generic` removes only the EvidenceSpine entry, preserving every
  other server and top-level key; empty files it created are deleted.

Shapes are grounded in vendor documentation fetched 2026-11-20 (see the repo
research notes); uncertain path assumptions carry an explicit note and a lower
grade so the CLI can surface them. stdlib only: no PyYAML anywhere.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

HARNESS = "generic"
SERVER_NAME = "evidencespine"
CANONICAL_FILENAME = "evidencespine.mcp.json"
NOTES_FILENAME = "evidencespine.mcp.md"
MAX_NOTES_LINES = 40

#: Project-scope files written by install_generic (relative to target_dir).
DEFAULT_PROJECTIONS: Tuple[str, ...] = (
    ".mcp.json",
    os.path.join(".cursor", "mcp.json"),
    os.path.join(".agents", "mcp_config.json"),
    os.path.join(".agents", ".mcp.json"),
    os.path.join(".continue", "mcpServers", "evidencespine.json"),
    os.path.join(".junie", "mcp", "mcp.json"),
    os.path.join(".roo", "mcp.json"),
)

#: Clients accepted by render_client_config (canonical names).
SUPPORTED_CLIENTS: Tuple[str, ...] = (
    "claude-code",
    "claude-desktop",
    "codex",
    "dsh",
    "cursor",
    "vscode",
    "vscode-copilot",
    "windsurf",
    "zed",
    "continue",
    "gemini",
    "opencode",
    "antigravity",
    "cline",
    "roo",
    "junie",
    "warp",
    "amp",
    "jetbrains",
    "generic",
)

_CLIENT_ALIASES: Dict[str, str] = {
    "claude": "claude-code",
    "claude-code-cli": "claude-code",
    "claude-desktop-app": "claude-desktop",
    "devin": "windsurf",
    "devin-desktop": "windsurf",
    "windsurf-devin": "windsurf",
    "vscode-legacy": "vscode",
    "vscode-local": "vscode",
    "vscode-deprecated": "vscode",
    "copilot": "vscode-copilot",
    "gemini-cli": "gemini",
    "roo-code": "roo",
    "amp-code": "amp",
    "ampcode": "amp",
    "jetbrains-ai": "jetbrains",
    "intellij": "jetbrains",
    "generic-mcp": "generic",
    "universal": "generic",
}


# ---------------------------------------------------------------------------
# small path / JSON helpers
# ---------------------------------------------------------------------------


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path or ".")))


def _project_path(project_dir: str, *rel: str) -> str:
    rel_path = os.path.join(*rel)
    if str(project_dir or "").strip():
        return os.path.join(_expand_path(project_dir), rel_path)
    return "<repo>/" + rel_path.replace(os.sep, "/")


def _user_path(*parts: str) -> str:
    return os.path.join(os.path.expanduser("~"), *parts)


def _claude_desktop_config_path() -> str:
    if sys.platform == "darwin":
        return os.path.join(
            os.path.expanduser("~"), "Library", "Application Support", "Claude", "claude_desktop_config.json"
        )
    if os.name == "nt":
        base = os.environ.get("APPDATA", "") or os.path.expanduser("~")
        return os.path.join(base, "Claude", "claude_desktop_config.json")
    return os.path.join(os.path.expanduser("~"), ".config", "Claude", "claude_desktop_config.json")


def _devin_config_path() -> str:
    if os.name == "nt":
        base = os.environ.get("APPDATA", "") or os.path.expanduser("~")
        return os.path.join(base, "devin", "mcp_config.json")
    xdg = str(os.environ.get("XDG_CONFIG_HOME", "") or "").strip()
    base = xdg or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "devin", "mcp_config.json")


def _copilot_user_config_path() -> str:
    copilot_home = str(os.environ.get("COPILOT_HOME", "") or "").strip()
    base = copilot_home or os.path.join(os.path.expanduser("~"), ".copilot")
    return os.path.join(base, "mcp-config.json")


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True) + "\n"


def _read_existing(path: str) -> Tuple[Optional[str], str]:
    """Return (text_or_None, error); a missing file is (None, "")."""
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return handle.read(), ""
    except FileNotFoundError:
        return None, ""
    except UnicodeDecodeError:
        return None, "file is not valid UTF-8"
    except OSError as exc:
        return None, str(exc)


def _atomic_write_text(path: str, text: str) -> None:
    """Write text atomically; preserve an existing file's mode, new files 0644."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    mode = 0o644
    try:
        mode = os.stat(path).st_mode & 0o777
    except OSError:
        pass
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


def _atomic_write_json(path: str, payload: Any) -> None:
    _atomic_write_text(path, _json_text(payload))


# ---------------------------------------------------------------------------
# canonical mcpServers shapes
# ---------------------------------------------------------------------------


def _argv(executable: str) -> List[str]:
    try:
        argv = shlex.split(str(executable or ""))
    except ValueError:
        argv = []
    return argv or ["evidencespine"]


def build_server_entry(*, executable: str = "", base_dir: str = ".evidencespine") -> Dict[str, Any]:
    """Build the portable stdio entry: command + args (no invented fields)."""
    argv = _argv(executable)
    return {"command": argv[0], "args": [*argv[1:], "mcp", "--base-dir", base_dir]}


def build_canonical_config(*, executable: str = "", base_dir: str = ".evidencespine") -> Dict[str, Any]:
    """Build the canonical mcpServers payload (includes an explicit empty env)."""
    entry = build_server_entry(executable=executable, base_dir=base_dir)
    entry["env"] = {}
    return {"mcpServers": {SERVER_NAME: entry}}


def merge_mcp_config(existing: Any, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Merge one server entry into an existing mcpServers payload, preserving the rest."""
    payload: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    servers = payload.get("mcpServers")
    merged: Dict[str, Any] = dict(servers) if isinstance(servers, dict) else {}
    merged[SERVER_NAME] = dict(entry)
    payload["mcpServers"] = merged
    return payload


def _without_server(payload: Any) -> Dict[str, Any]:
    """Return a copy with our server removed; {} means the file can be deleted."""
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, Any] = dict(payload)
    servers = out.get("mcpServers")
    if isinstance(servers, dict) and SERVER_NAME in servers:
        remaining = dict(servers)
        remaining.pop(SERVER_NAME, None)
        out["mcpServers"] = remaining
    if set(out) == {"mcpServers"} and not out.get("mcpServers"):
        return {}
    return out


# ---------------------------------------------------------------------------
# per-client snippets
# ---------------------------------------------------------------------------


def _canonical_client(client: str) -> str:
    name = str(client or "").strip().lower().replace("_", "-")
    name = _CLIENT_ALIASES.get(name, name)
    if name not in SUPPORTED_CLIENTS:
        raise ValueError(
            "unknown client: " + repr(client) + " (supported: " + ", ".join(SUPPORTED_CLIENTS) + ")"
        )
    return name


def _pick_path(paths: Dict[str, str], scope: str) -> str:
    prefer_user = str(scope or "project").strip().lower() in {"global", "user"}
    if prefer_user and paths.get("user"):
        return paths["user"]
    if paths.get("project"):
        return paths["project"]
    return paths.get("user", "")


def _result(
    client: str,
    *,
    fmt: str,
    paths: Dict[str, str],
    snippet: str,
    verify: str,
    grade: str,
    scope: str,
    note: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "client": client,
        "format": fmt,
        "path": _pick_path(paths, scope),
        "paths": paths,
        "snippet": snippet,
        "verify": verify,
        "grade": grade,
    }
    if note:
        data["note"] = note
    data.update(extra)
    return data


def _codex_toml(executable: str, base_dir: str) -> str:
    entry = build_server_entry(executable=executable, base_dir=base_dir)
    lines = [
        "# EvidenceSpine - Codex MCP server (append to config.toml; do not duplicate the table).",
        "[mcp_servers.evidencespine]",
        "command = " + json.dumps(entry["command"], ensure_ascii=True),
        "args = " + json.dumps(entry["args"], ensure_ascii=True),
        "",
    ]
    return "\n".join(lines)


def _continue_yaml(executable: str, base_dir: str) -> str:
    entry = build_server_entry(executable=executable, base_dir=base_dir)
    lines = [
        "name: evidencespine",
        "version: 0.1.0",
        "schema: v1",
        "mcpServers:",
        "  - name: evidencespine",
        "    type: stdio",
        "    command: " + json.dumps(entry["command"], ensure_ascii=True),
        "    args:",
    ]
    for arg in entry["args"]:
        lines.append("      - " + json.dumps(arg, ensure_ascii=True))
    lines.append("")
    return "\n".join(lines)


def _opencode_payload(executable: str, base_dir: str) -> Dict[str, Any]:
    argv = _argv(executable)
    return {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {
            "servers": {
                SERVER_NAME: {
                    "type": "local",
                    "command": [*argv, "mcp", "--base-dir", base_dir],
                }
            }
        },
    }


DEFAULT_HTTP_URL = "http://127.0.0.1:8000/mcp"
AUTH_TOKEN_ENV = "EVIDENCESPINE_MCP_AUTH_TOKEN"
HTTP_FORMAT_CLIENTS = (
    "claude-code",
    "codex",
    "cursor",
    "dsh",
    "vscode",
    "vscode-copilot",
    "windsurf",
    "zed",
    "continue",
    "gemini",
    "opencode",
    "antigravity",
    "cline",
    "roo",
    "junie",
    "warp",
    "amp",
    "jetbrains",
    "generic",
)


def _bearer_header(style: str = "") -> Dict[str, str]:
    """Authorization header using the client's documented env-interpolation dialect."""
    tokens = {
        "claude-code": "Bearer ${" + AUTH_TOKEN_ENV + "}",
        "vscode": "Bearer ${" + AUTH_TOKEN_ENV + "}",
        "cursor": "Bearer ${env:" + AUTH_TOKEN_ENV + "}",
        "gemini": "Bearer $" + AUTH_TOKEN_ENV,
        "opencode": "Bearer {env:" + AUTH_TOKEN_ENV + "}",
        "amp": "Bearer ${" + AUTH_TOKEN_ENV + "}",
    }
    literal = "Bearer <" + AUTH_TOKEN_ENV + ">"
    return {"Authorization": tokens.get(style, literal)}


def _http_codex_toml(http_url: str) -> str:
    return "\n".join(
        [
            "# EvidenceSpine - Codex streamable-HTTP MCP server (append to config.toml).",
            "[mcp_servers.evidencespine]",
            "url = " + json.dumps(http_url, ensure_ascii=True),
            "bearer_token_env_var = " + json.dumps(AUTH_TOKEN_ENV, ensure_ascii=True),
            "",
        ]
    )


def _http_continue_yaml(http_url: str) -> str:
    return "\n".join(
        [
            "name: evidencespine",
            "version: 0.1.0",
            "schema: v1",
            "mcpServers:",
            "  - name: evidencespine",
            "    type: streamable-http",
            "    url: " + json.dumps(http_url, ensure_ascii=True),
            "    requestOptions:",
            "      headers:",
            "        Authorization: Bearer <" + AUTH_TOKEN_ENV + ">",
            "",
        ]
    )


def _render_http_client(
    name: str,
    *,
    paths: Dict[str, str],
    http_url: str,
    scope: str,
    verify: str,
) -> Optional[Dict[str, Any]]:
    """Render the streamable-HTTP snippet for one client (None = no file-based form)."""
    if name == "claude-desktop":
        return None

    def result(
        fmt: str,
        snippet: str,
        *,
        note: str,
        http_grade: str = "C",
        **extra: Any,
    ) -> Dict[str, Any]:
        return _result(
            name,
            fmt=fmt,
            paths=paths,
            snippet=snippet,
            verify=verify,
            grade=http_grade,
            scope=scope,
            note=note,
            **extra,
        )

    def mcp_payload(server: Dict[str, Any]) -> str:
        return _json_text({"mcpServers": {SERVER_NAME: server}})

    if name == "claude-code":
        server = {"type": "http", "url": http_url, "headers": _bearer_header("claude-code")}
        return result(
            "json",
            mcp_payload(server),
            note="Claude Code requires type 'http' for URL entries (a url without type is a "
            "config error). The header supports " + "${VAR} interpolation.",
        )

    if name == "codex":
        toml = _http_codex_toml(http_url)
        return result(
            "toml",
            toml,
            note="Codex HTTP mode uses url + bearer_token_env_var (no command/args); add an "
            "http_headers table for extra headers.",
            toml=toml,
        )

    if name == "dsh":
        from evidencespine.harness import dsh as dsh_module

        snippet = dsh_module.render_http_patch_block(http_url, auth_env=AUTH_TOKEN_ENV)
        return result(
            "yaml",
            snippet,
            note="DSH resolves headers through a Loader !!js expression at composition time; a "
            "plain string header is NOT interpolated. Keep the token in the host environment.",
            yaml=snippet,
        )

    if name == "cursor":
        server = {"url": http_url, "headers": _bearer_header("cursor")}
        return result(
            "json",
            mcp_payload(server),
            note="Cursor remote entries use url+headers (no type required); "
            + "${env:NAME} interpolation is supported.",
        )

    if name == "vscode-copilot":
        server = {"type": "http", "url": http_url, "headers": _bearer_header("vscode")}
        return result(
            "json",
            mcp_payload(server),
            note="Portable Copilot HTTP shape (type http + url + headers); "
            + "${NAME} interpolation is supported.",
        )

    if name == "vscode":
        legacy = {"type": "http", "url": http_url, "headers": _bearer_header("vscode")}
        payload = {"servers": {SERVER_NAME: legacy}, "inputs": []}
        return result(
            "json",
            _json_text(payload),
            note="Deprecated editor-managed Local-harness shape: the server entry lives under "
            "the top-level 'servers' key.",
        )

    if name == "windsurf":
        server = {"serverUrl": http_url, "headers": _bearer_header()}
        return result(
            "json",
            mcp_payload(server),
            note="Windsurf/Devin remote key is serverUrl (the url alias is accepted) plus "
            "headers; header interpolation token syntax is not documented ([U]).",
        )

    if name == "zed":
        server = {"url": http_url, "headers": _bearer_header()}
        return result(
            "json",
            _json_text({"context_servers": {SERVER_NAME: server}}),
            http_grade="I",
            note="Zed remote context_servers use url+headers with an Authorization header "
            "(without one Zed starts its OAuth prompt). The settings path is conventional XDG "
            "([I]); for remote hosts use an ssh tunnel/gateway rather than a public bind.",
        )

    if name == "continue":
        yaml_text = _http_continue_yaml(http_url)
        return result(
            "yaml",
            yaml_text,
            note="Continue remote servers require type 'streamable-http' + url; headers pass "
            "through requestOptions (exact nesting not captured in research, [U]).",
            yaml=yaml_text,
        )

    if name == "gemini":
        server = {"httpUrl": http_url, "headers": _bearer_header("gemini")}
        return result(
            "json",
            mcp_payload(server),
            note="Gemini CLI uses httpUrl for streamable HTTP (url is the legacy SSE key); "
            + "$VAR expands from the process env.",
        )

    if name == "opencode":
        payload = {
            "$schema": "https://opencode.ai/config.json",
            "mcp": {
                "servers": {
                    SERVER_NAME: {
                        "type": "remote",
                        "url": http_url,
                        "headers": _bearer_header("opencode"),
                        "oauth": False,
                    }
                }
            },
        }
        return result(
            "json",
            _json_text(payload),
            note="OpenCode V2 remote servers use type 'remote' + url; oauth:false keeps the "
            "static Bearer header (otherwise it runs OAuth PKCE).",
        )

    if name == "antigravity":
        server = {"serverUrl": http_url, "headers": _bearer_header()}
        return result(
            "json",
            mcp_payload(server),
            note="Antigravity prefers serverUrl (url is accepted as an alias); SSE and httpUrl "
            "are unsupported. Header env interpolation is not documented ([U]).",
        )

    if name == "cline":
        cline_server: Dict[str, Any] = {
            "type": "streamableHttp",
            "url": http_url,
            "headers": _bearer_header(),
            "disabled": False,
            "autoApprove": [],
        }
        return result(
            "json",
            mcp_payload(cline_server),
            note="Cline requires type 'streamableHttp' for streamable HTTP (omitting type "
            "defaults to legacy SSE).",
        )

    if name == "roo":
        server = {"type": "streamable-http", "url": http_url, "headers": _bearer_header()}
        return result(
            "json",
            mcp_payload(server),
            note="Roo Code requires type 'streamable-http' plus url and headers.",
        )

    if name == "junie":
        server = {"url": http_url, "headers": _bearer_header()}
        return result(
            "json",
            mcp_payload(server),
            note="Junie remote entries use url+headers with a Bearer token; timeout (ms) is optional.",
        )

    if name == "warp":
        server = {"url": http_url, "headers": _bearer_header()}
        return result(
            "json",
            mcp_payload(server),
            http_grade="U",
            note="Warp supports streamable HTTP with custom headers, but the exact remote config "
            "keys were not captured from the docs ([U]) - verify in the Warp MCP panel.",
        )

    if name == "amp":
        server = {"url": http_url, "headers": _bearer_header("amp")}
        payload = {"amp.mcpServers": {SERVER_NAME: server}}
        return result(
            "json",
            _json_text(payload),
            note="Amp remote entries use url+headers with " + "${VAR} interpolation; remote "
            "definitions can also live in the account (amp:id-token for orb identity).",
        )

    if name == "jetbrains":
        server = {"url": http_url, "headers": _bearer_header()}
        return result(
            "json",
            mcp_payload(server),
            http_grade="U",
            note="UI-managed only ([U]): paste this url+headers JSON into Settings > Tools > AI "
            "Assistant > Model Context Protocol (MCP).",
        )

    server = {"url": http_url, "headers": _bearer_header()}
    return result(
        "json",
        mcp_payload(server),
        http_grade="I",
        note="Lowest common denominator for remote: streamable-HTTP url + Authorization header "
        "(valid for every surveyed client except Claude Desktop).",
    )


def _render_known_client(
    name: str,
    *,
    executable: str,
    base_dir: str,
    project_dir: str,
    scope: str,
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    """Render one known client's MCP snippet (internal; see render_client_config)."""
    if name == "claude-code":
        paths = {"project": _project_path(project_dir, ".mcp.json"), "user": _user_path(".claude.json")}
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"mcpServers": {SERVER_NAME: entry}}),
            verify="claude mcp list  (or /mcp inside a session)",
            grade="C",
            scope=scope,
            note="Project scope .mcp.json; user scope ~/.claude.json is normally written by "
            + repr("claude mcp add --scope user --transport stdio evidencespine -- evidencespine mcp"),
        )

    if name == "claude-desktop":
        paths = {"user": _claude_desktop_config_path()}
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"mcpServers": {SERVER_NAME: entry}}),
            verify="Restart Claude Desktop and check the tools/connectors menu",
            grade="C",
            scope=scope,
            note="Remote (HTTP) servers cannot be expressed in this file; add them via "
            "Settings > Connectors. The JSON file is stdio-only in practice.",
        )

    if name == "codex":
        toml = _codex_toml(executable, base_dir)
        paths = {
            "project": _project_path(project_dir, ".codex", "config.toml"),
            "user": _user_path(".codex", "config.toml"),
        }
        return _result(
            name,
            fmt="toml",
            paths=paths,
            snippet=toml,
            verify="codex mcp list",
            grade="C",
            scope=scope,
            note="Project .codex/config.toml only applies to trusted projects; append the table "
            "once (a second [mcp_servers.evidencespine] header is invalid TOML).",
            toml=toml,
        )

    if name == "dsh":
        from evidencespine.harness import dsh as dsh_module

        target = str(project_dir or "").strip() or os.getcwd()
        row = dsh_module.build_insert_row(
            executable=executable,
            base_dir=base_dir,
            project_dir=target,
            target_dir=target,
        )
        home = dsh_module.dsh_home()
        cli = dsh_module.find_dsh_cli() or "dsh"
        snippet = dsh_module.render_patch_block(row)
        paths = {
            "project": _project_path(project_dir, ".dsh", "evidencespine.dsh.patch.yml"),
            "user": dsh_module.patch_path(home),
        }
        return _result(
            name,
            fmt="yaml",
            paths=paths,
            snippet=snippet,
            verify=(
                "DSH_HOME=" + home + " " + cli + " --dump-config --profile headless | grep -A12 evidencespine-mcp"
            ),
            grade="C",
            scope=scope,
            note="The user patch layer applies to every DSH profile (web/desktop/headless). The "
            "project file only works with 'dsh web --patch <file>'; DSH does not auto-discover "
            "repo MCP files. Prefer: evidencespine harness install --harness dsh.",
            yaml=snippet,
        )

    if name == "cursor":
        paths = {
            "project": _project_path(project_dir, ".cursor", "mcp.json"),
            "user": _user_path(".cursor", "mcp.json"),
        }
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"mcpServers": {SERVER_NAME: entry}}),
            verify="cursor mcp list  (or Cursor Settings > MCP)",
            grade="C",
            scope=scope,
            note="Project scope .cursor/mcp.json; user scope ~/.cursor/mcp.json. Cursor also reads "
            "AGENTS.md, and .agents/skills/ for skills.",
        )

    if name == "vscode-copilot":
        portable = {"type": "stdio", **entry}
        paths = {
            "project": _project_path(project_dir, ".mcp.json"),
            "user": _copilot_user_config_path(),
        }
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"mcpServers": {SERVER_NAME: portable}}),
            verify="VS Code: run 'MCP: List Servers' from the Command Palette",
            grade="C",
            scope=scope,
            note="Portable Copilot shape (mcpServers in .mcp.json / ~/.copilot/mcp-config.json): "
            "the preferred VS Code Copilot format. The vscode client is the deprecated legacy "
            ".vscode/mcp.json 'servers' shape.",
        )

    if name == "vscode":
        legacy = {"type": "stdio", **entry}
        paths = {"project": _project_path(project_dir, ".vscode", "mcp.json")}
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"servers": {SERVER_NAME: legacy}, "inputs": []}),
            verify="VS Code: run 'MCP: List Servers' from the Command Palette",
            grade="C",
            scope=scope,
            note="Legacy editor-managed Local-harness format (.vscode/mcp.json, top-level "
            "'servers'), deprecated for new configs; use vscode-copilot for portable mcpServers.",
        )

    if name == "windsurf":
        paths = {"user": _devin_config_path()}
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"mcpServers": {SERVER_NAME: entry}}),
            verify="Windsurf/Devin: Cascade MCP panel (reload the window)",
            grade="C",
            scope=scope,
            note="Windsurf is now Devin Desktop and MCP config is user-global (no documented "
            "project file): ~/.config/devin/mcp_config.json. Legacy discovery path: "
            "~/.codeium/windsurf/mcp_config.json.",
            legacy_paths={"user": _user_path(".codeium", "windsurf", "mcp_config.json")},
        )

    if name == "zed":
        zed_entry = {"command": entry["command"], "args": entry["args"], "env": {}}
        paths = {
            "project": _project_path(project_dir, ".zed", "settings.json"),
            "user": _user_path(".config", "zed", "settings.json"),
        }
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"context_servers": {SERVER_NAME: zed_entry}}),
            verify="Zed: open the Agent panel settings (zed: open settings)",
            grade="I",
            scope=scope,
            note="Zed reads MCP servers from 'context_servers' inside settings.json. The user path "
            "~/.config/zed/settings.json is the conventional XDG location ([I]); merge the object "
            "into the existing settings file.",
        )

    if name == "continue":
        paths = {
            "project": _project_path(project_dir, ".continue", "mcpServers", "evidencespine.yaml"),
            "user": _user_path(".continue", "config.yaml"),
        }
        yaml_text = _continue_yaml(executable, base_dir)
        return _result(
            name,
            fmt="yaml",
            paths=paths,
            snippet=yaml_text,
            verify="Continue: reload the window and check that MCP tools are listed",
            grade="C",
            scope=scope,
            note="Project scope: drop the standalone block into .continue/mcpServers/ (Continue "
            "auto-imports JSON/YAML dropped there). User scope: merge it under mcpServers: in "
            "~/.continue/config.yaml.",
            yaml=yaml_text,
        )

    if name == "gemini":
        paths = {
            "project": _project_path(project_dir, ".gemini", "settings.json"),
            "user": _user_path(".gemini", "settings.json"),
        }
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"mcpServers": {SERVER_NAME: entry}}),
            verify="gemini mcp list",
            grade="C",
            scope=scope,
            note="Gemini CLI shape (mcpServers). Antigravity uses a different file; see the "
            "antigravity client.",
        )

    if name == "opencode":
        paths = {
            "project": _project_path(project_dir, "opencode.json"),
            "user": _user_path(".config", "opencode", "opencode.json"),
        }
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text(_opencode_payload(executable, base_dir)),
            verify="opencode mcp list",
            grade="C",
            scope=scope,
            note="V2 shape: server names live under mcp.servers (not directly under mcp) and "
            "command is an argv array. Project files may also be .opencode/opencode.json(c).",
        )

    if name == "antigravity":
        paths = {
            "project": _project_path(project_dir, ".agents", "mcp_config.json"),
            "user": _user_path(".gemini", "config", "mcp_config.json"),
        }
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"mcpServers": {SERVER_NAME: entry}}),
            verify="Antigravity: MCP settings panel",
            grade="C",
            scope=scope,
            note="Workspace file .agents/mcp_config.json; global file "
            "~/.gemini/config/mcp_config.json. SSE and httpUrl are unsupported (stdio or "
            "serverUrl only).",
        )

    if name == "cline":
        cline_entry = {**entry, "disabled": False, "autoApprove": []}
        paths = {"user": _user_path(".cline", "data", "settings", "cline_mcp_settings.json")}
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"mcpServers": {SERVER_NAME: cline_entry}}),
            verify="cline mcp list  (or the Cline MCP Servers panel)",
            grade="C",
            scope=scope,
            note="Cline keeps MCP config in user-global settings shared by IDE/CLI/SDK; "
            "CLINE_DIR / CLINE_MCP_SETTINGS_PATH override the location. No project file is "
            "documented.",
        )

    if name == "roo":
        roo_entry = dict(entry)
        if str(project_dir or "").strip():
            roo_entry["cwd"] = _expand_path(project_dir)
        paths = {"project": _project_path(project_dir, ".roo", "mcp.json")}
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"mcpServers": {SERVER_NAME: roo_entry}}),
            verify="Roo Code: MCP Servers tab (reload the window)",
            grade="C",
            scope=scope,
            note="Project .roo/mcp.json is documented; the global mcp_settings.json lives in VS "
            "Code globalStorage and its exact path is installation-dependent ([U]) - configure "
            "the global server through the Roo UI.",
        )

    if name == "junie":
        paths = {
            "project": _project_path(project_dir, ".junie", "mcp", "mcp.json"),
            "user": _user_path(".junie", "mcp", "mcp.json"),
        }
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"mcpServers": {SERVER_NAME: entry}}),
            verify="junie mcp list",
            grade="C",
            scope=scope,
            note="Project scope .junie/mcp/mcp.json; user scope ~/.junie/mcp/mcp.json.",
        )

    if name == "warp":
        warp_entry = {"command": entry["command"], "args": entry["args"]}
        paths = {
            "project": _project_path(project_dir, ".warp", ".mcp.json"),
            "user": _user_path(".warp", ".mcp.json"),
        }
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"mcpServers": {SERVER_NAME: warp_entry}}),
            verify="Warp: MCP servers panel (or /mcp)",
            grade="C",
            scope=scope,
            note="Warp also inherits project .mcp.json, ~/.claude.json, ~/.codex/config.toml and "
            "the .agents/ configs, so the generic projection usually covers Warp already.",
        )

    if name == "amp":
        paths = {
            "project": _project_path(project_dir, ".amp", "settings.json"),
            "user": _user_path(".config", "amp", "settings.json"),
        }
        return _result(
            name,
            fmt="json",
            paths=paths,
            snippet=_json_text({"amp.mcpServers": {SERVER_NAME: entry}}),
            verify="amp mcp list",
            grade="I",
            scope=scope,
            note="The Amp settings key is literally 'amp.mcpServers' (a dot inside the JSON "
            "key), not a nested object; merge it into the existing settings.json.",
        )

    if name == "jetbrains":
        return _result(
            "jetbrains",
            fmt="json",
            paths={},
            snippet=_json_text({"mcpServers": {SERVER_NAME: entry}}),
            verify="Settings > Tools > AI Assistant > Model Context Protocol (MCP): paste the "
            "JSON, or use 'Import from Claude'",
            grade="U",
            scope=scope,
            note="UI-managed only: no portable config file path is documented ([U]). Paste the "
            "snippet into the JetBrains AI Assistant MCP settings.",
        )

    # name == "generic"
    paths = {"project": _project_path(project_dir, CANONICAL_FILENAME)}
    result = _result(
        name,
        fmt="json",
        paths=paths,
        snippet=_json_text(build_canonical_config(executable=executable, base_dir=base_dir)),
        verify="python3 -m json.tool " + CANONICAL_FILENAME,
        grade="I",
        scope=scope,
        note="Lowest common denominator: stdio + mcpServers; most MCP clients import this "
        "object unchanged. Use: evidencespine harness install --harness generic.",
    )
    return result


def render_client_config(
    client: str,
    *,
    executable: str = "",
    base_dir: str = ".evidencespine",
    project_dir: str = "",
    **kw: Any,
) -> Dict[str, Any]:
    """Render one client's copy-paste MCP snippet plus path/verify/grade metadata.

    Returns a dict with at least client, format, path, paths, snippet, verify and
    grade; note carries uncertainty, and format-specific convenience keys (toml
    for Codex, yaml for DSH/Continue) are additive.
    """
    requested = str(client or "")
    name = _canonical_client(requested)
    scope = str(kw.get("scope") or "project").strip().lower()
    transport = str(kw.get("transport") or "stdio").strip().lower()
    if transport not in {"", "stdio", "http"}:
        transport = "stdio"
    entry = build_server_entry(executable=executable, base_dir=base_dir)
    result = _render_known_client(
        name,
        executable=executable,
        base_dir=base_dir,
        project_dir=project_dir,
        scope=scope,
        entry=entry,
    )
    if requested.strip().lower().replace("_", "-") != name:
        result["requested_client"] = requested
    result["transport"] = "http" if transport == "http" else "stdio"
    if transport == "http":
        http_url = str(kw.get("url") or kw.get("mcp_url") or DEFAULT_HTTP_URL).strip() or DEFAULT_HTTP_URL
        http_result = _render_http_client(
            name,
            paths=result.get("paths", {}),
            http_url=http_url,
            scope=scope,
            verify=str(result.get("verify", "")),
        )
        if http_result is None:
            note = str(result.get("note") or "").strip()
            remote_note = (
                "Remote MCP servers for Claude Desktop are UI-only (Settings > Connectors); the "
                "config JSON is stdio-only, so no url entry is emitted here. Keep stdio for this "
                "file and register the remote server through the UI."
            )
            result["note"] = (note + " " + remote_note).strip() if note else remote_note
            result["grade"] = "U"
        else:
            result = http_result
            result["transport"] = "http"
        result["requested_transport"] = "http"
        result["http_url"] = http_url
    return result


# ---------------------------------------------------------------------------
# setup notes
# ---------------------------------------------------------------------------


def render_setup_notes(*, base_dir: str = ".evidencespine", executable: str = "") -> str:
    """Render the sibling evidencespine.mcp.md setup notes (at most 40 lines)."""
    command = executable or "evidencespine"
    lines: List[str] = [
        "# EvidenceSpine MCP setup (generated file - safe to delete)",
        "",
        "Written by: evidencespine harness install --harness generic",
        "Canonical registration: " + CANONICAL_FILENAME + " (mcpServers + stdio).",
        "Server command: " + command + " mcp --base-dir " + str(base_dir),
        "",
        "Project projections (merge-preserving, keep them in version control):",
        "- .mcp.json - Claude Code (project), VS Code Copilot portable, Warp",
        "- .cursor/mcp.json - Cursor",
        "- .agents/mcp_config.json - Antigravity workspace",
        "- .agents/.mcp.json - Warp (other agents)",
        "- .continue/mcpServers/evidencespine.json - Continue auto-import",
        "- .junie/mcp/mcp.json - Junie",
        "- .roo/mcp.json - Roo Code",
        "",
        "Other clients: print a copy-paste snippet (never writes):",
        "",
        "    evidencespine harness print-config --harness <client>",
        "",
        "Supported: claude-code, claude-desktop, codex, cursor, dsh, gemini, opencode,",
        "vscode (legacy), vscode-copilot, windsurf (devin), zed, continue, antigravity,",
        "cline, roo, junie, warp, amp, generic (jetbrains is UI-only).",
        "",
        "Uninstall: evidencespine harness uninstall --harness generic",
        "Docs: README.md, docs/INTEGRATION.md, AGENTS.md",
        "",
    ]
    if len(lines) > MAX_NOTES_LINES:
        lines = lines[: MAX_NOTES_LINES - 1] + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------


def _normalize_projections(raw: Any) -> List[str]:
    if raw is None:
        return [str(rel) for rel in DEFAULT_PROJECTIONS]
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, (list, tuple)):
        items = [str(part).strip() for part in raw if str(part).strip()]
    else:
        return [str(rel) for rel in DEFAULT_PROJECTIONS]
    safe: List[str] = []
    for rel in items:
        normalized = os.path.normpath(rel)
        if os.path.isabs(normalized) or normalized.startswith(".."):
            continue
        safe.append(normalized)
    return safe


def _base_result(*, target: str, path: str) -> Dict[str, Any]:
    return {
        "status": "ok",
        "harness": HARNESS,
        "path": path,
        "target_dir": target,
        "wrote": [],
        "warnings": [],
        "projections": {},
    }


def _install_invalid(reason: str, *, path: str, target: str, warnings: List[str], dry_run: bool) -> Dict[str, Any]:
    result = _base_result(target=target, path=path)
    result.update(
        {
            "status": "invalid",
            "reason": reason,
            "warnings": [*warnings, reason],
            "dry_run": dry_run,
        }
    )
    return result


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


def install_generic(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    scope: str = "project",
    executable: str = "",
    dry_run: bool = False,
    **kw: Any,
) -> Dict[str, Any]:
    """Write the canonical MCP file plus merge-preserving project projections."""
    target = _expand_path(target_dir or ".")
    scope_name = "global" if str(scope or "project").strip().lower() in {"global", "user"} else "project"
    projections = _normalize_projections(kw.get("projections"))
    with_notes = bool(kw.get("notes", True))
    with_skill = bool(kw.get("with_skill", False))
    with_agents_md = bool(kw.get("with_agents_md", False))
    warnings: List[str] = []
    entry = build_server_entry(executable=executable, base_dir=base_dir)
    canonical_entry = dict(entry)
    canonical_entry["env"] = {}
    canonical = os.path.join(target, CANONICAL_FILENAME)

    existing_text, read_error = _read_existing(canonical)
    if read_error:
        return _install_invalid(
            "cannot read " + canonical + ": " + read_error,
            path=canonical,
            target=target,
            warnings=warnings,
            dry_run=dry_run,
        )
    canonical_payload = {"mcpServers": {SERVER_NAME: dict(canonical_entry)}}
    action = "added"
    if existing_text is not None:
        try:
            parsed = json.loads(existing_text)
        except json.JSONDecodeError as exc:
            return _install_invalid(
                canonical + " is not valid JSON (" + str(exc) + "); refusing to overwrite it",
                path=canonical,
                target=target,
                warnings=warnings,
                dry_run=dry_run,
            )
        if not isinstance(parsed, dict):
            return _install_invalid(
                canonical + " must contain a JSON object; refusing to overwrite it",
                path=canonical,
                target=target,
                warnings=warnings,
                dry_run=dry_run,
            )
        canonical_payload = merge_mcp_config(parsed, canonical_entry)
        action = "unchanged" if canonical_payload == parsed else "updated"

    planned: List[Tuple[str, str, Dict[str, Any]]] = [(CANONICAL_FILENAME, canonical, canonical_payload)]
    projection_actions: Dict[str, str] = {CANONICAL_FILENAME: action}
    absolute_planned = {canonical}
    for rel in projections:
        path = os.path.join(target, rel)
        if path in absolute_planned:
            continue
        if scope_name == "global" and not os.path.isdir(os.path.dirname(path)):
            warnings.append("skipped " + rel + ": parent directory does not exist (global scope)")
            projection_actions[rel] = "skipped"
            continue
        text, perr = _read_existing(path)
        if perr:
            warnings.append("skipped " + rel + ": " + perr)
            projection_actions[rel] = "skipped"
            continue
        refreshed = build_server_entry(executable=executable, base_dir=base_dir)
        if text is None:
            payload = {"mcpServers": {SERVER_NAME: refreshed}}
            p_action = "added"
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                warnings.append("skipped " + rel + ": existing file is not valid JSON (left untouched)")
                projection_actions[rel] = "invalid"
                continue
            if not isinstance(parsed, dict):
                warnings.append("skipped " + rel + ": existing JSON is not an object (left untouched)")
                projection_actions[rel] = "invalid"
                continue
            payload = merge_mcp_config(parsed, refreshed)
            p_action = "unchanged" if payload == parsed else "updated"
        absolute_planned.add(path)
        planned.append((rel, path, payload))
        projection_actions[rel] = p_action

    notes_path = os.path.join(target, NOTES_FILENAME)
    notes_text = render_setup_notes(base_dir=base_dir, executable=entry["command"]) if with_notes else ""

    result = _base_result(target=target, path=canonical)
    result["action"] = action
    result["warnings"] = warnings
    result["projections"] = projection_actions
    result["notes_path"] = notes_path

    if dry_run:
        result["dry_run"] = True
        result["content"] = _json_text(canonical_payload)
        result["planned"] = [path for _rel, path, _payload in planned if path != canonical]
        _maybe_install_instructions(
            result,
            target_dir=target,
            base_dir=base_dir,
            scope=scope,
            executable=executable,
            dry_run=dry_run,
            enabled=with_agents_md,
        )
        return result

    wrote: List[str] = []
    if action != "unchanged":
        _atomic_write_json(canonical, canonical_payload)
        wrote.append(canonical)
    for rel, path, payload in planned:
        if path == canonical or projection_actions.get(rel) in {"unchanged", "skipped", "invalid"}:
            continue
        _atomic_write_json(path, payload)
        if path not in wrote:
            wrote.append(path)
    if with_notes:
        previous_notes, _err = _read_existing(notes_path)
        if previous_notes != notes_text:
            _atomic_write_text(notes_path, notes_text)
            wrote.append(notes_path)
    result["wrote"] = wrote
    if with_skill:
        try:
            from evidencespine.harness import skills as skills_module

            skills_result = skills_module.materialize(
                target_dir=target,
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
        except Exception as exc:
            result["warnings"].append("skill materialization failed: " + str(exc))
    _maybe_install_instructions(
        result,
        target_dir=target,
        base_dir=base_dir,
        scope=scope,
        executable=executable,
        dry_run=dry_run,
        enabled=with_agents_md,
    )
    return result


def uninstall_generic(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    scope: str = "project",
    executable: str = "",
    dry_run: bool = False,
    **kw: Any,
) -> Dict[str, Any]:
    """Remove only the EvidenceSpine entry from the canonical file and projections."""
    target = _expand_path(target_dir or ".")
    scope_name = "global" if str(scope or "project").strip().lower() in {"global", "user"} else "project"
    projections = _normalize_projections(kw.get("projections"))
    with_notes = bool(kw.get("notes", True))
    warnings: List[str] = []
    canonical = os.path.join(target, CANONICAL_FILENAME)
    text, read_error = _read_existing(canonical)
    if read_error:
        return _install_invalid(
            "cannot read " + canonical + ": " + read_error,
            path=canonical,
            target=target,
            warnings=warnings,
            dry_run=dry_run,
        )
    if text is not None:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return _install_invalid(
                canonical + " is not valid JSON (" + str(exc) + "); refusing to touch it",
                path=canonical,
                target=target,
                warnings=warnings,
                dry_run=dry_run,
            )
        if not isinstance(parsed, dict):
            return _install_invalid(
                canonical + " must contain a JSON object; refusing to touch it",
                path=canonical,
                target=target,
                warnings=warnings,
                dry_run=dry_run,
            )
    result = _base_result(target=target, path=canonical)
    result["action"] = "absent"

    targets: List[Tuple[str, str]] = [(CANONICAL_FILENAME, canonical)]
    seen = {canonical}
    for rel in projections:
        path = os.path.join(target, rel)
        if path in seen:
            continue
        if scope_name == "global" and not os.path.isdir(os.path.dirname(path)):
            continue
        seen.add(path)
        targets.append((rel, path))

    planned_removals: List[str] = []
    for rel, path in targets:
        existing_text, perr = _read_existing(path)
        if perr:
            warnings.append("skipped " + rel + ": " + perr)
            continue
        if existing_text is None:
            result["projections"][rel] = "absent"
            continue
        try:
            parsed = json.loads(existing_text)
        except json.JSONDecodeError:
            warnings.append("skipped " + rel + ": existing file is not valid JSON (left untouched)")
            result["projections"][rel] = "invalid"
            continue
        if not isinstance(parsed, dict):
            warnings.append("skipped " + rel + ": existing JSON is not an object (left untouched)")
            result["projections"][rel] = "invalid"
            continue
        remaining = _without_server(parsed)
        if remaining == parsed:
            result["projections"][rel] = "absent"
            continue
        result["projections"][rel] = "removed" if not remaining else "removed_entry"
        planned_removals.append(path)

    notes_path = os.path.join(target, NOTES_FILENAME)
    if dry_run:
        result["dry_run"] = True
        would_remove = list(planned_removals)
        if with_notes and os.path.exists(notes_path):
            would_remove.append(notes_path)
        result["would_remove"] = would_remove
        return result

    removed: List[str] = []
    for path in planned_removals:
        text_now, _err = _read_existing(path)
        if text_now is None:
            continue
        parsed_now = json.loads(text_now)
        remaining_now = _without_server(parsed_now)
        if not remaining_now:
            try:
                os.unlink(path)
            except OSError:
                pass
        else:
            _atomic_write_json(path, remaining_now)
        removed.append(path)
    if with_notes and os.path.exists(notes_path):
        try:
            os.unlink(notes_path)
            removed.append(notes_path)
        except OSError:
            pass
    if removed:
        result["action"] = "removed"
        result["wrote"] = list(removed)
        result["removed"] = list(removed)
    return result
