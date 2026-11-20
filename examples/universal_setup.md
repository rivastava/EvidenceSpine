# Universal MCP setup

The primary setup guide is the README section
["Universal setup"](../README.md#universal-setup-any-mcp-client). This page is a
short recipe for wiring EvidenceSpine into any MCP-capable client.

## 1. Run the server

```bash
# stdio (what config files below launch)
evidencespine mcp --base-dir /absolute/path/to/repo/.evidencespine

# HTTP variant (bearer auth required for non-loopback; loopback default)
export EVIDENCESPINE_MCP_AUTH_TOKEN=change-me
evidencespine mcp --transport streamable-http --host 127.0.0.1 --port 8000 \
  --base-dir /absolute/path/to/repo/.evidencespine
```

Use absolute `--base-dir` values in client configs: MCP hosts do not spawn the
server from your repository root.

## 2. Print (do not hand-write) a client config

```bash
evidencespine harness print-config --harness claude-code --json
evidencespine harness print-config --harness codex --json
evidencespine harness print-config --harness cursor --json
evidencespine harness print-config --harness vscode-copilot --json
evidencespine harness print-config --harness opencode --json
evidencespine harness print-config --harness dsh --json
evidencespine harness print-config --harness generic --json   # canonical mcpServers object
evidencespine harness print-config --harness all --json       # every surveyed client
```

Each payload includes the target path, the snippet, a verification command, and
a confidence grade; add `--transport http` for the streamable-HTTP variant.

## 3. Or install project files non-destructively

```bash
evidencespine harness install --harness generic --target-dir .   # writes .mcp.json + projections
evidencespine harness install --harness dsh                     # $DSH_HOME/cordis.patch.yml row
evidencespine harness install --harness all --target-dir .      # every supported harness
evidencespine harness install --harness generic --dry-run       # show planned writes
```

## 4. Verify

```bash
evidencespine harness doctor --json
evidencespine harness debug --json
evidencespine snapshot --json
```

The full matrix (surfaces, config paths, transports, auth support, uncertainty
grades, print-only clients, and the legacy VS Code migration) is in
[`docs/HARNESS_MATRIX.md`](../docs/HARNESS_MATRIX.md). DSH-specific details are in
[`docs/DSH.md`](../docs/DSH.md).
