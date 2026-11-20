# Security Policy

EvidenceSpine is a local-first memory side-car: it stores claim text, grounded
excerpts of your repository, and verification provenance on disk. Treat the
store (`.evidencespine/`) as sensitive data.

The full threat model, hardening measures, and deployment checklist live in
[docs/SECURITY.md](docs/SECURITY.md). This page is the reporting policy.

## Supported versions

| Version | Supported |
|---|---|
| 0.6.x | Yes |
| Earlier versions | No (upgrade before reporting) |

## Reporting a vulnerability

Please **do not open a public issue** for a security-sensitive report.

1. Use GitHub's private vulnerability reporting for this repository:
   <https://github.com/rivastava/EvidenceSpine/security/advisories/new>
2. Include the affected version, a minimal reproduction, the impact you
   observed, and any suggested fix.
3. You will receive an acknowledgement, and we will coordinate a fix and a
   disclosure date with you. Please allow reasonable time before public
   disclosure.

If GitHub private reporting is unavailable, contact the maintainer through
the repository profile and mark the message clearly as a security report.

## Scope

In scope:

- the evidence store (SQLite/JSONL), redaction and grounding confinement;
- the MCP server (stdio and streamable HTTP) and its auth/bind controls;
- the A2A server and agent card;
- the harness installers (Claude Code, Codex, opencode, DSH, generic MCP,
  Cursor, VS Code, git hooks, instructions handlers);
- the CLI and adapter surfaces.

Out of scope:

- vulnerabilities in third-party MCP clients or model providers;
- deployments that deliberately disable the documented safeguards
  (`EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED`,
  `EVIDENCESPINE_A2A_ALLOW_UNAUTHENTICATED`) or expose unauthenticated
  network binds;
- secrets that users deliberately place in grounded excerpts or verification
  references (see the redaction trade-off in [docs/SECURITY.md](docs/SECURITY.md)).

## Good practice when reporting

Do not include real credentials or private source code beyond what is needed
to reproduce the issue. Prefer synthetic repository content in proofs of
concept.
