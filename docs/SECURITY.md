# Security

EvidenceSpine is designed as a **local-first side-car**: store on disk, agent
sessions on the same host, and MCP transport defaulting to `stdio` or
`127.0.0.1`. This document states the threat model and the explicit trade-offs
in its persistence and server layers.

## Threat model

The store is intended for **trusted, local or controlled multi-agent
environments**. Treat the on-disk store (`.evidencespine/`) as sensitive data:
it may contain grounded excerpts of your repository, claim text, and
verification provenance. Do not place it in a public repository.

If you expose the MCP server over a network, that server can be reached by any
client that can authenticate — or, without a token, by anyone who can reach the
port. Never bind to `0.0.0.0` on an untrusted network without
`EVIDENCESPINE_MCP_AUTH_TOKEN`.

## Redaction trade-off

On persistence, EvidenceSpine redacts string values that look like keys,
tokens, secrets, long hex blobs, or long digit runs. Redaction is **intentionally
skipped** for the following fields so that evidence checksums, provenance, and
identity stay intact:

- `excerpt`, `reference`, `checksum`, `source_id`, `commit`
- `event_hash`, `event_id`, `fact_id`, `packet_id`
- `thread_id`, `role`, `scope_id`, `owner_agent_id`, `state`, `status`
- timestamps (`validated_at`, `fresh_until`, `written_at`, ...)

Consequence: a secret that appears verbatim inside a grounded `excerpt` or a
verification `reference` is persisted as-is. This preserves checksum
integrity and drift-checkability, at the cost of not scrubbing secrets from
evidence content. When ingesting untrusted content, do not attach excerpts or
references that may contain secrets.

## Grounding confinement

The `ground` MCP tool and `ground_refs` on `ingest_event` resolve file paths
**relative to a server-configured source root** (env
`EVIDENCESPINE_SOURCE_ROOT`, defaulting to the server working directory) and
**reject absolute paths and `../` escapes**. A remote caller therefore cannot
use grounding to read arbitrary host files. The local CLI (`evidencespine
ground`, `ingest --ground-ref --source-root`) is a trusted shell surface and
may still ground absolute paths.

## HTTP authentication

- Transport default: `stdio` (local), or `streamable-http` on `127.0.0.1`.
- Set `EVIDENCESPINE_MCP_AUTH_TOKEN=<token>` to require
  `Authorization: Bearer <token>` on every streamable-http request.
- If the token is **not** set, the server prints a warning at startup and
  serves unauthenticated. Keep the default loopback binding; put it behind a
  reverse proxy with TLS for remote use.

## Artifact safety

Brief and handoff artifacts are written with **random UUID filenames** and
**atomic exclusive creation** (`O_CREAT | O_EXCL`), so caller-controlled
`thread_id`/`role` values (which live in the JSON payload only) can never
traverse out of the configured `briefs/` or `handoffs/` directories, and two
writes in the same second cannot silently overwrite each other.

## Deduplication

Event dedupe honors `EVIDENCESPINE_DEDUPE_WINDOW_SEC` (default 2h): the
in-memory hash ring and the SQLite `dedup_hashes` table both expire entries
older than the window, and pruning removes stale hash rows. An identical event
re-issued after the window is accepted again.

## Verification

- Runs the full test suite (`python -m pytest tests/`).
- CI enforces `ruff` and `mypy` (see `.github/workflows/ci.yml`).
- To report a security issue, open a private GitHub advisory or an issue on
  the repository; do not include real secrets in excerpts/references.

## Deployment checklist for remote MCP use

- [ ] Set `EVIDENCESPINE_MCP_AUTH_TOKEN` to a strong, random value
- [ ] Bind to a non-public interface only behind a TLS reverse proxy
- [ ] Set `EVIDENCESPINE_SOURCE_ROOT` to the exact directory grounding may read
- [ ] Do not commit `.evidencespine/`
- [ ] Review the `excerpt`/`reference` trade-off above before ingesting
      untrusted content
