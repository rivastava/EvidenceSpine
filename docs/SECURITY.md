# Security

EvidenceSpine is a **local-first side-car**: the store lives on disk, agent
sessions usually run on the same host, and MCP defaults to `stdio` (or
streamable-HTTP bound to loopback). This document states the threat model, the
verified semantics behind the words "verified" and "grounded", and the explicit
trade-offs in the persistence and server layers.

## Threat model

- Trusted, local or controlled multi-agent environments are the target.
- Treat `.evidencespine/` as sensitive: it contains claim text, grounded
  excerpts of your repository, verification references, and state metadata. Do
  not commit it to a public repository.
- A network-reachable MCP or A2A server can be reached by anyone who can reach
  the port (authenticated or not). Never bind to `0.0.0.0` on an untrusted
  network.
- A malicious MCP/A2A caller is untrusted input: it must not be able to read
  arbitrary host files through grounding/drift/import, and it must not be able
  to silently wipe memory.

## Verified semantics (read this before trusting a brief)

The word "verified" is scoped, and the scope matters:

1. **At ingest**, `fact_state="verified"` means the stored evidence is
   *self-consistent*: the item has a checksummed excerpt whose stored checksum
   equals sha256(excerpt), or verification provenance (method + reference).
   The policy is enforced for both event-derived facts and raw
   `append_fact` rows (`EVIDENCESPINE_VERIFIED_REQUIRES_SPAN`, default on);
   ungrounded "verified" claims are downgraded to `asserted` with
   `metadata.policy = "verified_requires_span"` and counted in the result's
   `policy_downgrades`.
2. **Self-consistency is not truth.** A stored checksum matching its excerpt
   proves the excerpt was not altered *after* grounding; it does not prove the
   excerpt is accurate, current, or complete. The repo is ground truth.
3. **Drift-check proves the live source.** Only
   `check_drift` / `evidencespine drift-check` re-reads the current file at
   the stored anchor, recomputes the checksum, and reports
   `changed` / `missing` / `unverifiable`. That is what establishes
   whether the cited evidence still matches. A fact can be "verified" at
   ingest and stale a minute later after an edit.
4. **Stale evidence is never emitted as verified.** Facts whose evidence
   drifted are surfaced as `STALE EVIDENCE` risks, included in the
   `stale_claims` view, and exported in handoffs as `asserted` rows with
   `evidence_stale: true`. On import, a stale row is downgraded to
   `asserted` even if the packet says verified.
5. **`unverifiable` fails closed.** If an item cannot be re-checked
   (unsupported/ambiguous anchor, out-of-range character span, or a source id
   outside the allowed root), drift does not clear an existing stale flag and
   does not mark the fact verified.

## Grounding confinement

Grounding reads files, so it is the highest-risk surface. The split is:

- **Network surface (MCP tools / A2A) is confined.** The server resolves a
  grounding root (explicit arg -> `EVIDENCESPINE_SOURCE_ROOT` -> server cwd,
  realpathed). `ground`, `ingest_event ground_refs` and the server-side
  drift check reject absolute paths and refused escapes (including
  `..` and symlinks) — a remote caller cannot use grounding as a file-read
  oracle. `check_drift` ignores any caller-supplied `source_root` and runs
  with `allow_absolute_source_id=False`, so absolute source ids and ids
  escaping the root are reported `unverifiable` rather than read. On a
  runtime without that confinement mode the tool returns
  `{"status":"unsupported","reason":"runtime_missing_evidence_confinement"}`
  instead of silently allowing the old behavior.
- **Handoff imports are confined too.** `import_handoff` accepts a packet
  object or a **relative** path under the server grounding root; absolute
  paths and escaping paths return `path_outside_source_root` / rejected.
- **The CLI is a trusted shell surface.** `evidencespine ground` and
  `ingest --ground-ref --source-root` run as your user and may ground
  absolute paths; that is intentional (useful in scripts), and the reason
  remote callers must never get that mode.
- Redaction runs **before** persistence and hashing; grounded `excerpt` /
  `reference` values are explicitly not redacted (see below), so **never
  ground a secret**.

## MCP authentication

- Default transport is `stdio`: the client spawns the server as a child
  process; access is the OS user's access. No token needed.
- `--transport streamable-http` serves HTTP. Set
  `EVIDENCESPINE_MCP_AUTH_TOKEN` to require
  `Authorization: Bearer <token>` on **every** request. Comparison is
  constant-time (`hmac.compare_digest`); a missing/wrong token gets a
  `401` with `WWW-Authenticate: Bearer`.
- **Non-loopback binds are refused** without a token. Starting
  `--transport streamable-http --host 0.0.0.0` with no token raises instead
  of serving. The explicit dev escape hatch is
  `EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED=1` (prints a loud warning; anyone
  who can reach the port can read, write and prune memory).
- There is **no OAuth authorization server** in EvidenceSpine and no TLS
  termination. The token is a static shared secret; TLS must be terminated at
  a reverse proxy / gateway, and OAuth-only clients must be fronted by a
  gateway that translates to the static bearer.
- Destructive tools are conservative where it matters: both MCP and A2A
  `prune` **default to `dry_run=true`** (preview counts; pass `dry_run=false`
  to delete). `prune` and `reconcile` are available over any authenticated
  transport, so a shared token is a write credential.

## A2A authentication

- Discovery document: `GET /.well-known/agent-card.json` is **public**
  (discovery must work before credentials are obtained). Every other path
  requires `Authorization: Bearer <token>` when a token is configured.
- Token resolution: explicit `auth_token` argument, then
  `EvidenceSpineSettings.a2a_auth_token`, then
  `EVIDENCESPINE_A2A_AUTH_TOKEN`. Comparison is constant-time.
- **Non-loopback binds are refused** without a token; the escape hatch is
  `EVIDENCESPINE_A2A_ALLOW_UNAUTHENTICATED=1`. Loopback without a token
  prints a warning but serves (local development).
- The server is a bearer-guarded A2A endpoint, not an OAuth issuer: it does not
  mint tokens, run authorization-code flows, or declare security schemes for
  you. The shipped agent card advertises memory skills; if you publish the
  endpoint beyond a trusted network, front it with TLS and an OAuth-capable
  gateway that injects the bearer, or keep it on loopback behind an SSH
  tunnel.
- A2A exposes a **subset** of operations: `ingest_event, append_fact,
  build_brief, query_view, emit_handoff, import_handoff, reconcile, snapshot,
  prune`. There is no `ground`/`check_drift`/`verify_fact` over A2A.

## Redaction trade-off

On persistence, string values that look like keys/tokens/secrets, long hex
blobs, or long digit runs are redacted. Redaction is intentionally **skipped**
for identity/integrity fields: `excerpt`, `reference`, `checksum`,
`source_id`, `commit`, `event_hash`, `event_id`, `fact_id`,
`packet_id`, `thread_id`, `role`, `scope_id`, `owner_agent_id`,
`state`, `status` and timestamps.

Consequences:

- A secret that appears verbatim inside a grounded `excerpt` or a
  verification `reference` is persisted as-is. This preserves checksum
  integrity and drift-checkability at the cost of not scrubbing evidence
  content. Do not ground untrusted content that may contain secrets.
- Because redaction now runs **before** validation/hashing, the stored
  `event_hash` (used for dedupe) is computed over the redacted row. A claim
  that is redacted therefore has a different hash than the same claim would
  unredacted — expected, but it means dedupe keys changed relative to
  pre-0.6 stores.
- Env parsing is fail-safe: an unrecognized boolean value
  (`EVIDENCESPINE_FAIL_OPEN=tru`) keeps the default (fail-open stays on)
  instead of silently flipping to fail-closed, and an invalid enable/format
  value keeps the default rather than silently disabling memory.

## Artifact safety

Brief and handoff artifacts are written with **random UUID filenames** and
atomic exclusive creation, so caller-controlled `thread_id`/`role` values
(living only in the JSON payload) can never traverse out of `briefs/` or
`handoffs/`, and concurrent writes cannot silently overwrite each other.
Handoff packet JSON is checksummed (`checksum`) for integrity and the
importer validates schema/role/claims before ingesting.

## Deduplication

Event dedupe honors `EVIDENCESPINE_DEDUPE_WINDOW_SEC` (default 2 h). In
SQLite the `dedup_hashes` table is authoritative (WAL, indexed); JSONL keeps
an in-memory ring. Entries older than the window are expired, and the same
event re-issued after the window is accepted again. Dedupe is a correctness
convenience, not a security boundary: a caller with write access can always
ingest new distinct events.

## Deployment checklist

- [ ] Keep `.evidencespine/` out of version control (add it to
      `.gitignore`); treat store files as sensitive.
- [ ] Prefer `stdio` MCP. For HTTP: set a strong random
      `EVIDENCESPINE_MCP_AUTH_TOKEN`, bind `127.0.0.1`, and terminate TLS
      at a reverse proxy. Never set `EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED`
      on a shared network.
- [ ] For A2A: set `EVIDENCESPINE_A2A_AUTH_TOKEN` and bind loopback. Publish
      only the agent card publicly; front remote A2A with TLS/OAuth at the
      gateway.
- [ ] Set `EVIDENCESPINE_SOURCE_ROOT` to the exact directory grounding may
      read; do not make it `/` or your home directory.
- [ ] Rotate tokens like any shared credential; restart the server after
      changing them (a long-lived process keeps its start-time environment).
- [ ] Use the `prune` previews (MCP and A2A default to `dry_run=true`),
      then pass `dry_run=false` deliberately when you really mean to delete.
- [ ] Review the excerpt/reference redaction trade-off before ingesting
      untrusted content; never ground secrets.
- [ ] Run `evidencespine harness doctor --json` after installs and after
      upgrades; compare `snapshot.evidencespine_version` with the installed
      version and restart stale servers.

## Reporting

Report vulnerabilities privately (GitHub security advisory or a direct
maintainer contact); do not include real secrets in excerpts or references.
