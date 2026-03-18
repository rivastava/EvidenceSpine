# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.4.0] - 2026-03-18

### Added
- Added agent-state `state_context` support across events, facts, brief citations, handoff rows, and transcript normalization.
- Added derived control views for `active_scopes`, `my_work`, `open_gates`, `stale_claims`, and `contradictions`.
- Added a generic `reconcile_state` runtime hook plus `AgentMemoryRuntime.reconcile(...)`.
- Added CLI support for structured control-state ingest and new `view` / `reconcile` commands.
- Added runtime, protocol, CLI, and adapter tests for freshness, ownership, conflicts, and active-scope derivation.

### Changed
- Extended Protocol v2 without changing `schema_version`, storage layout, or the dependency-free runtime.
- Propagated control-state metadata through brief generation and handoff import/export.
- Added snapshot metrics for active-scope count, open gates, owner coverage, freshness coverage, stale rate, and conflict rate.
- Updated docs and examples to position EvidenceSpine explicitly as a side-car rather than the final runtime authority.

### Notes
- Existing refs-only and free-form callers remain supported.
- No storage migration or framework-specific truth adapter was introduced.

## [0.3.0] - 2026-03-07

### Added
- Added Protocol v2 structured evidence spans via `evidence_items` and public helper types `EvidenceItem` and `ClaimCitation`.
- Added CLI ingest support for structured evidence via `--evidence-item-json` and `--evidence-item-file`.
- Added brief and handoff span-grounding metrics, including ref coverage, span coverage, excerpt fidelity, and handoff span grounding.
- Added protocol, runtime, CLI, adapter, and benchmark result-shape tests for the new evidence span flow.

### Changed
- Upgraded brief `citations` output to structured citation bundles while preserving `citation_refs` as a legacy ref-list alias.
- Propagated structured evidence through runtime ingest, fact derivation, brief assembly, handoff emission, handoff import, and transcript adapter ingestion.
- Updated benchmark harnesses to generate synthetic grounded evidence spans and report grounding-quality metrics.
- Updated docs and examples to describe Protocol v2 claim-to-span grounding.

### Notes
- The package remains dependency-free at runtime.
- Existing refs-only callers remain supported; `evidence_refs` is still accepted and emitted alongside `evidence_items`.

## [0.2.0] - 2026-03-04

### Added
- Added `TranscriptAdapter` as the new generic transcript-first integration path for plain `messages[]` and transcript-like payloads.
- Added public adapter contract types: `NormalizedTranscriptMessage`, `TranscriptAdapterConfig`, and `AdapterIngestResult`.
- Added replay fixtures and replay harness examples for realistic brief and handoff validation.
- Added adapter-focused documentation in `docs/ADAPTERS.md`.

### Changed
- Refactored `LangGraphAdapter` and `AutoGenAdapter` to delegate to the shared transcript adapter while preserving backward compatibility.
- Updated integration and architecture docs to recommend transcript-first integration as the default path.
- Expanded adapter coverage with normalization, replay, and compatibility tests.
- Bumped package version to `0.2.0`.

### Notes
- The package remains dependency-free.
- No protocol, storage, or CLI breaking changes were introduced in this release.
