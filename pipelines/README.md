# Pipeline Boundary

This directory is reserved for ingestion, sanitation, deduplication, review,
promotion, and observability pipeline code.

Pipelines must fail closed. Missing provenance, integrity mismatches, schema
violations, malware signals, overclaims, and red-team-to-confirmatory leakage
route to quarantine or review according to the Phase 1 architecture spec.
