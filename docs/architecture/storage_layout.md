# Storage Layout

This document defines the deterministic artifact layout for cc-framework runs.

## Root folders

```
runs/<shard>/<hash>/manifest.json
results/<shard>/<hash>/results.json
figures/<shard>/<hash>/<artifact_name>
```

* `<hash>` is the SHA-256 hash of the artifact content.
* `<shard>` is the first two hex characters of `<hash>` for filesystem fan-out.

## Manifest contents

Each run writes a manifest with provenance data:

* `config_sha256`: SHA-256 of the stable JSON configuration.
* `dataset_sha256`: SHA-256 derived from the dataset paths referenced by config.
* `git_sha`: git commit SHA of the repository at run time.
* `started_at_unix` / `completed_at_unix`: timestamps for the run lifecycle.
* `artifacts`: paths to analysis JSON, audit logs, and optional CSVs.

This layout allows downstream tooling to map inputs to deterministic outputs without
relying on wall-clock timestamps.
