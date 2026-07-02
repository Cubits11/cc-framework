# Contributing to CC-Framework

Thanks for helping make CC-Framework easier to inspect, reproduce, and use.
This repository is research software, so contributions should keep claims
narrow: the package bounds what available evidence supports under stated
assumptions; it does not certify that a stacked system is safe.

## Development setup

Use Python 3.10 or newer. Python 3.12 is the primary local development target
in the current Makefile.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Before opening a pull request, run the release-facing checks:

```bash
make test-kernel
make test-release
```

As verified on 2026-07-01, `make test-kernel` runs the kernel unit tests,
strict mypy on `src/cc/kernel`, and ruff checks for the kernel surface.
`make test-release` runs that same gate plus the minimal bounds example and
release artifact integration tests.

## Coding standards

- Prefer the existing public kernel surface in `cc.kernel.strict` for
  documentation and examples.
- Keep statistical claims tied to assumptions, evidence, and tests that are
  checked in this repository.
- Add or update tests for changes to bound computation, witness construction,
  metrics, serialization, or adapter behavior.
- Keep generated artifacts out of commits unless the file is explicitly part of
  a documented release, paper, or launch asset.
- Do not introduce public language that implies deployment approval.

## Pull request process

1. Keep the PR focused on one reviewable change.
2. Include the commands you ran and their actual outcome.
3. Link any related issue or design note.
4. Call out compatibility risks, especially public API, schema, or artifact
   format changes.
5. Wait for maintainer review before merge.

## Adding a guardrail adapter

Adapters should live under `src/cc/adapters/` unless an established module gives
a more specific home. A new adapter should include:

- a small, typed interface that records per-item pass/fail outcomes;
- tests covering success, failure, malformed input, and empty input;
- a README or example if the adapter depends on a third-party tool;
- clear mapping from the external tool's result model to the repository
  convention that `Z_i = 1` means guardrail failure / unsafe pass.

Do not choose adapter names that collide with the existing `Claim` envelope
schema or other established module namespaces. If a proposed name is close to a
core schema, open a design issue before implementing it.
