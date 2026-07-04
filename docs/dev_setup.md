# Developer Setup

This guide covers local development dependencies for the CC Framework. The core
project is Python-first; Node.js tooling is only required for the dashboard and
infrastructure reference lanes.

## Python environment (required)

1. Install Python 3.10+.
2. Create a virtual environment and install dependencies:

   ```bash
   make install
   ```

3. Run a smoke test:

   ```bash
   make reproduce-smoke
   ```

## Node.js tooling (optional)

Node.js is only needed if you plan to work on `apps/dashboard/` or `infra/`.
It is **not** required for running the core Python experiments.

1. Install Node.js (LTS) and npm.
2. Install dashboard dependencies when working on the dashboard:

   ```bash
   cd apps/dashboard
   npm ci
   ```

The historical root-level Node package has been removed. `apps/dashboard/` and
`infra/` are the active JavaScript package roots. Older static artifacts such as
`tools/week6_artifact.html` load their chart libraries from CDNs and do not
require a repo-root npm install.

## Validation lanes

Use [Validation Matrix](validation_matrix.md) to choose the narrowest command
lane for the claim you want to check. Local package support starts at Python
3.10; GitHub Actions currently validates code and docs on Python 3.10, 3.11,
3.12, and 3.13.

Optional enterprise, dashboard, vendor, serialization, experiment, and
performance dependencies may be skipped by the full local test suite when their
lane-specific extras or environment gates are not installed. A skip is honest
only when the corresponding optional lane is not being claimed as passed.

For Enterprise Reference validation, run:

```bash
make enterprise-smoke
```

That target installs/checks the enterprise Python extras, installs dashboard
package dependencies, and runs the moto-backed AWS emulation plus dashboard e2e
smoke as a single pass/fail lane.
