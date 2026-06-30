# Developer Setup

This guide covers local development dependencies for the CC Framework. The core
project is Python-first; Node.js tooling is only required for optional UI/chart
work.

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

Node.js is only needed if you plan to work on Recharts-based UI/chart artifacts
(for example, iterating on local chart prototypes or UI visualizations). It is
**not** required for running the core Python experiments.

1. Install Node.js (LTS) and npm.
2. Install JavaScript dependencies:

   ```bash
   npm install
   ```

This installs the `recharts` dependency defined in `package.json`, enabling local
development of chart/UI assets.

## Validation lanes

Use [Validation Matrix](validation_matrix.md) to choose the narrowest command
lane for the claim you want to check. Local package support starts at Python
3.10; GitHub Actions currently validates code and docs on Python 3.10, 3.11,
3.12, and 3.13.

Optional enterprise, dashboard, vendor, serialization, experiment, and
performance dependencies may be skipped by the full local test suite when their
lane-specific extras or environment gates are not installed. A skip is honest
only when the corresponding optional lane is not being claimed as passed.
