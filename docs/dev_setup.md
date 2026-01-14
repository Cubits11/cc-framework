# Developer Setup

This guide covers local development dependencies for the CC Framework. The core
project is Python-first; Node.js tooling is only required for optional UI/chart
work.

## Python environment (required)

1. Install Python 3.9+.
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
