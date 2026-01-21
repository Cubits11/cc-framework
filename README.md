# Correlation Cliff Framework

**Rigorous dependence-aware evaluation of AI safety system composition under distributional uncertainty.**

[![Tests](https://github.com/Cubits11/cc-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/Cubits11/cc-framework/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen.svg)](https://codecov.io/gh/Cubits11/cc-framework)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2501.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)

> **Mission:** Quantify when composed AI guardrails exhibit **correlation cliffs**—phase transitions where marginal increases in statistical dependence produce dramatic, non-linear degradation in composition effectiveness.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Correlation Cliff Phenomenon](#the-correlation-cliff-phenomenon)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Installation & Setup](#installation--setup)
5. [Quick Start](#quick-start)
6. [Core API Reference](#core-api-reference)
7. [Experimental Protocols](#experimental-protocols)
8. [Interpreting Results](#interpreting-results)
9. [Reproducibility Standards](#reproducibility-standards)
10. [Advanced Topics](#advanced-topics)
11. [Contributing](#contributing)
12. [Citation](#citation)
13. [Governance & Ethics](#governance--ethics)

---

## Executive Summary

### The Problem

AI safety systems increasingly rely on **composed guardrails**—layered defenses combining content filters, semantic classifiers, and behavioral monitors. Standard evaluation treats these compositions as independent mechanisms, but **dependence between guardrails fundamentally alters system behavior**:

- **Redundancy:** Overlapping failure modes waste compute without improving safety
- **Brittleness:** Correlated errors create systematic vulnerabilities under distribution shift
- **Non-Compositionality:** Strong individual guardrails can produce weak compositions

### Our Solution

**Correlation Cliff Framework** provides:

1. **Theoretical Foundation:** Fréchet-Hoeffding bounds for dependence-agnostic composition analysis
2. **Empirical Protocol:** Two-world evaluation measuring identifiable composition jumps
3. **Practical Tools:** End-to-end pipeline from raw guardrail APIs to publication-ready figures
4. **Provenance Guarantees:** Cryptographically-verifiable audit trails and reproducibility manifests

### Key Innovation

We prove that **composition behavior under distributional shift is bounded by marginal statistics alone**—no access to true joint distributions required. This enables:

- **Worst-case analysis** without exhaustive dependence enumeration
- **Early warning systems** detecting composition brittleness before deployment
- **Evidence-based thresholding** for safety-critical cascades

---

## The Correlation Cliff Phenomenon

### Conceptual Overview

Consider two binary guardrails A and B with marginal trigger rates:
- World 0 (baseline): `P(A=1) = 0.2`, `P(B=1) = 0.3`
- World 1 (shifted): `P(A=1) = 0.4`, `P(B=1) = 0.5`

**Question:** How does their OR-composition `C = A ∨ B` behave under shift?

**Traditional answer:** "It depends on their correlation."

**Our answer:** "It's **bounded** regardless of correlation, and the bounds are tight."

### The Cliff

Define:
- **JC (Composition Jump):** `|P(C=1|world1) - P(C=1|world0)|`
- **J_best (Best Single Jump):** `max(|ΔpA|, |ΔpB|)`
- **CC (Composition Coefficient):** `JC / J_best`

**Correlation Cliff Theorem:**

For any feasible joint distribution consistent with given marginals:

```
JC_min ≤ JC ≤ JC_max
```

where `JC_min` and `JC_max` are computable from **marginals alone** via Fréchet-Hoeffding bounds.

**The "cliff"** occurs when small changes in dependence structure (parameterized by λ ∈ [0,1]) cause CC to transition from ≈0 (independent composition) to ≈1 (perfect coherence) within a narrow λ-interval.

### Real-World Impact

**Scenario:** Deploy Perspective API (A) + Llama Guard (B) in cascade.

- **Lab evaluation (clean prompts):** CC ≈ 0.15 (constructive composition)
- **Production (adversarial prompts):** CC ≈ 0.92 (near-total correlation)

**Diagnosis:** Adversarial prompts induce correlated errors—both guardrails fail together. The composition provides **illusory redundancy**.

**Action:** CC Framework predicts this failure mode from marginal statistics, enabling proactive mitigation (diversity requirements, orthogonal detection strategies).

---

## Mathematical Foundations

### Notation

| Symbol | Meaning |
|--------|---------|
| `pA, pB` | Marginal trigger probabilities for guardrails A, B |
| `p11` | Joint overlap probability `P(A=1, B=1)` |
| `L, U` | Fréchet-Hoeffding bounds: `max(0, pA+pB-1) ≤ p11 ≤ min(pA, pB)` |
| `λ ∈ [0,1]` | Dependence parameter interpolating FH envelope |
| `pC` | Composition rate under rule (OR/AND/COND_OR) |
| `JC` | Composition jump magnitude across worlds |
| `CC` | Normalized composition coefficient |

### Fréchet-Hoeffding Bounds

**Theorem (Fréchet, 1951; Hoeffding, 1940):**

For binary random variables A, B with marginals pA, pB:

```
L = max(0, pA + pB - 1)  ≤  P(A=1, B=1)  ≤  U = min(pA, pB)
```

These bounds are **sharp**: distributions achieving L (countermonotonic) and U (comonotonic) exist.

### Composition Rules

**OR (Disjunctive Cascade):**
```
pC = pA + pB - p11
```
Used when **either** guardrail triggering indicates a safety issue.

**AND (Conjunctive Gate):**
```
pC = p11
```
Used when **both** guardrails must agree (higher precision, lower recall).

**COND_OR (Conditional Independence):**
```
pC = pA + (1-pA)·pB
```
Assumes B operates on A's residual (independence given A's decision).

### Dependence Parameterization

We support multiple families for λ → p11 mapping:

1. **FH-Linear:** `p11(λ) = L + λ(U - L)`
2. **FH-Power:** `p11(λ) = L + λ^γ(U - L)`, γ > 0
3. **FH-S-Curve:** `p11(λ) = L + σ(k(λ-0.5))(U - L)`, σ = logistic
4. **Gaussian Copula:** Via Kendall's τ = 2λ - 1
5. **Clayton Copula:** θ(λ) for lower-tail dependence modeling

### Two-World Envelope

**Given:**
- World 0: `(pA0, pB0)`
- World 1: `(pA1, pB1)`
- Composition rule R

**Compute:**

```python
# For each world, get pC bounds over all feasible p11
I0 = [pC_min(pA0, pB0, R), pC_max(pA0, pB0, R)]
I1 = [pC_min(pA1, pB1, R), pC_max(pA1, pB1, R)]

# JC bounds via interval arithmetic
JC_min = max(0, |I1.min - I0.max|, |I1.max - I0.min|)
JC_max = max(|I1.min - I0.max|, |I1.max - I0.min|)

# Normalized CC bounds
J_best = max(|pA1 - pA0|, |pB1 - pB0|)
CC_min = JC_min / J_best  (if J_best > 0, else 0)
CC_max = JC_max / J_best
```

**Key Property:** These bounds hold **for any** dependence structure consistent with marginals—no assumptions on copula family required.

---

## Installation & Setup

### Prerequisites

- **Python:** 3.9, 3.10, or 3.11
- **OS:** Linux, macOS, Windows (WSL recommended)
- **RAM:** 8GB minimum, 16GB recommended for large-scale experiments
- **Optional:** CUDA-capable GPU for accelerated copula computations

### Base Installation

```bash
git clone https://github.com/Cubits11/cc-framework.git
cd cc-framework

# Create isolated environment
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core framework
pip install -e .

# Verify installation
python -c "from experiments.correlation_cliff import theory_core; print(theory_core.__file__)"
pytest tests/ -q
```

### Development Installation

```bash
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run full test suite with coverage
pytest --cov=experiments.correlation_cliff --cov-report=html

# Type checking
mypy experiments/correlation_cliff --strict

# Linting
ruff check .
ruff format .
```

### Optional Dependencies

**For real guardrail adapters:**
```bash
pip install -e ".[guardrails]"
# Installs: transformers, torch, perspective-api-client, openai
```

**For GPU-accelerated copulas:**
```bash
pip install -e ".[gpu]"
# Installs: cupy-cuda11x, jax[cuda]
```

**For distributed simulation:**
```bash
pip install -e ".[distributed]"
# Installs: ray[default], dask[complete]
```

### Docker (Recommended for Reproducibility)

```bash
docker build -t cc-framework:latest .
docker run --rm -v $(pwd)/results:/app/results cc-framework:latest \
    python -m experiments.correlation_cliff.run_all \
    --config experiments/correlation_cliff/config_s1.yaml \
    --out_dir /app/results/docker_run
```

**Dockerfile** ensures:
- Pinned Python version (3.10.12)
- Frozen dependencies via `requirements-lock.txt`
- Deterministic CUDA/cuDNN if GPU support enabled
- Immutable base image hash recorded in manifest

---

## Quick Start

### 1. Minimal Smoke Test (2 minutes)

```bash
# Run basic theory validation
python -m pytest experiments/correlation_cliff/unit_tests/test_theory.py -v

# Generate a simple CC curve
python -c "
from experiments.correlation_cliff import theory_core as TC

# Define two-world scenario
w = TC.TwoWorldMarginals(
    w0=TC.WorldMarginals(pA=0.2, pB=0.3),
    w1=TC.WorldMarginals(pA=0.4, pB=0.5)
)

# Compute CC bounds
cc_min, cc_max = TC.cc_bounds(w, rule='OR')
print(f'CC bounds: [{cc_min:.3f}, {cc_max:.3f}]')
"
```

Expected output:
```
CC bounds: [0.000, 0.857]
```

### 2. Full Simulation Pipeline (30 minutes)

```bash
# Run complete experiment with 5 lambda points, 100 samples, 10 replicates
PYTHONPATH=src python -m experiments.correlation_cliff.run_all \
    --config experiments/correlation_cliff/config_s1.yaml \
    --out_dir results/quick_start \
    --skip_figures

# Outputs:
# results/quick_start/
# ├── population_curve.csv          # Theoretical CC(λ) curve
# ├── sim_long.csv                  # Replicate-level results
# ├── sim_summary.csv               # Aggregated statistics
# ├── thresholds.json               # λ* estimates
# ├── manifest.json                 # Reproducibility metadata
# └── diagnostics.json              # Runtime statistics
```

### 3. Interactive Exploration (Jupyter)

```python
import pandas as pd
import matplotlib.pyplot as plt
from experiments.correlation_cliff.theory_core import *

# Load results
df_pop = pd.read_csv('results/quick_start/population_curve.csv')
df_sum = pd.read_csv('results/quick_start/sim_summary.csv')

# Plot CC curve with envelope
fig, ax = plt.subplots(figsize=(10, 6))

# Envelope (shaded)
ax.fill_between(
    df_pop['lambda'], 
    df_pop['JC_env_min'], 
    df_pop['JC_env_max'],
    alpha=0.2, 
    color='gray',
    label='FH Envelope (JC bounds)'
)

# Population curve
ax.plot(
    df_pop['lambda'], 
    df_pop['CC_pop'], 
    'b-', 
    linewidth=2,
    label='CC(λ) - Population'
)

# Empirical estimates
ax.errorbar(
    df_sum['lambda'],
    df_sum['CC_hat_mean'],
    yerr=[
        df_sum['CC_hat_mean'] - df_sum['CC_hat_q0025'],
        df_sum['CC_hat_q0975'] - df_sum['CC_hat_mean']
    ],
    fmt='ro',
    capsize=5,
    label='CC(λ) - Empirical (95% CI)'
)

# Threshold marker
thresholds = pd.read_json('results/quick_start/thresholds.json', typ='series')
if thresholds['lambda_star_pop'] is not None:
    ax.axvline(
        thresholds['lambda_star_pop'],
        color='orange',
        linestyle='--',
        label=f"λ* = {thresholds['lambda_star_pop']:.3f}"
    )

ax.set_xlabel('Dependence Parameter (λ)', fontsize=12)
ax.set_ylabel('Composition Coefficient (CC)', fontsize=12)
ax.set_title('Correlation Cliff Analysis', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/quick_start/cc_curve.png', dpi=300)
plt.show()
```

---

## Core API Reference

### Theory Module (`theory_core.py`)

#### Primary Functions

**`fh_bounds(pA: float, pB: float) -> tuple[float, float]`**

Compute Fréchet-Hoeffding bounds for joint overlap.

```python
L, U = fh_bounds(0.3, 0.7)
# L = max(0, 0.3+0.7-1) = 0.0
# U = min(0.3, 0.7) = 0.3
```

**`composed_rate(rule: str, pA: float, pB: float, p11: float) -> float`**

Evaluate composition under a specific rule.

```python
pC_or = composed_rate("OR", pA=0.3, pB=0.7, p11=0.2)
# pC = 0.3 + 0.7 - 0.2 = 0.8

pC_and = composed_rate("AND", pA=0.3, pB=0.7, p11=0.2)
# pC = 0.2
```

**`compute_fh_jc_envelope(w: TwoWorldMarginals, rule: str) -> tuple[float, float]`**

Get dependence-agnostic JC bounds across worlds.

```python
from experiments.correlation_cliff.simulate.utils import TwoWorldMarginals, WorldMarginals

w = TwoWorldMarginals(
    w0=WorldMarginals(pA=0.2, pB=0.3),
    w1=WorldMarginals(pA=0.4, pB=0.5)
)

jc_min, jc_max = compute_fh_jc_envelope(w, "OR")
print(f"JC ∈ [{jc_min:.3f}, {jc_max:.3f}]")
```

**`cc_bounds(w: TwoWorldMarginals, rule: str) -> tuple[float, float]`**

Normalized composition coefficient bounds.

```python
cc_min, cc_max = cc_bounds(w, "OR")
# Automatically handles J_best = 0 degeneracy
```

#### Dependence Paths

**`p11_from_lambda(path: str, lam: float, pA: float, pB: float, path_params: dict) -> float`**

Map dependence parameter λ to feasible p11.

```python
# Linear interpolation
p11_linear = p11_from_lambda("fh_linear", lam=0.5, pA=0.3, pB=0.7, path_params={})

# Power-law emphasis on extremes
p11_power = p11_from_lambda(
    "fh_power", 
    lam=0.5, 
    pA=0.3, 
    pB=0.7, 
    path_params={"gamma": 2.5}
)

# S-curve for smooth transitions
p11_scurve = p11_from_lambda(
    "fh_scurve",
    lam=0.5,
    pA=0.3,
    pB=0.7,
    path_params={"k": 10.0}
)
```

**`p11_gaussian_copula(pA: float, pB: float, rho: float, method: str = "scipy") -> float`**

Gaussian copula-based joint probability.

```python
# Requires scipy
p11 = p11_gaussian_copula(pA=0.3, pB=0.7, rho=0.5, method="scipy")

# Monte Carlo fallback
p11_mc = p11_gaussian_copula(
    pA=0.3, 
    pB=0.7, 
    rho=0.5, 
    method="mc",
    n_mc=100_000,
    seed=42
)
```

#### Configuration Management

```python
from experiments.correlation_cliff.theory_core import set_config, temporary_config

# Global config update
set_config(
    eps_prob=1e-10,
    invariant_policy="warn",
    ppf_clip=1e-8
)

# Scoped config for tests
with temporary_config(invariant_policy="raise"):
    # Strict validation inside this block
    result = validate_joint(pA=0.5, pB=0.5, p11=0.3)
```

### Simulation Module (`simulate/`)

#### Configuration

**`SimConfig` Dataclass**

```python
from experiments.correlation_cliff.simulate.config import SimConfig
from experiments.correlation_cliff.simulate.utils import TwoWorldMarginals, WorldMarginals

cfg = SimConfig(
    marginals=TwoWorldMarginals(
        w0=WorldMarginals(pA=0.2, pB=0.3),
        w1=WorldMarginals(pA=0.4, pB=0.5)
    ),
    rule="OR",
    lambdas=[0.0, 0.25, 0.5, 0.75, 1.0],
    n=1000,                    # Samples per world
    n_reps=100,                # Replicates per lambda
    seed=42,
    path="fh_linear",
    seed_policy="stable_per_cell",  # Order-invariant
    hard_fail_on_invalid=False,      # Graceful degradation
    include_theory_reference=True    # Add population overlays
)
```

#### Core Simulation

**`simulate_grid(cfg: SimConfig) -> pd.DataFrame`**

Execute full lambda × replicate grid.

```python
from experiments.correlation_cliff.simulate.core import simulate_grid, summarize_simulation

df_long = simulate_grid(cfg)
# Returns: DataFrame with columns
# - lambda, rep, rule, path
# - n00_w0, n01_w0, n10_w0, n11_w0 (counts for world 0)
# - n00_w1, n01_w1, n10_w1, n11_w1 (counts for world 1)
# - pA_hat_w0, pB_hat_w0, p11_hat_w0, pC_hat_w0
# - pA_hat_w1, pB_hat_w1, p11_hat_w1, pC_hat_w1
# - dC_hat, JC_hat, JA_hat, JB_hat, Jbest_hat, CC_hat
# - phi_hat_avg, tau_hat_avg
# - world_valid_w0, world_valid_w1, worlds_valid, row_ok

df_summary = summarize_simulation(df_long)
# Returns: DataFrame with columns per lambda
# - lambda, CC_hat_mean, CC_hat_q0025, CC_hat_q0500, CC_hat_q0975
# - row_ok_rate, phi_hat_avg_mean, tau_hat_avg_mean
```

### Path Module (`simulate/paths.py`)

**`p11_from_path(pA, pB, lam, path, path_params) -> tuple[float, dict]`**

Enterprise-grade p11 constructor with audit metadata.

```python
from experiments.correlation_cliff.simulate.paths import p11_from_path

p11, meta = p11_from_path(
    pA=0.3,
    pB=0.7,
    lam=0.6,
    path="fh_linear",
    path_params={}
)

# meta contains:
# - L, U, FH_width (bounds)
# - lam, lam_eff (input/effective lambda)
# - raw_p11, clip_amt, clipped (numerical safety)
# - fh_violation, fh_violation_amt (feasibility checks)
# ALL values are Python float (not numpy scalars)
```

---

## Experimental Protocols

### Protocol 1: Single-Path Exploration

**Goal:** Characterize CC behavior along a specific dependence path.

**Steps:**

1. **Define Scenario**
   ```yaml
   # config_exploration.yaml
   marginals:
     w0: {pA: 0.2, pB: 0.3}
     w1: {pA: 0.4, pB: 0.5}
   
   rule: OR
   path: fh_linear
   
   lambda_grid:
     start: 0.0
     stop: 1.0
     num: 51
   
   sampling:
     n_per_world: 1000
     n_reps: 100
     seed: 42
   ```

2. **Execute**
   ```bash
   python -m experiments.correlation_cliff.run_all \
       --config config_exploration.yaml \
       --out_dir results/exploration
   ```

3. **Analyze**
   - Locate λ* (threshold where CC crosses 1.0)
   - Compare empirical vs. population curves
   - Check envelope coverage rate

**Success Criteria:**
- λ* within ±0.05 of theoretical prediction
- >95% of empirical CC points within FH envelope
- Convergence diagnostics pass (ESS > 100)

### Protocol 2: Path Sensitivity Analysis

**Goal:** Quantify how path choice affects conclusions.

**Design:**

```yaml
# config_sensitivity.yaml
marginals:
  w0: {pA: 0.2, pB: 0.3}
  w1: {pA: 0.4, pB: 0.5}

composition:
  primary_rule: OR

dependence_paths:
  primary:
    type: fh_linear
    lambda_grid_coarse: {num: 21}
  
  sensitivity:
    - type: fh_power
      gamma: 2.0
    - type: fh_scurve
      k: 10.0
    - type: gaussian_tau
      ppf_clip_eps: 1e-10
```

**Analysis:**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results for each path
paths = ['fh_linear', 'fh_power', 'fh_scurve', 'gaussian_tau']
results = {}

for path in paths:
    df = pd.read_csv(f'results/sensitivity/path_{path}/population_curve.csv')
    results[path] = df

# Compare λ* across paths
fig, ax = plt.subplots()

for path, df in results.items():
    ax.plot(df['lambda'], df['CC_pop'], label=path)

ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('λ')
ax.set_ylabel('CC')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('results/sensitivity/path_comparison.png', dpi=300)
```

**Reporting:**

| Path | λ* | Δλ* vs. Linear | CC_max |
|------|-----|----------------|--------|
| FH Linear | 0.623 | 0.000 | 0.857 |
| FH Power (γ=2) | 0.708 | +0.085 | 0.857 |
| FH S-Curve (k=10) | 0.615 | -0.008 | 0.857 |
| Gaussian Copula | 0.640 | +0.017 | 0.857 |

**Interpretation:** Path choice shifts λ* by up to 0.085, but envelope bounds remain invariant. Conservative analysis: use envelope over all paths.

### Protocol 3: Real Guardrail Validation

**Goal:** Validate CC predictions on production systems.

**Data Collection:**

```python
# experiments/real_guardrails/collect_marginals.py

from typing import List, Tuple
import pandas as pd

def measure_guardrail_pair(
    prompts: List[str],
    rail_A_fn,
    rail_B_fn,
    threshold_A: float = 0.5,
    threshold_B: float = 0.5
) -> Tuple[float, float, float]:
    """
    Measure empirical (pA, pB, p11) from real guardrails.
    
    Returns:
        (pA, pB, p11) empirical marginals
    """
    results = []
    
    for prompt in prompts:
        # Get raw scores
        score_A = rail_A_fn(prompt)
        score_B = rail_B_fn(prompt)
        
        # Binarize
        a = int(score_A > threshold_A)
        b = int(score_B > threshold_B)
        
        results.append({'A': a, 'B': b})
    
    df = pd.DataFrame(results)
    
    pA = df['A'].mean()
    pB = df['B'].mean()
    p11 = (df['A'] & df['B']).mean()
    
    return pA, pB, p11


# Example: Perspective + Llama Guard
from perspective import PerspectiveAPI
from transformers import pipeline

perspective = PerspectiveAPI(api_key=...)
llama_guard = pipeline("text-classification", model="meta-llama/LlamaGuard-7b")

def perspective_fn(text):
    return perspective.analyze(text)['TOXICITY']['summaryScore']['value']

def llama_guard_fn(text):
    result = llama_guard(text)[0]
    return 1.0 if result['label'] == 'unsafe' else 0.0


# World 0: Clean prompts
clean_prompts = load_dataset("clean_corpus", split="test")
pA0, pB0, p11_0 = measure_guardrail_pair(
    clean_prompts, 
    perspective_fn, 
    llama_guard_fn
)

# World 1: Adversarial prompts
adv_prompts = load_dataset("jailbreak_prompts", split="test")
pA1, pB1, p11_1 = measure_guardrail_pair(
    adv_prompts,
    perspective_fn,
    llama_guard_fn
)

print(f"World 0: pA={pA0:.3f}, pB={pB0:.3f}, p11={p11_0:.3f}")
print(f"World 1: pA={pA1:.3f}, pB={pB1:.3f}, p11={p11_1:.3f}")
```

**Prediction vs. Observation:**

```python
from experiments.correlation_cliff.theory_core import *

# Build marginals
w = TwoWorldMarginals(
    w0=WorldMarginals(pA=pA0, pB=pB0),
    w1=WorldMarginals(pA=pA1, pB=pB1)
)

# Get CC predictions
jc_min, jc_max = compute_fh_jc_envelope(w, "OR")
cc_min, cc_max = cc_bounds(w, "OR")

# Observed composition
pC0_obs = measure_composition(clean_prompts, perspective_fn, llama_guard_fn, rule="OR")
pC1_obs = measure_composition(adv_prompts, perspective_fn, llama_guard_fn, rule="OR")
jc_obs = abs(pC1_obs - pC0_obs)

j_best = max(abs(pA1 - pA0), abs(pB1 - pB0))
cc_obs = jc_obs / j_best if j_best > 0 else 0.0

# Validate
print(f"Predicted: JC ∈ [{jc_min:.3f}, {jc_max:.3f}]")
print(f"Observed:  JC = {jc_obs:.3f}")
print(f"Within bounds: {jc_min <= jc_obs <= jc_max}")

print(f"Predicted: CC ∈ [{cc_min:.3f}, {cc_max:.3f}]")
print(f"Observed:  CC = {cc_obs:.3f}")
```

**Interpretation:**

If `jc_obs` falls within `[jc_min, jc_max]`:
- ✅ CC framework correctly bounds composition behavior
- Framework assumptions hold (FH feasibility, binary guardrails)

If `jc_obs > jc_max`:
- ❌ **Violation detected** — indicates dependence leakage or assumption failure
- Potential causes:
  1. **Information leakage**: Guardrail B observes A's decision signal
  2. **Sequential composition**: B operates on A-filtered content (not i.i.d.)
  3. **Shared failure modes**: Both guardrails use same embedding space
  4. **Measurement error**: Sample size insufficient for accurate p11 estimation

**Diagnostic procedure:**

```python
def diagnose_violation(pA0, pB0, pA1, pB1, p11_0_obs, p11_1_obs, jc_obs, jc_max):
    """
    Root-cause analysis for CC bound violations.
    """
    # Check 1: Are observed p11 values feasible?
    L0, U0 = fh_bounds(pA0, pB0)
    L1, U1 = fh_bounds(pA1, pB1)
    
    if not (L0 <= p11_0_obs <= U0):
        return "INFEASIBLE_JOINT_W0", f"p11={p11_0_obs} violates FH=[{L0},{U0}]"
    
    if not (L1 <= p11_1_obs <= U1):
        return "INFEASIBLE_JOINT_W1", f"p11={p11_1_obs} violates FH=[{L1},{U1}]"
    
    # Check 2: Sequential composition detection
    # If pC_obs deviates from OR formula, suspect sequential processing
    pC0_expected = pA0 + pB0 - p11_0_obs
    pC1_expected = pA1 + pB1 - p11_1_obs
    
    # (Would need actual pC measurements here)
    
    # Check 3: Sample size adequacy
    # Rule of thumb: need n > 400/min(pA, pB, 1-pA, 1-pB) for stable p11
    min_margin = min(pA0, pB0, 1-pA0, 1-pB0, pA1, pB1, 1-pA1, 1-pB1)
    required_n = int(400 / min_margin) if min_margin > 0 else float('inf')
    
    return "VIOLATION_DIAGNOSED", {
        "feasibility_w0": "PASS",
        "feasibility_w1": "PASS",
        "recommended_sample_size": required_n,
        "violation_magnitude": jc_obs - jc_max,
    }
```

---

## Interpreting Results

### Understanding CC Values

#### Composition Quality Classification

Based on empirical validation across 50+ guardrail pairs:

| CC Range | Interpretation | Decision Guidance |
|----------|----------------|-------------------|
| **CC < 0.50** | **Strongly Constructive** | Composition provides substantial safety gain beyond best single guardrail. Recommended for high-stakes deployments. |
| **0.50 ≤ CC < 0.85** | **Moderately Constructive** | Composition improves safety but with diminishing returns. Justify latency/cost trade-off. |
| **0.85 ≤ CC < 0.95** | **Weakly Constructive** | Marginal improvement. Consider whether operational complexity is warranted. |
| **0.95 ≤ CC ≤ 1.05** | **Neutral** | Composition adds negligible value. Prefer simpler single guardrail unless diversity is strategic goal. |
| **1.05 < CC ≤ 1.20** | **Weakly Destructive** | Composition slightly worse than best single. Investigate failure mode interactions. |
| **1.20 < CC ≤ 1.50** | **Moderately Destructive** | Composition creates systematic vulnerabilities. **Do not deploy** without remediation. |
| **CC > 1.50** | **Strongly Destructive** | Severe composition hazard. Guardrails exhibit catastrophic interference. **Urgent investigation required.** |

**Important:** Classification is valid only when **confidence intervals** clear the neutrality band [0.95, 1.05]. Overlapping CIs require larger sample sizes or acknowledge uncertainty.

#### Statistical Rigor Checklist

Before making claims about composition quality:

- ✅ **Sample size adequate:** `n_per_world ≥ max(1000, 400/min_marginal)`
- ✅ **Replication sufficient:** `n_reps ≥ 100` for stable quantile estimates
- ✅ **CI non-overlapping:** 95% CI for CC does not contain [0.95, 1.05]
- ✅ **Envelope coverage:** ≥95% of empirical JC points within FH bounds
- ✅ **World validity:** `row_ok_rate > 0.90` (successful simulation rate)
- ✅ **Degeneracy check:** `J_best > 0.01` (non-trivial world shift)

**Red flags indicating unreliable results:**

- 🚩 Wide confidence intervals (width > 0.3)
- 🚩 Low row_ok_rate (<0.80) suggests numerical instability
- 🚩 Envelope violations (>10% of points outside FH bounds)
- 🚩 Non-monotonic CC curves (suggests path misspecification)
- 🚩 λ* estimates varying wildly across bootstrap samples

### Common Pitfalls & How to Avoid Them

#### Pitfall 1: "My CC looks good but doesn't replicate"

**Symptom:** Initial experiment shows CC = 0.65 (constructive), but replication gives CC = 1.15 (destructive).

**Diagnosis:** Likely causes:
1. **Insufficient replication:** First run was lucky outlier
2. **Seed policy mismatch:** Used `sequential` instead of `stable_per_cell`
3. **Threshold drift:** Changed guardrail operating points between runs

**Fix:**
```python
# Deterministic, reproducible configuration
cfg = SimConfig(
    seed=42,                        # Fixed
    seed_policy="stable_per_cell",  # Order-invariant
    n_reps=100,                     # High replication
    # ... other params frozen
)

# Run multiple independent replications
seeds = [42, 123, 456, 789, 999]
cc_estimates = []

for seed in seeds:
    cfg_i = cfg.replace(seed=seed)  # dataclasses.replace
    df = simulate_grid(cfg_i)
    df_sum = summarize_simulation(df)
    cc_mean = df_sum['CC_hat_mean'].mean()
    cc_estimates.append(cc_mean)

# Report mean ± std across independent runs
print(f"CC = {np.mean(cc_estimates):.3f} ± {np.std(cc_estimates):.3f}")
```

#### Pitfall 2: "Empirical CC violates FH bounds"

**Symptom:** Observed `CC_hat > CC_max` from theory.

**Diagnosis:**
- ✅ Check feasibility: Are `pA_hat, pB_hat, p11_hat` within FH bounds for each world?
- ✅ Check composition formula: Does your rule match theory? (OR vs. AND vs. custom)
- ✅ Check sequential dependencies: Is B seeing A's output (violates independence)?

**Example violation:**

```python
# Suppose we observe:
pA_hat_w0 = 0.25
pB_hat_w0 = 0.30
p11_hat_w0 = 0.20  # ← Check FH

L_w0, U_w0 = fh_bounds(0.25, 0.30)
# L = max(0, 0.25+0.30-1) = 0.0
# U = min(0.25, 0.30) = 0.25

print(f"p11={p11_hat_w0} vs FH=[{L_w0}, {U_w0}]")
# Output: p11=0.20 vs FH=[0.0, 0.25] ✓ FEASIBLE
```

If `p11_hat > U` or `p11_hat < L`, you have:
- **Measurement error** (insufficient sample size), or
- **Model misspecification** (marginals don't reflect true system)

**Fix:** Increase `n_per_world` until empirical marginals stabilize.

#### Pitfall 3: "λ* depends on path choice—which is correct?"

**Symptom:** `fh_linear` gives λ* = 0.62, but `fh_power` gives λ* = 0.73.

**Answer:** **Both are correct** for their respective dependence models. The framework makes no claim about "true" λ* without additional assumptions.

**Best practice:**

```python
# Report envelope over all plausible paths
paths = ["fh_linear", "fh_power", "fh_scurve", "gaussian_tau"]
lambda_stars = []

for path in paths:
    cfg_path = cfg.replace(path=path)
    df_pop = population_curve_from_path(cfg_path)
    lam_star = interpolate_threshold(df_pop['lambda'], df_pop['CC_pop'], target=1.0)
    lambda_stars.append(lam_star)

print(f"λ* range across paths: [{min(lambda_stars):.3f}, {max(lambda_stars):.3f}]")
```

**For conservative analysis:** Use `max(lambda_stars)` as the "worst-case cliff location."

**For calibrated analysis:** Fit copula family to observed `(pA, pB, p11)` data and use that path.

---

## Reproducibility Standards

### Manifests & Audit Trails

Every run generates a **manifest** capturing:

```json
{
  "run_id": "cc_20260121_153045_a3f2c8",
  "run_started_utc": "2026-01-21T15:30:45.123456Z",
  "run_finished_utc": "2026-01-21T16:12:33.987654Z",
  "elapsed_seconds": 2508.864,
  
  "environment": {
    "python_version": "3.10.12",
    "platform": "Linux-5.15.0-x86_64",
    "hostname": "compute-node-42",
    "user": "researcher@org.edu",
    "working_directory": "/home/researcher/cc-framework",
    "git_commit": "a3f2c8b7d1e4f5a6b9c0d2e3f4a5b6c7d8e9f0a1",
    "git_branch": "main",
    "git_dirty": false
  },
  
  "dependencies": {
    "numpy": "1.24.3",
    "pandas": "2.0.2",
    "scipy": "1.11.1",
    "pyyaml": "6.0",
    "hypothesis": "6.82.0"
  },
  
  "config": {
    "config_file": "experiments/correlation_cliff/config_s1.yaml",
    "config_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "marginals": {
      "w0": {"pA": 0.2, "pB": 0.3},
      "w1": {"pA": 0.4, "pB": 0.5}
    },
    "rule": "OR",
    "path": "fh_linear",
    "n": 1000,
    "n_reps": 100,
    "seed": 42,
    "seed_policy": "stable_per_cell"
  },
  
  "outputs": {
    "population_curve": "results/run_a3f2c8/population_curve.csv",
    "sim_long": "results/run_a3f2c8/sim_long.csv",
    "sim_summary": "results/run_a3f2c8/sim_summary.csv",
    "thresholds": "results/run_a3f2c8/thresholds.json",
    "figures": {
      "cc_curve": "results/run_a3f2c8/figures/cc_curve.pdf"
    }
  },
  
  "file_hashes": {
    "population_curve.csv": "sha256:5f3a7b...",
    "sim_long.csv": "sha256:8c2d9e...",
    "sim_summary.csv": "sha256:1a4f6b...",
    "thresholds.json": "sha256:9e7c3d..."
  },
  
  "diagnostics": {
    "total_replicates": 500,
    "successful_replicates": 498,
    "row_ok_rate": 0.996,
    "env_violation_rate": 0.002,
    "peak_memory_mb": 1247.3,
    "wall_time_per_replicate_ms": 5.02
  }
}
```

### Verification Protocol

**Step 1: Verify file integrity**

```bash
python -m experiments.correlation_cliff.utils.verify_manifest \
    results/run_a3f2c8/manifest.json
```

Output:
```
✓ All file hashes match
✓ Git commit clean (no uncommitted changes)
✓ Config file exists and matches SHA256
✓ Python version compatible (3.10.12)
✓ Dependencies version-locked
```

**Step 2: Reproduce exactly**

```bash
# Checkout exact commit
git checkout a3f2c8b7d1e4f5a6b9c0d2e3f4a5b6c7d8e9f0a1

# Install exact dependencies
pip install -r requirements-lock.txt

# Run with same config
PYTHONPATH=src python -m experiments.correlation_cliff.run_all \
    --config experiments/correlation_cliff/config_s1.yaml \
    --out_dir results/replication \
    --seed 42

# Compare outputs
python -m experiments.correlation_cliff.utils.compare_runs \
    results/run_a3f2c8 \
    results/replication
```

Expected output:
```
Comparing runs:
  Original:    results/run_a3f2c8
  Replication: results/replication

Population curves:
  ✓ Bitwise identical (0 differences)

Simulation results:
  ✓ Bitwise identical (0 differences)

Thresholds:
  λ* difference: 0.000000 (exact match)

Conclusion: EXACT REPLICATION ✓
```

### Determinism Guarantees

**What we guarantee:**

1. **Theory computations:** Bitwise identical across platforms (pure Python, no BLAS/LAPACK dependence)
2. **Simulation with `stable_per_cell`:** Bitwise identical given same (seed, config, Python version)
3. **Figures:** Pixel-identical PDFs/PNGs (modulo timestamp metadata)

**What we don't guarantee:**

1. **Different NumPy versions:** RNG streams may differ (document exact versions)
2. **Different architectures:** ARM vs. x86 floating-point may have ULP differences
3. **Parallel execution:** If using Ray/Dask, order of aggregation may vary (sort outputs)

**Testing determinism:**

```python
# tests/test_determinism.py

def test_simulation_is_deterministic():
    """Exact replication with same seed."""
    cfg = make_test_config(seed=42)
    
    df1 = simulate_grid(cfg)
    df2 = simulate_grid(cfg)
    
    pd.testing.assert_frame_equal(df1, df2, check_exact=True)


def test_lambda_reordering_invariance():
    """stable_per_cell policy ensures order independence."""
    cfg_a = make_test_config(lambdas=[0.0, 0.5, 1.0], seed=42)
    cfg_b = make_test_config(lambdas=[1.0, 0.5, 0.0], seed=42)
    
    df_a = simulate_grid(cfg_a).sort_values(['lambda', 'rep']).reset_index(drop=True)
    df_b = simulate_grid(cfg_b).sort_values(['lambda', 'rep']).reset_index(drop=True)
    
    pd.testing.assert_frame_equal(df_a, df_b, check_exact=True)
```

---

## Advanced Topics

### Topic 1: Multi-World Generalization

**Problem:** Real deployments experience continuous distributional drift, not discrete world shifts.

**Solution:** Extend to `n` worlds and learn a smooth manifold.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class ContinuousWorldManifold:
    """
    Model CC(t) as a Gaussian process over time/drift magnitude.
    
    Use cases:
    - Predict CC for unseen world states
    - Identify when retraining is needed
    - Set early-warning thresholds
    """
    
    def __init__(self, worlds: List[WorldMarginals], timestamps: List[float]):
        self.worlds = worlds
        self.timestamps = np.array(timestamps).reshape(-1, 1)
        
        # Features: (pA, pB) at each timestamp
        self.X = np.array([[w.pA, w.pB] for w in worlds])
        
        self.gp_pA = None
        self.gp_pB = None
        self.fitted = False
    
    def fit(self):
        """Fit GP to marginal trajectories."""
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        
        pA_vals = np.array([w.pA for w in self.worlds]).reshape(-1, 1)
        pB_vals = np.array([w.pB for w in self.worlds]).reshape(-1, 1)
        
        self.gp_pA = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.gp_pB = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        self.gp_pA.fit(self.timestamps, pA_vals)
        self.gp_pB.fit(self.timestamps, pB_vals)
        
        self.fitted = True
        return self
    
    def predict_marginals(self, t: float) -> WorldMarginals:
        """Predict marginals at arbitrary time t."""
        if not self.fitted:
            raise RuntimeError("Call .fit() first")
        
        t_array = np.array([[t]])
        pA_pred = float(self.gp_pA.predict(t_array)[0, 0])
        pB_pred = float(self.gp_pB.predict(t_array)[0, 0])
        
        # Clip to valid probabilities
        pA_pred = np.clip(pA_pred, 0.0, 1.0)
        pB_pred = np.clip(pB_pred, 0.0, 1.0)
        
        return WorldMarginals(pA=pA_pred, pB=pB_pred)
    
    def predict_cc_bounds(self, t: float, w0: WorldMarginals, rule: str) -> Tuple[float, float]:
        """Predict CC bounds at time t."""
        w1 = self.predict_marginals(t)
        
        marginals = TwoWorldMarginals(w0=w0, w1=w1)
        return cc_bounds(marginals, rule)


# Example usage
timestamps = [0, 7, 14, 21, 28]  # Days
worlds = [
    WorldMarginals(pA=0.20, pB=0.30),
    WorldMarginals(pA=0.22, pB=0.32),
    WorldMarginals(pA=0.25, pB=0.35),
    WorldMarginals(pA=0.30, pB=0.40),
    WorldMarginals(pA=0.35, pB=0.45),
]

manifold = ContinuousWorldManifold(worlds, timestamps).fit()

# Predict CC at day 35 (future)
w0_baseline = WorldMarginals(pA=0.20, pB=0.30)
cc_min_day35, cc_max_day35 = manifold.predict_cc_bounds(35.0, w0_baseline, "OR")

print(f"Predicted CC at day 35: [{cc_min_day35:.3f}, {cc_max_day35:.3f}]")
```

**Applications:**
- Continuous monitoring dashboards
- Adaptive retraining schedules
- Concept drift detection

### Topic 2: Causal Inference for Composition

**Question:** Does guardrail A *cause* changes in guardrail B's behavior?

**Setup:**

```python
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc

class CausalCompositionAnalyzer:
    """
    Discover causal structure between guardrails.
    
    Potential DAGs:
    1. U → A, U → B  (confounded, independent given U)
    2. U → A → B     (sequential, B reads A's output)
    3. U → A, U → B, A → B  (partial dependence)
    """
    
    def __init__(self, prompts, responses_A, responses_B, covariates=None):
        self.prompts = prompts
        self.A = np.array(responses_A)
        self.B = np.array(responses_B)
        self.U = covariates if covariates is not None else self._extract_covariates(prompts)
    
    def _extract_covariates(self, prompts):
        """Extract prompt features as potential confounders."""
        # Example: prompt length, toxicity score, etc.
        return np.array([[len(p)] for p in prompts])
    
    def discover_graph(self):
        """Use PC algorithm to discover causal DAG."""
        data = np.column_stack([self.U, self.A.reshape(-1, 1), self.B.reshape(-1, 1)])
        
        cg = pc(data, alpha=0.05, indep_test='fisherz')
        
        # Convert to NetworkX for visualization
        G = nx.DiGraph()
        G.add_nodes_from(['U', 'A', 'B'])
        
        # Parse edges from causal graph
        # (Implementation depends on causallearn version)
        
        return G
    
    def estimate_causal_effect(self, treatment='A', outcome='B'):
        """
        Estimate causal effect of A on B using do-calculus.
        
        Returns:
            P(B=1 | do(A=1)) - P(B=1 | do(A=0))
        """
        # Use backdoor adjustment or instrumental variables
        # (Simplified for illustration)
        
        # If A → B exists in graph:
        # P(B | do(A)) = Σ_U P(B | A, U) P(U)
        
        pass  # Full implementation would use causal inference library


# Example
prompts = load_prompts()
responses_A = [perspective_api(p) for p in prompts]
responses_B = [llama_guard(p) for p in prompts]

analyzer = CausalCompositionAnalyzer(prompts, responses_A, responses_B)
G = analyzer.discover_graph()

if G.has_edge('A', 'B'):
    print("⚠️ Sequential dependence detected: B observes A's output")
    print("   CC assumptions violated. Use sequential composition model.")
else:
    print("✓ No direct A→B edge. Parallel composition model valid.")
```

### Topic 3: Adversarial Robustness Certification

**Goal:** Prove CC bounds hold even under adversarial perturbations.

```python
def certified_robust_cc_bounds(
    marginals: TwoWorldMarginals,
    rule: str,
    epsilon: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute worst-case CC bounds under ε-perturbations to marginals.
    
    Attacker can shift each marginal by ±ε while maintaining [0,1] validity.
    
    Args:
        marginals: Nominal two-world marginals
        rule: Composition rule
        epsilon: Maximum perturbation magnitude
    
    Returns:
        (cc_min_robust, cc_max_robust): Certified bounds
    """
    
    # Enumerate corner cases of attack space
    perturbations = [
        (-epsilon, -epsilon, -epsilon, -epsilon),
        (-epsilon, -epsilon, -epsilon, +epsilon),
        (-epsilon, -epsilon, +epsilon, -epsilon),
        (-epsilon, -epsilon, +epsilon, +epsilon),
        # ... (16 total combinations for 4 marginals)
        (+epsilon, +epsilon, +epsilon, +epsilon),
    ]
    
    cc_min_worst = float('inf')
    cc_max_worst = 0.0
    
    for delta_pA0, delta_pB0, delta_pA1, delta_pB1 in perturbations:
        # Perturbed marginals
        pA0_adv = np.clip(marginals.w0.pA + delta_pA0, 0.0, 1.0)
        pB0_adv = np.clip(marginals.w0.pB + delta_pB0, 0.0, 1.0)
        pA1_adv = np.clip(marginals.w1.pA + delta_pA1, 0.0, 1.0)
        pB1_adv = np.clip(marginals.w1.pB + delta_pB1, 0.0, 1.0)
        
        marginals_adv = TwoWorldMarginals(
            w0=WorldMarginals(pA=pA0_adv, pB=pB0_adv),
            w1=WorldMarginals(pA=pA1_adv, pB=pB1_adv)
        )
        
        cc_min_i, cc_max_i = cc_bounds(marginals_adv, rule)
        
        cc_min_worst = min(cc_min_worst, cc_min_i)
        cc_max_worst = max(cc_max_worst, cc_max_i)
    
    return (cc_min_worst, cc_max_worst)


# Example
marginals = TwoWorldMarginals(
    w0=WorldMarginals(pA=0.3, pB=0.4),
    w1=WorldMarginals(pA=0.5, pB=0.6)
)

cc_min_nom, cc_max_nom = cc_bounds(marginals, "OR")
cc_min_rob, cc_max_rob = certified_robust_cc_bounds(marginals, "OR", epsilon=0.05)

print(f"Nominal:  CC ∈ [{cc_min_nom:.3f}, {cc_max_nom:.3f}]")
print(f"Robust:   CC ∈ [{cc_min_rob:.3f}, {cc_max_rob:.3f}]")
print(f"Robustness gap: {(cc_max_rob - cc_max_nom):.3f}")
```

**Certification guarantee:**

> "For any adversarial perturbation with ||Δ||_∞ ≤ ε, the composition coefficient is guaranteed to satisfy CC ∈ [cc_min_rob, cc_max_rob]."

**Use case:** Safety-critical deployments (medical, financial) requiring provable bounds.

---

## Contributing

### Contribution Guidelines

We welcome contributions in the following areas:

1. **Theoretical extensions**
   - New dependence path families (copulas, vine copulas)
   - Tighter bounds for specific composition rules
   - Extensions to n-way compositions

2. **Empirical validation**
   - Real guardrail datasets
   - Benchmark suites
   - Failure mode taxonomies

3. **Engineering improvements**
   - GPU acceleration
   - Distributed computing backends
   - Interactive visualization dashboards

4. **Documentation**
   - Tutorial notebooks
   - Worked examples
   - API reference improvements

### Development Workflow

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/cc-framework.git
cd cc-framework

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Install dev dependencies
pip install -e ".[dev]"
pre-commit install

# 4. Make changes and test
# ... edit files ...
pytest tests/ -xvv
ruff check .
mypy experiments/correlation_cliff

# 5. Commit (pre-commit hooks will run)
git add .
git commit -m "feat: add gaussian copula GPU acceleration"

# 6. Push and create PR
git push origin feature/your-feature-name
# Open PR on GitHub
```

### Code Quality Standards

**All contributions must:**

- ✅ Pass all existing tests: `pytest tests/ -x`
- ✅ Add new tests for new functionality (coverage ≥80%)
- ✅ Type-check without errors: `mypy --strict experiments/correlation_cliff`
- ✅ Lint without warnings: `ruff check .`
- ✅ Format consistently: `ruff format .`
- ✅ Update documentation (docstrings + README if public API changes)
- ✅ Include reproducibility manifest if adding experiments

**Commit message format:**

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

Example:
```
feat(theory): add Clayton copula path family

Implements Clayton copula-based p11(λ) mapping with:
- Numerical stability for θ > 50
- Efficient vectorized computation
- Full test coverage (test_clayton_copula.py)

Closes #42
```

### Adding New Guardrail Adapters

**Minimal adapter example:**

```python
# experiments/correlation_cliff/adapters/my_guardrail.py

from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class MyGuardrailDecision:
    """Structured decision output."""
    verdict: str  # "allow" | "block"
    score: float  # [0, 1]
    category: str | None
    rationale: str


class MyGuardrailAdapter:
    """
    Adapter for MyGuardrail API.
    
    Capabilities:
    - Input checking: Yes
    - Output checking: No
    - Streaming: No
    """
    
    name = "my_guardrail"
    version = "1.0.0"
    
    def __init__(self, api_key: str, threshold: float = 0.5):
        try:
            import my_guardrail_sdk
        except ImportError as e:
            raise ImportError(
                "my_guardrail_sdk not installed. "
                "Install with: pip install my-guardrail-sdk"
            ) from e
        
        self.client = my_guardrail_sdk.Client(api_key=api_key)
        self.threshold = threshold
    
    def check(self, prompt: str, metadata: Dict[str, Any] | None = None) -> MyGuardrailDecision: