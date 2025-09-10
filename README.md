# README.md
"""
# CC Framework: Composability Coefficient for AI Guardrail Evaluation

[![Tests](https://github.com/Cubits11/cc-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/Cubits11/cc-framework/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The first quantitative framework for measuring interaction effects between AI safety guardrails.**

## 🎯 Overview

The Composability Coefficient (CC) Framework provides empirical methods to quantify how AI safety mechanisms interact when composed together. Instead of deploying guardrail combinations blindly, CC enables data-driven decisions about when composition helps, hurts, or has no effect.

### Key Innovation

- **CC < 0.95**: 🟢 **Constructive** - Mechanisms synergize beyond individual capabilities
- **0.95 ≤ CC ≤ 1.05**: 🟡 **Independent** - No interaction effects, prefer single best mechanism  
- **CC > 1.05**: 🔴 **Destructive** - Composition creates vulnerabilities absent individually

### Research Impact

This framework bridges the gap between theoretical security composition (differential privacy, cryptography) and practical AI safety engineering, providing the first systematic methodology for guardrail interaction analysis.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/pranavbhave/cc-framework
cd cc-framework

# Install dependencies  
make install

# Run smoke test (< 5 minutes)
make reproduce-smoke

# Run full MVP experiment (2-4 hours)
make reproduce-mvp
```

### Basic Usage

```python
from cc.exp.run_two_world import main
from cc.core.attackers import RandomInjectionAttacker
from cc.guardrails import KeywordBlocker, SemanticFilter

# Define guardrails to test
guardrail_A = KeywordBlocker(keywords=["hack", "exploit"])
guardrail_B = SemanticFilter(harmful_templates=["bypass security"])

# Run CC analysis
results = run_cc_experiment(
    guardrails=[guardrail_A, guardrail_B],
    n_sessions=5000,
    attacker_type="genetic_algorithm"
)

print(f"CC_max: {results['cc_max']:.3f}")
print(f"Interaction: {results['interaction_type']}")
```

## 📊 Methodology

### Two-World Protocol

CC uses an adaptive two-world distinguishability protocol:

1. **World 0**: Unprotected system (baseline)
2. **World 1**: System with guardrail composition
3. **Adaptive Attacker**: Learns to distinguish between worlds
4. **J-Statistic**: Measures distinguishing advantage (Youden's J)
5. **Bootstrap CIs**: Statistical confidence in effects

### Theoretical Foundation

Based on Youden's J-statistic from medical diagnostics, adapted for security evaluation:

```
J = max_threshold (TPR(t) - FPR(t))
CC_max = J_composed / max(J_individual)
```

Where J ∈ [0,1] with J=0 indicating perfect privacy and J=1 indicating complete leakage.

## 🔬 Experimental Design

### Attack Strategies

**Tier A (CPU-Safe)**:
- Random Injection: Vocabulary-based prompt generation
- Genetic Algorithm: Evolution-based optimization  

**Tier B (GPU-Required)**:
- Deep RL Attackers: PPO-trained adaptive adversaries
- Side-Channel Analysis: Timing and resource fingerprinting

### Guardrail Types

- **Keyword Blocking**: Pattern-based filtering with fuzzy matching
- **Semantic Filtering**: Embedding similarity to harmful templates  
- **Regex Filtering**: Regular expression based blocking
- **Composition**: Arbitrary combinations with utility matching

### Statistical Framework

- **Bootstrap Confidence Intervals**: 2,000 resamples, percentile method
- **Preregistered Hypotheses**: OSF registration prevents p-hacking
- **Multiple Testing Correction**: Holm-Bonferroni for family-wise error
- **Effect Size Classification**: Cohen's conventions adapted for J-statistic

## 📁 Project Structure

```
cc-framework/
├── src/cc/
│   ├── core/           # Framework core
│   │   ├── models.py   # Data structures  
│   │   ├── stats.py    # Statistical methods
│   │   ├── protocol.py # Two-world implementation
│   │   ├── attackers.py # Attack strategies
│   │   └── logging.py  # Cryptographic audit logs
│   ├── guardrails/     # Guardrail implementations
│   │   ├── base.py     # Abstract interface
│   │   ├── keyword_blocker.py
│   │   ├── semantic_filter.py
│   │   └── regex_filter.py
│   ├── exp/           # Experiment runners
│   │   ├── run_two_world.py # Main runner
│   │   └── configs/   # YAML configurations
│   ├── analysis/      # Results analysis
│   │   └── generate_figures.py
│   └── tests/         # Test suite
├── logs/              # Audit logs (JSONL)
├── results/           # Analysis outputs
├── figures/           # Generated plots  
├── paper/             # LaTeX manuscript
└── docs/              # Documentation
```

## 🎛️ Configuration

Experiments are configured via YAML files:

```yaml
# src/cc/exp/configs/example.yaml
seed: 42
baseline_success_rate: 0.6
episode_length: 10

vocab_harmful: ["exploit", "bypass", "hack"]
vocab_benign: ["help", "please", "normal"]

attacker:
  type: "genetic_algorithm"
  population_size: 50
  mutation_rate: 0.1

guardrails:
  - name: "keyword"
    params:
      keywords: ["exploit", "hack"]
      fuzzy_threshold: 0.8
    fpr_target: 0.05
    
  - name: "semantic"
    params:
      harmful_templates: ["bypass security controls"]
      similarity_threshold: 0.7
    fpr_target: 0.05
```

## 🔧 Advanced Usage

### Custom Guardrails

```python
from cc.guardrails.base import Guardrail

class MyCustomGuardrail(Guardrail):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def blocks(self, text: str) -> bool:
        return self.score(text) > self.threshold
        
    def score(self, text: str) -> float:
        # Your custom scoring logic
        return my_ml_model.predict_proba(text)[1]
        
    def calibrate(self, benign_texts, target_fpr=0.05):
        scores = [self.score(text) for text in benign_texts]
        self.threshold = np.percentile(scores, 100 * (1 - target_fpr))
```

### Custom Attackers

```python
from cc.core.attackers import AttackStrategy

class MyAttacker(AttackStrategy):
    def generate_attack(self, history):
        # Generate attack based on history
        return {"attack_id": "custom_001", "prompt": "..."}
    
    def update_strategy(self, attack, result):
        # Learn from attack result
        pass
        
    def reset(self):
        # Reset for new experiment
        pass
```

### Batch Experiments

```python
# Run parameter sweep
configs = [
    "configs/toy_A.yaml",
    "configs/toy_B.yaml", 
    "configs/main_study.yaml"
]

results = []
for config in configs:
    result = run_experiment(config, n_sessions=10000)
    results.append(result)

# Generate comparative analysis
create_cc_phase_diagram(results, "comparison.png")
```

## 📈 Results & Analysis

### Automated Analysis

```bash
# Generate all figures
make reproduce-figures

# Full analysis pipeline
make reproduce-paper
```

### Key Outputs

1. **CC Phase Diagram**: Interaction effects across parameter space
2. **Bootstrap CI Plots**: Statistical confidence visualization  
3. **Metrics Dashboard**: CC_max, Δ_add, effect sizes
4. **Audit Logs**: Complete experiment provenance with hash chaining

### Interpretation Guide

| CC_max Range | Interpretation | Action |
|-------------|----------------|---------|
| < 0.90 | Strong constructive | Deploy composition |
| 0.90 - 0.95 | Weak constructive | Consider composition |
| 0.95 - 1.05 | Independent | Use single best |
| 1.05 - 1.10 | Weak destructive | Avoid composition |
| > 1.10 | Strong destructive | Redesign architecture |

## 🧪 Reproducibility

### Computational Requirements

| Tier | Sessions | Time | Resources |
|------|----------|------|-----------|
| Smoke Test | 200 | 5 min | Single CPU |
| MVP | 5,000 | 2-4 hours | 4 CPU cores |
| Main Study | 20,000 | 8-16 hours | 8 CPU cores or PSU cluster |
| Full Scale | 100,000+ | 1-2 days | GPU cluster recommended |

### Verification Suite

```bash
# Test bootstrap coverage
make verify-statistics

# Test guardrail calibration  
make verify-invariants

# Test audit chain integrity
make verify-audit

# Run full test suite
make test
```

### Reports
- CC only: `python -m cc.cartographer.cli build-reports --mode cc`
- CCC only: `python -m cc.cartographer.cli build-reports --mode ccc` (requires addenda CSVs under `evaluation/ccc/addenda/`)
- Full pane: `python -m cc.cartographer.cli build-reports --mode all`

**Neutrality band:** `[0.95, 1.05]`. We classify only if CI(CC) clears the band.
**Assumptions:** thresholds jointly realizable; FH applied per class; no independence assumed.

## 📚 Academic Context

### IST 496 Independent Study (Fall 2025)
- **Student**: Pranav Bhave  
- **Advisor**: Dr. Peng Liu
- **Institution**: Penn State University
- **Credits**: 3 (135 hours)

### Publication Pipeline
- **Tier A**: Technical report (15-20 pages)
- **Tier B**: Workshop paper (AISec, ML Safety Workshop)
- **Future**: Conference paper (IEEE S&P, ACM CCS, NeurIPS)

### Related Work

This framework builds on:
- **Differential Privacy**: Composition theorems and privacy accounting
- **Cryptographic Security**: Two-world distinguishability games
- **Medical Diagnostics**: ROC analysis and Youden's J-statistic  
- **Quantitative Information Flow**: Decision-theoretic leakage metrics
- **AI Safety**: Constitutional AI, guardrail engineering practices

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
make install
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code  
black src/ tests/
```

### Contribution Guidelines

1. **Issues**: Use GitHub issues for bugs and feature requests
2. **Pull Requests**: Include tests and documentation
3. **Code Style**: Black formatting, type hints, docstrings
4. **Testing**: Maintain >85% coverage
5. **Security**: No hardcoded secrets, validate inputs

### Research Collaboration

Interested in collaboration? Areas of active development:

- **N-way Composition**: Beyond pairwise interactions
- **Theoretical Bounds**: Analytical CC limits for guardrail families  
- **Industry Applications**: Real-world deployment case studies
- **Standardization**: IEEE/ISO standard development

## 📄 License & Citation

### License
MIT License - see [LICENSE](LICENSE) for details.

### Citation

```bibtex
@misc{bhave2025cc,
  title={The Composability Coefficient: Measuring AI Guardrail Interactions},
  author={Bhave, Pranav and Liu, Peng},
  year={2025},
  institution={Penn State University},
  url={https://github.com/Cubits11/cc-framework}
}
```

### Acknowledgments

- **Dr. Peng Liu**: Research supervision and methodology guidance
- **PSU IST**: Computing resources and academic support
- **CC Framework Contributors**: Open source development

## 🆘 Support

### Documentation
- 📖 [Full Documentation](docs/)
- 🎥 [Video Tutorials](docs/tutorials/)  
- 📊 [Example Notebooks](examples/)

### Getting Help
- 💬 [GitHub Discussions](https://github.com/Cubits11/cc-framework/discussions)
- 🐛 [Issue Tracker](https://github.com/Cubits11/cc-framework/issues)
- 📧 Email: [ppb5272@psu.edu](mailto:ppb5272@psu.edu)
