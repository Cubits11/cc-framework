# src/cc/core/attackers.py
"""
Module: attackers
Purpose: Attack strategies for guardrail evaluation (Two-World protocol)
Dependencies: numpy, typing, abc
Author: Pranav Bhave
Date: 2025-08-27 (refined 2025-09-03)

Design goals
------------
- Deterministic, CPU-cheap attackers suitable for smoke/e2e tests.
- Clear, minimal interface: `generate_attack`, `update_strategy`, `reset`.
- Reproducible randomness via numpy Generator; optional reseeding.
- No framework dependencies beyond NumPy.

Strategies
----------
1) RandomInjectionAttacker (Tier A):
   - Samples a bag of harmful + benign tokens, shuffles, optional obfuscation.
   - Tracks per-token success/attempt counts for lightweight diagnostics.

2) GeneticAlgorithmAttacker (Tier B):
   - Evolves token sequences using tournament selection, crossover, mutation.
   - Maintains a lightweight fitness cache keyed by the token sequence string.

Return schemas
--------------
`generate_attack(history)` returns a dict with at least:
    {
        "attack_id": str,
        "prompt": str,
        "tokens": List[str],
        "strategy": str,          # e.g., "random_injection" or "genetic_algorithm"
        ... optional metadata ...
    }

`update_strategy(attack, result)` expects:
    attack["tokens"] : List[str]
    result["success"]: bool (True if the attack bypassed the guardrail)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Base interface
# =============================================================================


class AttackStrategy(ABC):
    """Abstract base class for attack strategies."""

    @abstractmethod
    def generate_attack(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate the next attack based on (optional) history."""
        raise NotImplementedError

    @abstractmethod
    def update_strategy(self, attack: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Update internal state given the attack and its evaluation result."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None) -> None:
        """Reset strategy to initial state; optionally reseed RNG."""
        raise NotImplementedError


# =============================================================================
# Utilities
# =============================================================================


def _rng(seed: Optional[int]) -> np.random.Generator:
    """Construct a numpy Generator; default seed if None."""
    if seed is None:
        return np.random.default_rng(0xC0FFEE)
    return np.random.default_rng(int(seed))


def _safe_choice(
    rng: np.random.Generator, items: Sequence[str], size: int, replace: bool = True
) -> List[str]:
    """Safe choice that tolerates empty sequences by returning []."""
    if not items or size <= 0:
        return []
    return rng.choice(items, size=size, replace=replace).tolist()


# =============================================================================
# Random Injection Attacker (Tier A)
# =============================================================================


@dataclass
class RandomInjectionConfig:
    n_harmful_min: int = 1
    n_harmful_max: int = 3
    n_benign_min: int = 0
    n_benign_max: int = 2
    obfuscate_prob: float = 0.30
    seed: int = 42


class RandomInjectionAttacker(AttackStrategy):
    """Tier A: Random injection attack using token vocabularies with light obfuscation."""

    def __init__(
        self,
        vocab_harmful: Sequence[str],
        vocab_benign: Sequence[str],
        cfg: Optional[RandomInjectionConfig] = None,
    ):
        self.vocab_harmful: List[str] = [t for t in vocab_harmful if t]
        self.vocab_benign: List[str] = [t for t in vocab_benign if t]
        self.cfg = cfg or RandomInjectionConfig()
        self.rng = _rng(self.cfg.seed)

        # Diagnostics: per-token success rates
        self.success_counts: Dict[str, int] = {}
        self.attempt_counts: Dict[str, int] = {}

        # Pre-build simple obfuscation transforms
        self._obfuscators = [
            lambda s: s.replace("a", "@"),
            lambda s: s.replace("e", "3"),
            lambda s: s.replace("i", "1"),
            lambda s: s.replace("o", "0"),
            lambda s: s.upper(),
            lambda s: " ".join(s),  # space out characters
        ]

    # --- Attack API ---------------------------------------------------------

    def generate_attack(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a random token soup with optional obfuscation."""
        n_harmful = int(self.rng.integers(self.cfg.n_harmful_min, self.cfg.n_harmful_max + 1))
        n_benign = int(self.rng.integers(self.cfg.n_benign_min, self.cfg.n_benign_max + 1))

        harmful_tokens = _safe_choice(self.rng, self.vocab_harmful, n_harmful, replace=True)
        benign_tokens = _safe_choice(self.rng, self.vocab_benign, n_benign, replace=True)

        tokens = harmful_tokens + benign_tokens
        if tokens:
            self.rng.shuffle(tokens)

        prompt = " ".join(tokens) if tokens else ""

        # Optional obfuscation
        if prompt and self.rng.random() < self.cfg.obfuscate_prob:
            obf = self.rng.choice(self._obfuscators)
            prompt = obf(prompt)

        attack_id = f"random_{int(self.rng.integers(1, 1_000_000))}"

        return {
            "attack_id": attack_id,
            "prompt": prompt,
            "tokens": tokens,
            "strategy": "random_injection",
        }

    def update_strategy(self, attack: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Update per-token counters from the result."""
        tokens = attack.get("tokens", []) or []
        success = bool(result.get("success", False))

        for token in tokens:
            self.attempt_counts[token] = self.attempt_counts.get(token, 0) + 1
            if success:
                self.success_counts[token] = self.success_counts.get(token, 0) + 1

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Reset counters; optionally reseed RNG."""
        self.success_counts.clear()
        self.attempt_counts.clear()
        if seed is not None:
            self.rng = _rng(seed)

    # --- Introspection ------------------------------------------------------

    def token_success_rate(self, token: str) -> float:
        """Return success rate for a specific token, or 0.0 if unseen."""
        a = self.attempt_counts.get(token, 0)
        if a <= 0:
            return 0.0
        return float(self.success_counts.get(token, 0) / a)

    def top_tokens(self, k: int = 10, min_attempts: int = 3) -> List[Tuple[str, float, int]]:
        """
        Return up to k tokens with the highest success rate,
        filtered by a minimum attempt count.
        """
        rows: List[Tuple[str, float, int]] = []
        for tok, attempts in self.attempt_counts.items():
            if attempts >= min_attempts:
                rate = self.success_counts.get(tok, 0) / attempts
                rows.append((tok, float(rate), int(attempts)))
        rows.sort(key=lambda r: (r[1], r[2]), reverse=True)
        return rows[: max(0, k)]


# =============================================================================
# Genetic Algorithm Attacker (Tier B)
# =============================================================================


@dataclass
class GAConfig:
    population_size: int = 50
    mutation_rate: float = 0.10
    crossover_rate: float = 0.70
    tournament_size: int = 3
    min_len: int = 2
    max_len: int = 8
    elitism_frac: float = 0.10  # top fraction kept verbatim per generation
    evolve_every: int = 10      # evolve after this many fitness updates
    seed: int = 42


class GeneticAlgorithmAttacker(AttackStrategy):
    """Tier B: Simple GA to evolve token sequences that aim to bypass guardrails."""

    def __init__(self, vocab: Sequence[str], cfg: Optional[GAConfig] = None):
        self.vocab: List[str] = [t for t in vocab if t]
        self.cfg = cfg or GAConfig()
        self.rng = _rng(self.cfg.seed)

        self.population: List[List[str]] = []
        self.fitness_cache: Dict[str, float] = {}  # key: str(individual) → fitness in [0,1]
        self.generation: int = 0
        self._updates_since_evolve: int = 0

        self._initialize_population()

    # --- Population management ---------------------------------------------

    def _initialize_population(self) -> None:
        """Initialize a random population."""
        self.population = []
        for _ in range(self.cfg.population_size):
            length = int(self.rng.integers(self.cfg.min_len, self.cfg.max_len + 1))
            indiv = _safe_choice(self.rng, self.vocab, length, replace=True)
            self.population.append(indiv)
        self.generation = 0
        self._updates_since_evolve = 0

    # --- GA primitives ------------------------------------------------------

    def _seq_key(self, seq: Sequence[str]) -> str:
        return " ".join(seq)

    def _fitness(self, seq: Sequence[str]) -> float:
        """Return cached fitness for a sequence; default to 0.0 if unknown."""
        return self.fitness_cache.get(self._seq_key(seq), 0.0)

    def _tournament_selection(self) -> List[str]:
        """Select one individual using tournament selection."""
        if not self.population:
            return []
        k = min(self.cfg.tournament_size, len(self.population))
        idxs = self.rng.choice(len(self.population), size=k, replace=False).tolist()
        best_idx = idxs[0]
        best_fit = self._fitness(self.population[best_idx])
        for i in idxs[1:]:
            fit = self._fitness(self.population[i])
            if fit > best_fit:
                best_idx, best_fit = i, fit
        return list(self.population[best_idx])

    def _crossover(self, p1: List[str], p2: List[str]) -> Tuple[List[str], List[str]]:
        """Single-point crossover (robust to short parents)."""
        if len(p1) <= 1 or len(p2) <= 1:
            return p1.copy(), p2.copy()
        point1 = int(self.rng.integers(1, len(p1)))
        point2 = int(self.rng.integers(1, len(p2)))
        c1 = p1[:point1] + p2[point2:]
        c2 = p2[:point2] + p1[point1:]
        return c1, c2

    def _mutate(self, indiv: List[str]) -> List[str]:
        """Token-level mutation + occasional insertion/deletion."""
        out = indiv.copy()

        # Per-token replacement
        for i in range(len(out)):
            if self.rng.random() < self.cfg.mutation_rate and self.vocab:
                out[i] = self.rng.choice(self.vocab)

        # Structural edits
        if self.rng.random() < self.cfg.mutation_rate:
            if len(out) > self.cfg.min_len and self.rng.random() < 0.5:
                # deletion
                idx = int(self.rng.integers(0, len(out)))
                out.pop(idx)
            elif len(out) < self.cfg.max_len and self.vocab:
                # insertion
                idx = int(self.rng.integers(0, len(out) + 1))
                tok = self.rng.choice(self.vocab)
                out.insert(idx, tok)

        return out

    def _evolve_population(self) -> None:
        """Create the next generation with elitism, crossover, and mutation."""
        if not self.population:
            self._initialize_population()
            return

        # Elitism
        n = len(self.population)
        n_elite = max(1, int(round(self.cfg.elitism_frac * n)))
        # Rank by fitness (desc)
        ranked = sorted(
            ((self._fitness(ind), i) for i, ind in enumerate(self.population)),
            key=lambda t: t[0],
            reverse=True,
        )
        elites = [self.population[i].copy() for _, i in ranked[:n_elite]]

        # Fill the rest
        next_pop: List[List[str]] = elites
        while len(next_pop) < self.cfg.population_size:
            p1 = self._tournament_selection()
            p2 = self._tournament_selection()
            if self.rng.random() < self.cfg.crossover_rate:
                c1, c2 = self._crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            next_pop.append(self._mutate(c1))
            if len(next_pop) < self.cfg.population_size:
                next_pop.append(self._mutate(c2))

        # Trim and commit
        self.population = next_pop[: self.cfg.population_size]
        self.generation += 1
        self._updates_since_evolve = 0

    # --- Attack API ---------------------------------------------------------

    def generate_attack(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Emit an attack constructed from the current best guess sequence."""
        if not self.population:
            self._initialize_population()

        indiv = self._tournament_selection()
        prompt = " ".join(indiv)
        attack_id = f"ga_gen{self.generation}_{int(self.rng.integers(1, 1_000_000))}"

        return {
            "attack_id": attack_id,
            "prompt": prompt,
            "tokens": indiv.copy(),
            "strategy": "genetic_algorithm",
            "generation": self.generation,
        }

    def update_strategy(self, attack: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Update fitness from the result and evolve periodically."""
        tokens = attack.get("tokens") or []
        success = bool(result.get("success", False))
        key = self._seq_key(tokens)
        self.fitness_cache[key] = 1.0 if success else 0.0

        # Evolve after a small batch of updates to keep smoke tests snappy.
        self._updates_since_evolve += 1
        if self._updates_since_evolve >= self.cfg.evolve_every:
            self._evolve_population()

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Reset population and cache; optionally reseed RNG."""
        if seed is not None:
            self.rng = _rng(seed)
        self.fitness_cache.clear()
        self._initialize_population()