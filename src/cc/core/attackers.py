# src/cc/core/attackers.py
"""
Module: attackers
Purpose: Attack strategies for guardrail evaluation
Dependencies: numpy, typing, abc
Author: Pranav Bhave
Date: 2025-08-27
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class AttackStrategy(ABC):
    """Abstract base class for attack strategies"""

    @abstractmethod
    def generate_attack(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate next attack based on history"""
        pass

    @abstractmethod
    def update_strategy(self, attack: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Update strategy based on attack result"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset strategy to initial state"""
        pass


class RandomInjectionAttacker(AttackStrategy):
    """Tier A: Random injection attack using vocabulary"""

    def __init__(
        self,
        vocab_harmful: List[str],
        vocab_benign: List[str],
        base_success: float = 0.6,
        seed: int = 42,
    ):
        self.vocab_harmful = vocab_harmful
        self.vocab_benign = vocab_benign
        self.base_success = base_success
        self.rng = np.random.default_rng(seed)
        self.success_counts: Dict[str, int] = {}
        self.attempt_counts: Dict[str, int] = {}

    def generate_attack(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate random injection attack"""
        # Choose number of harmful and benign tokens
        n_harmful = self.rng.integers(1, 4)
        n_benign = self.rng.integers(0, 3)

        # Sample tokens
        harmful_tokens = self.rng.choice(self.vocab_harmful, size=n_harmful, replace=True)
        benign_tokens = (
            self.rng.choice(self.vocab_benign, size=n_benign, replace=True) if n_benign > 0 else []
        )

        # Combine and shuffle
        all_tokens = list(harmful_tokens) + list(benign_tokens)
        self.rng.shuffle(all_tokens)

        # Create prompt
        prompt = " ".join(all_tokens)

        # Add obfuscation with some probability
        if self.rng.random() < 0.3:
            prompt = self._obfuscate(prompt)

        attack_id = f"random_{self.rng.integers(1, 1_000_000)}"

        return {
            "attack_id": attack_id,
            "prompt": prompt,
            "tokens": all_tokens,
            "strategy": "random_injection",
        }

    def _obfuscate(self, text: str) -> str:
        """Apply simple obfuscation techniques"""
        techniques = [
            lambda s: s.replace("a", "@"),
            lambda s: s.replace("e", "3"),
            lambda s: s.replace("i", "1"),
            lambda s: s.replace("o", "0"),
            lambda s: s.upper(),
            lambda s: " ".join(s),  # space out characters
        ]

        technique = self.rng.choice(techniques)
        return technique(text)

    def update_strategy(self, attack: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Update success tracking for adaptive weighting"""
        attack_id = attack["attack_id"]
        success = result.get("success", False)

        # Track per-token success rates
        for token in attack.get("tokens", []):
            if token not in self.success_counts:
                self.success_counts[token] = 0
                self.attempt_counts[token] = 0

            if success:
                self.success_counts[token] += 1
            self.attempt_counts[token] += 1

    def reset(self) -> None:
        """Reset strategy state"""
        self.success_counts.clear()
        self.attempt_counts.clear()


class GeneticAlgorithmAttacker(AttackStrategy):
    """Tier B: Genetic algorithm-based attacker"""

    def __init__(
        self,
        vocab: List[str],
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        seed: int = 42,
    ):
        self.vocab = vocab
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.rng = np.random.default_rng(seed)

        # Population: list of token sequences
        self.population: List[List[str]] = []
        self.fitness_cache: Dict[str, float] = {}
        self.generation = 0

        self._initialize_population()

    def _initialize_population(self):
        """Initialize random population"""
        self.population = []
        for _ in range(self.population_size):
            length = self.rng.integers(2, 8)
            individual = self.rng.choice(self.vocab, size=length, replace=True).tolist()
            self.population.append(individual)

    def generate_attack(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate attack using current population"""
        if not self.population:
            self._initialize_population()

        # Select individual via tournament selection
        individual = self._tournament_selection()

        # Create prompt
        prompt = " ".join(individual)
        attack_id = f"ga_gen{self.generation}_{self.rng.integers(1, 10000)}"

        return {
            "attack_id": attack_id,
            "prompt": prompt,
            "tokens": individual.copy(),
            "strategy": "genetic_algorithm",
            "generation": self.generation,
        }

    def _tournament_selection(self) -> List[str]:
        """Select individual via tournament selection"""
        tournament = self.rng.choice(
            len(self.population),
            size=min(self.tournament_size, len(self.population)),
            replace=False,
        )

        # Select best individual from tournament (higher fitness = better)
        best_idx = tournament[0]
        best_fitness = self.fitness_cache.get(str(self.population[best_idx]), 0)

        for idx in tournament[1:]:
            fitness = self.fitness_cache.get(str(self.population[idx]), 0)
            if fitness > best_fitness:
                best_idx = idx
                best_fitness = fitness

        return self.population[best_idx].copy()

    def update_strategy(self, attack: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Update fitness and evolve population"""
        tokens = attack.get("tokens", [])
        success = result.get("success", False)

        # Update fitness cache
        key = str(tokens)
        self.fitness_cache[key] = float(success)

        # Evolve population every few updates
        if len(self.fitness_cache) % 10 == 0:
            self._evolve_population()

    def _evolve_population(self):
        """Evolve the population"""
        new_population = []

        # Keep best individuals (elitism)
        elite_count = max(1, self.population_size // 10)
        elite_indices = self._get_elite_indices(elite_count)
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())

        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            if self.rng.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])
            else:
                new_population.extend([self._mutate(parent1.copy()), self._mutate(parent2.copy())])

        # Trim to exact size
        new_population = new_population[: self.population_size]
        self.population = new_population
        self.generation += 1

    def _get_elite_indices(self, count: int) -> List[int]:
        """Get indices of elite individuals"""
        fitness_scores = []
        for i, individual in enumerate(self.population):
            key = str(individual)
            fitness = self.fitness_cache.get(key, 0)
            fitness_scores.append((fitness, i))

        fitness_scores.sort(reverse=True)  # Higher fitness first
        return [idx for _, idx in fitness_scores[:count]]

    def _crossover(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        """Single-point crossover"""
        if len(parent1) <= 1 or len(parent2) <= 1:
            return parent1.copy(), parent2.copy()

        point1 = self.rng.integers(1, len(parent1))
        point2 = self.rng.integers(1, len(parent2))

        child1 = parent1[:point1] + parent2[point2:]
        child2 = parent2[:point2] + parent1[point1:]

        return child1, child2

    def _mutate(self, individual: List[str]) -> List[str]:
        """Mutate individual"""
        mutated = individual.copy()

        for i in range(len(mutated)):
            if self.rng.random() < self.mutation_rate:
                mutated[i] = self.rng.choice(self.vocab)

        # Sometimes add or remove tokens
        if self.rng.random() < self.mutation_rate:
            if len(mutated) > 2 and self.rng.random() < 0.5:
                # Remove token
                idx = self.rng.integers(0, len(mutated))
                mutated.pop(idx)
            elif len(mutated) < 10:
                # Add token
                idx = self.rng.integers(0, len(mutated) + 1)
                new_token = self.rng.choice(self.vocab)
                mutated.insert(idx, new_token)

        return mutated

    def reset(self) -> None:
        """Reset strategy state"""
        self.fitness_cache.clear()
        self.generation = 0
        self._initialize_population()
