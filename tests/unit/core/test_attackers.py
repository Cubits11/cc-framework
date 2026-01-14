# tests/unit/test_attackers.py
"""Deterministic tests for attack strategies."""

from cc.core.attackers import (
    GAConfig,
    GeneticAlgorithmAttacker,
    RandomInjectionAttacker,
    RandomInjectionConfig,
    _rng,
    _safe_choice,
)


class TestRandomInjectionAttacker:
    def test_generate_update_reset(self) -> None:
        vocab_harmful = ["h1", "h2", "h3"]
        vocab_benign = ["b1", "b2"]
        cfg = RandomInjectionConfig(
            n_harmful_min=2,
            n_harmful_max=2,
            n_benign_min=1,
            n_benign_max=1,
            obfuscate_prob=0.0,
            seed=123,
        )

        attacker = RandomInjectionAttacker(vocab_harmful, vocab_benign, cfg)

        attack = attacker.generate_attack([])

        # Compute expected deterministic tokens using the same RNG sequence
        rng = _rng(cfg.seed)
        n_harm = int(rng.integers(cfg.n_harmful_min, cfg.n_harmful_max + 1))
        n_ben = int(rng.integers(cfg.n_benign_min, cfg.n_benign_max + 1))
        expected_tokens = _safe_choice(rng, vocab_harmful, n_harm, replace=True)
        expected_tokens += _safe_choice(rng, vocab_benign, n_ben, replace=True)
        rng.shuffle(expected_tokens)
        expected_prompt = " ".join(expected_tokens) if expected_tokens else ""

        assert attack["tokens"] == expected_tokens
        assert attack["prompt"] == expected_prompt

        # update_strategy increments counts correctly
        attacker.update_strategy(attack, {"success": True})
        for tok in expected_tokens:
            assert attacker.attempt_counts[tok] == 1
            assert attacker.success_counts[tok] == 1

        attacker.update_strategy(attack, {"success": False})
        for tok in expected_tokens:
            assert attacker.attempt_counts[tok] == 2
            assert attacker.success_counts[tok] == 1

        # reset clears counts and reseeds RNG
        attacker.reset(seed=123)
        assert attacker.attempt_counts == {}
        assert attacker.success_counts == {}

        attack2 = attacker.generate_attack([])
        assert attack2 == attack


class TestGeneticAlgorithmAttacker:
    def test_generate_update_reset(self) -> None:
        vocab = ["a", "b", "c"]
        cfg = GAConfig()
        cfg.population_size = 4
        cfg.min_len = cfg.max_len = 3
        cfg.mutation_rate = 0.0
        cfg.crossover_rate = 0.0
        cfg.tournament_size = 2
        cfg.evolve_every = 2
        cfg.seed = 123

        attacker = GeneticAlgorithmAttacker(vocab, cfg)

        attack = attacker.generate_attack([])

        # Reproduce expected sequence
        rng = _rng(cfg.seed)
        population = []
        for _ in range(cfg.population_size):
            length = int(rng.integers(cfg.min_len, cfg.max_len + 1))
            indiv = _safe_choice(rng, vocab, length, replace=True)
            population.append(indiv)
        k = min(cfg.tournament_size, len(population))
        idxs = rng.choice(len(population), size=k, replace=False).tolist()
        expected_tokens = population[idxs[0]]
        expected_prompt = " ".join(expected_tokens)

        assert attack["tokens"] == expected_tokens
        assert attack["prompt"] == expected_prompt
        assert attack["generation"] == 0

        key = " ".join(attack["tokens"])
        attacker.update_strategy(attack, {"success": True})
        assert attacker.fitness_cache[key] == 1.0
        assert attacker.generation == 0

        attacker.update_strategy(attack, {"success": False})
        assert attacker.fitness_cache[key] == 0.0
        assert attacker.generation == 1  # evolved after two updates

        attacker.reset(seed=123)
        assert attacker.generation == 0
        assert attacker.fitness_cache == {}

        attack2 = attacker.generate_attack([])
        assert attack2["tokens"] == expected_tokens
        assert attack2["prompt"] == expected_prompt
        assert attack2["generation"] == 0
