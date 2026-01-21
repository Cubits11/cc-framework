#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-World Attack Game — PhD Edition
==================================

This version of the two-world attack game extends the "Pro" version by
incorporating statistical hypothesis testing and decision theory to
illustrate research-grade techniques for measuring guardrail leakage.
Specifically, it implements:

  * A sequential probability ratio test (SPRT) to decide which label
    corresponds to the secret world with bounded error probabilities.
  * Online Bayesian inference and likelihood ratios printed after each
    round (optional).
  * An information-gain policy that chooses the next query type to
    maximise expected log-likelihood ratio gain, contrasting with
    simple ε-greedy exploration.
  * End-of-game summary including Youden’s Ĵ statistic and a 95 %
    bootstrap confidence interval for attack success rates.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class WorldConfig:
    p_block_benign: float
    p_block_attack: float
    mean_latency_ms: float = 80.0
    std_latency_ms: float = 8.0
    extra_delay_ms: float = 0.0

    def p_block(self, qtype: str) -> float:
        return self.p_block_benign if qtype == "benign" else self.p_block_attack


class TwoWorldGamePhD:
    def __init__(
        self,
        n_queries: int = 20,
        seed: Optional[int] = None,
        difficulty: str = "normal",
        leaky_timing: bool = False,
        bot: bool = False,
        epsilon: float = 0.15,
        strategy: str = "info",
        sprt_alpha: float = 0.05,
        sprt_beta: float = 0.05,
        log_path: Optional[str] = None,
    ) -> None:
        self.n_queries = max(1, n_queries)
        self.bot = bot
        self.epsilon = max(0.0, min(1.0, epsilon))
        self.strategy = strategy
        self.leaky_timing = leaky_timing
        if seed is not None:
            random.seed(seed)

        if difficulty == "easy":
            p_block_attack_0, p_block_attack_1 = 0.25, 0.80
        elif difficulty == "hard":
            p_block_attack_0, p_block_attack_1 = 0.35, 0.55
        elif difficulty == "normal":
            p_block_attack_0, p_block_attack_1 = 0.30, 0.70
        else:
            raise ValueError("Unknown difficulty level. Choose easy, normal, or hard.")

        p_block_benign = 0.08
        self.world_configs: Dict[int, WorldConfig] = {
            0: WorldConfig(p_block_benign=p_block_benign, p_block_attack=p_block_attack_0),
            1: WorldConfig(p_block_benign=p_block_benign, p_block_attack=p_block_attack_1),
        }
        if leaky_timing:
            self.world_configs[1].extra_delay_ms = 12.0

        self.secret_world = random.choice([0, 1])
        self.label_map: Dict[str, int] = (
            {"A": 0, "B": 1} if random.choice([True, False]) else {"A": 1, "B": 0}
        )

        self.sprt_A = math.log((1.0 - sprt_beta) / sprt_alpha)
        self.sprt_B = math.log(sprt_beta / (1.0 - sprt_alpha))
        self.log_lr = 0.0
        self.history: List[Tuple[int, str, str, str]] = []
        self.counts = {
            "A": {"benign": 0, "attack": 0, "b_block": 0, "a_block": 0},
            "B": {"benign": 0, "attack": 0, "b_block": 0, "a_block": 0},
        }

        self.log_path = log_path
        if log_path:
            log_dir = os.path.dirname(log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["round", "query_type", "response_A", "response_B", "logLR", "posterior_H0"]
                )

    def _latency_ms(self, cfg: WorldConfig) -> int:
        mu = cfg.mean_latency_ms + cfg.extra_delay_ms
        sigma = max(0.1, cfg.std_latency_ms)
        return int(max(1.0, random.gauss(mu, sigma)))

    def _query(self, world_idx: int, qtype: str) -> Tuple[str, int]:
        cfg = self.world_configs[world_idx]
        blocked = random.random() < cfg.p_block(qtype)
        latency = self._latency_ms(cfg)
        return ("Blocked" if blocked else "Allowed", latency)

    def _p_outcome(self, label: str, qtype: str, resp: str, hypothesis: str) -> float:
        if (hypothesis == "H0" and label == "A") or (hypothesis == "H1" and label == "B"):
            world_idx = 0
        else:
            world_idx = 1
        cfg = self.world_configs[world_idx]
        p_block = cfg.p_block(qtype)
        return p_block if resp == "Blocked" else (1.0 - p_block)

    def _update_log_lr(self, qtype: str, respA: str, respB: str) -> None:
        pA_H0 = clamp01(self._p_outcome("A", qtype, respA, "H0"))
        pB_H0 = clamp01(self._p_outcome("B", qtype, respB, "H0"))
        pA_H1 = clamp01(self._p_outcome("A", qtype, respA, "H1"))
        pB_H1 = clamp01(self._p_outcome("B", qtype, respB, "H1"))
        numerator = max(1e-12, pA_H0 * pB_H0)
        denominator = max(1e-12, pA_H1 * pB_H1)
        self.log_lr += math.log(numerator / denominator)

    def posterior_H0(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.log_lr))

    def _tally(self, label: str, qtype: str, resp: str) -> None:
        c = self.counts[label]
        c[qtype] += 1
        if resp == "Blocked":
            if qtype == "benign":
                c["b_block"] += 1
            else:
                c["a_block"] += 1

    def _epsilon_policy(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(["benign", "attack"])

        def diff_rate(qtype: str) -> float:
            ca, cb = self.counts["A"], self.counts["B"]
            if qtype == "benign":
                ra = ca["b_block"] / ca["benign"] if ca["benign"] else 0.0
                rb = cb["b_block"] / cb["benign"] if cb["benign"] else 0.0
            else:
                ra = ca["a_block"] / ca["attack"] if ca["attack"] else 0.0
                rb = cb["a_block"] / cb["attack"] if cb["attack"] else 0.0
            return abs(ra - rb)

        return "attack" if diff_rate("attack") >= diff_rate("benign") else "benign"

    def _info_policy(self) -> str:
        def expected_llr(qtype: str) -> float:
            llr = 0.0
            for respA in ("Allowed", "Blocked"):
                for respB in ("Allowed", "Blocked"):
                    pH0 = self._p_outcome("A", qtype, respA, "H0") * self._p_outcome(
                        "B", qtype, respB, "H0"
                    )
                    pH1 = self._p_outcome("A", qtype, respA, "H1") * self._p_outcome(
                        "B", qtype, respB, "H1"
                    )
                    if pH0 > 0.0 and pH1 > 0.0:
                        llr += pH0 * math.log(pH0 / pH1)
            return llr

        return "attack" if expected_llr("attack") >= expected_llr("benign") else "benign"

    def choose_query(self) -> str:
        if not self.bot:
            while True:
                try:
                    q = input("Enter query type ('benign' or 'attack'): ").strip().lower()
                except EOFError:
                    print("\nInput interrupted. Exiting game.")
                    sys.exit(1)
                if q in {"benign", "attack"}:
                    return q
                print("Please type 'benign' or 'attack'.")
        if self.strategy == "epsilon":
            return self._epsilon_policy()
        elif self.strategy == "info":
            return self._info_policy()
        else:
            raise ValueError("Unknown strategy. Choose 'epsilon' or 'info'.")

    def play(self, online: bool = False) -> None:
        print("\n=== Two-World Attack Game — PhD Edition ===")
        print("Hypotheses: H₀ (A=W0, B=W1) vs H₁ (A=W1, B=W0).\n")
        for i in range(1, self.n_queries + 1):
            qtype = self.choose_query()
            respA, latA = self._query(self.label_map["A"], qtype)
            respB, latB = self._query(self.label_map["B"], qtype)

            self._tally("A", qtype, respA)
            self._tally("B", qtype, respB)
            self.history.append((i, qtype, respA, respB))
            self._update_log_lr(qtype, respA, respB)

            if self.leaky_timing:
                print(f"Round {i}: {qtype} | A: {respA} ({latA} ms), B: {respB} ({latB} ms)")
            else:
                print(f"Round {i}: {qtype} | A: {respA}, B: {respB}")

            if self.log_path:
                with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [i, qtype, respA, respB, f"{self.log_lr:.6f}", f"{self.posterior_H0():.6f}"]
                    )

            if online:
                print(f"  logLR = {self.log_lr:+.3f}, posterior(H₀) = {self.posterior_H0():.3f}")

            if self.log_lr >= self.sprt_A:
                print(
                    "\nSPRT decision: Evidence favors H₀ → predict A=W0 (safe) and B=W1 (secret)."
                )
                break
            if self.log_lr <= self.sprt_B:
                print(
                    "\nSPRT decision: Evidence favors H₁ → predict A=W1 (secret) and B=W0 (safe)."
                )
                break
            print()

        if self.sprt_B < self.log_lr < self.sprt_A and not self.bot:
            while True:
                guess = (
                    input("Final guess — which hides the secret? ('A' or 'B'): ").strip().upper()
                )
                if guess in {"A", "B"}:
                    break
            guessed_world = self.label_map[guess]
            if guessed_world == self.secret_world:
                print("\n✅ Correct! You identified the secret world.")
            else:
                print("\n❌ Incorrect. The other world was secret.")

        self.print_summary()

    def _attack_pairs(self) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for _i, qtype, respA, respB in self.history:
            if qtype != "attack":
                continue
            outcomeA = 1 if respA == "Allowed" else 0
            outcomeB = 1 if respB == "Allowed" else 0
            if self.label_map["A"] == 0:
                pairs.append((outcomeA, outcomeB))
            else:
                pairs.append((outcomeB, outcomeA))
        return pairs

    def _bootstrap_ci(
        self, pairs: Sequence[Tuple[int, int]], level: float = 0.95, n: int = 1000
    ) -> Tuple[float, float, float, float, float]:
        if not pairs:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        n_pairs = len(pairs)
        p0 = sum(p[0] for p in pairs) / n_pairs
        p1 = sum(p[1] for p in pairs) / n_pairs
        j_hat = p0 - p1
        diffs: List[float] = []
        for _ in range(n):
            s0 = s1 = 0
            for _ in range(n_pairs):
                a, b = pairs[random.randint(0, n_pairs - 1)]
                s0 += a
                s1 += b
            diffs.append((s0 / n_pairs) - (s1 / n_pairs))
        diffs.sort()
        lower = (1.0 - level) / 2.0
        upper = 1.0 - lower
        li = max(0, int(lower * len(diffs)))
        ui = min(len(diffs) - 1, int(upper * len(diffs)))
        return p0, p1, j_hat, diffs[li], diffs[ui]

    def print_summary(self) -> None:
        ca, cb = self.counts["A"], self.counts["B"]
        total_benign = ca["benign"] + cb["benign"]
        total_attack = ca["attack"] + cb["attack"]
        print("\n--- Summary ---")
        print(f"Benign queries: total={total_benign}")
        if total_benign:
            rate_benign_a = ca["b_block"] / ca["benign"] if ca["benign"] else 0.0
            rate_benign_b = cb["b_block"] / cb["benign"] if cb["benign"] else 0.0
            print(f"  A benign block rate: {rate_benign_a:.2f}")
            print(f"  B benign block rate: {rate_benign_b:.2f}")
            print("  (Benign rates should be similar if utility is matched.)")

        print(f"Attack queries: total={total_attack}")
        if total_attack:
            rate_attack_a = ca["a_block"] / ca["attack"] if ca["attack"] else 0.0
            rate_attack_b = cb["a_block"] / cb["attack"] if cb["attack"] else 0.0
            print(f"  A attack block rate: {rate_attack_a:.2f}")
            print(f"  B attack block rate: {rate_attack_b:.2f}")
            print("  (Expect higher block rate in the secret world.)")

            p0, p1, j_hat, lo, hi = self._bootstrap_ci(self._attack_pairs(), level=0.95, n=1000)
            print("\nDistinguishability analysis (attack success):")
            print(f"  p₀ = P(Allowed | W0) = {p0:.3f}")
            print(f"  p₁ = P(Allowed | W1) = {p1:.3f}")
            print(f"  Ĵ = p₀ − p₁ = {j_hat:.3f}")
            print(f"  95 % bootstrap CI: ({lo:.3f}, {hi:.3f})")
            print("  (Positive Ĵ ⇒ safe world allows more attacks; CI quantifies uncertainty.)")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play the PhD-level two-world attack game with SPRT and information design"
    )
    parser.add_argument(
        "-n", "--rounds", type=int, default=20, help="Maximum number of query rounds"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "normal", "hard"],
        default="normal",
        help="Attack block gap preset",
    )
    parser.add_argument(
        "--leaky-timing", action="store_true", help="Introduce timing side-channel in secret world"
    )
    parser.add_argument("--bot", action="store_true", help="Use an automated attacker")
    parser.add_argument(
        "--epsilon", type=float, default=0.15, help="ε for ε-greedy when strategy='epsilon'"
    )
    parser.add_argument(
        "--strategy", choices=["epsilon", "info"], default="info", help="Query selection strategy"
    )
    parser.add_argument(
        "--online", action="store_true", help="Print logLR and posterior after each round"
    )
    parser.add_argument("--sprt-alpha", type=float, default=0.05, help="Type I error (α) for SPRT")
    parser.add_argument("--sprt-beta", type=float, default=0.05, help="Type II error (β) for SPRT")
    parser.add_argument("--log", type=str, default=None, help="CSV log path (optional)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    game = TwoWorldGamePhD(
        n_queries=args.rounds,
        seed=args.seed,
        difficulty=args.difficulty,
        leaky_timing=args.leaky_timing,
        bot=args.bot,
        epsilon=args.epsilon,
        strategy=args.strategy,
        sprt_alpha=args["sprt_alpha"] if isinstance(args, dict) else args.sprt_alpha,
        sprt_beta=args["sprt_beta"] if isinstance(args, dict) else args.sprt_beta,
        log_path=args.log,
    )
    game.play(online=args.online)


if __name__ == "__main__":
    main(sys.argv[1:])
