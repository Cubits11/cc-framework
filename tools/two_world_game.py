# tools/two_world_game.py
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


class TwoWorldGame:
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
        self.n_queries = max(1, int(n_queries))
        self.bot = bool(bot)
        self.epsilon = clamp01(float(epsilon))
        self.strategy = strategy
        self.leaky_timing = bool(leaky_timing)
        if seed is not None:
            random.seed(int(seed))

        if difficulty == "easy":
            p0, p1 = 0.25, 0.80
        elif difficulty == "hard":
            p0, p1 = 0.35, 0.55
        elif difficulty == "normal":
            p0, p1 = 0.30, 0.70
        else:
            raise ValueError("Unknown difficulty: choose easy|normal|hard")

        p_benign = 0.08
        self.world_configs: Dict[int, WorldConfig] = {
            0: WorldConfig(p_block_benign=p_benign, p_block_attack=p0),
            1: WorldConfig(p_block_benign=p_benign, p_block_attack=p1),
        }
        if self.leaky_timing:
            self.world_configs[1].extra_delay_ms = 12.0

        # Hide the mapping and the secret
        self.secret_world = random.choice([0, 1])
        self.label_map: Dict[str, int] = (
            {"A": 0, "B": 1} if random.choice([True, False]) else {"A": 1, "B": 0}
        )

        # Wald’s SPRT thresholds
        sprt_alpha = clamp01(sprt_alpha)
        sprt_beta = clamp01(sprt_beta)
        # guard against degenerate logs
        eps = 1e-12
        self.sprt_A = math.log(max(eps, (1.0 - sprt_beta) / max(eps, sprt_alpha)))
        self.sprt_B = math.log(max(eps, sprt_beta / max(eps, (1.0 - sprt_alpha))))
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
                csv.writer(f).writerow(["round", "query_type", "response_A", "response_B", "logLR", "posterior_H0"])

    def _latency_ms(self, cfg: WorldConfig) -> int:
        mu = cfg.mean_latency_ms + cfg.extra_delay_ms
        sigma = max(0.1, cfg.std_latency_ms)
        val = max(1.0, random.gauss(mu, sigma))
        return int(round(val))

    def _query(self, world_idx: int, qtype: str) -> Tuple[str, int]:
        cfg = self.world_configs[world_idx]
        blocked = random.random() < cfg.p_block(qtype)
        latency = self._latency_ms(cfg)
        return ("Blocked" if blocked else "Allowed", latency)

    def _p_outcome(self, label: str, qtype: str, resp: str, hypothesis: str) -> float:
        # H0: A→W0,B→W1 ; H1: A→W1,B→W0
        if (hypothesis == "H0" and label == "A") or (hypothesis == "H1" and label == "B"):
            world_idx = 0
        else:
            world_idx = 1
        cfg = self.world_configs[world_idx]
        p_block = cfg.p_block(qtype)
        return p_block if resp == "Blocked" else (1.0 - p_block)

    def _update_log_lr(self, qtype: str, respA: str, respB: str) -> None:
        eps = 1e-12
        pA_H0 = max(eps, self._p_outcome("A", qtype, respA, "H0"))
        pB_H0 = max(eps, self._p_outcome("B", qtype, respB, "H0"))
        pA_H1 = max(eps, self._p_outcome("A", qtype, respA, "H1"))
        pB_H1 = max(eps, self._p_outcome("B", qtype, respB, "H1"))
        self.log_lr += math.log((pA_H0 * pB_H0) / (pA_H1 * pB_H1))

    def posterior_H0(self) -> float:
        # logistic(logLR)
        try:
            return 1.0 / (1.0 + math.exp(-self.log_lr))
        except OverflowError:
            return 1.0 if self.log_lr > 0 else 0.0

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
        # Expected KL(H0||H1) over joint outcomes for next query type
        def expected_llr(qtype: str) -> float:
            llr = 0.0
            for respA in ("Allowed", "Blocked"):
                for respB in ("Allowed", "Blocked"):
                    pH0 = self._p_outcome("A", qtype, respA, "H0") * self._p_outcome("B", qtype, respB, "H0")
                    pH1 = self._p_outcome("A", qtype, respA, "H1") * self._p_outcome("B", qtype, respB, "H1")
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
                    print("\nInput interrupted. Exiting.")
                    sys.exit(1)
                if q in {"benign", "attack"}:
                    return q
                print("Please type 'benign' or 'attack'.")
        return self._epsilon_policy() if self.strategy == "epsilon" else self._info_policy()

    def play(self, online: bool = False) -> None:
        print("\n=== Two-World Attack Game ===")
        print("Hypotheses: H0 (A=W0,B=W1) vs H1 (A=W1,B=W0).\n")
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
                    csv.writer(f).writerow([i, qtype, respA, respB, f"{self.log_lr:.6f}", f"{self.posterior_H0():.6f}"])

            if online:
                print(f"  logLR = {self.log_lr:+.3f}, posterior(H0) = {self.posterior_H0():.3f}")

            if self.log_lr >= self.sprt_A:
                print("\nSPRT decision: Favor H0 → predict A=W0 (safe), B=W1 (secret).")
                break
            if self.log_lr <= self.sprt_B:
                print("\nSPRT decision: Favor H1 → predict A=W1 (secret), B=W0 (safe).")
                break
            print()

        if self.sprt_B < self.log_lr < self.sprt_A and not self.bot:
            while True:
                try:
                    guess = input("Final guess ('A' or 'B'): ").strip().upper()
                except EOFError:
                    print("\nInput interrupted. Exiting.")
                    sys.exit(1)
                if guess in {"A", "B"}:
                    break
            guessed_world = self.label_map[guess]
            print("\n✅ Correct!" if guessed_world == self.secret_world else "\n❌ Incorrect.")

        self.print_summary()

    def _attack_pairs(self) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for (_i, qtype, respA, respB) in self.history:
            if qtype != "attack":
                continue
            a = 1 if respA == "Allowed" else 0
            b = 1 if respB == "Allowed" else 0
            if self.label_map["A"] == 0:
                pairs.append((a, b))
            else:
                pairs.append((b, a))
        return pairs

    def _bootstrap_ci(
        self,
        pairs: Sequence[Tuple[int, int]],
        level: float = 0.95,
        n: int = 1000,
    ) -> Tuple[float, float, float, float, float]:
        if not pairs:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        m = len(pairs)
        p0 = sum(a for a, _ in pairs) / m
        p1 = sum(b for _, b in pairs) / m
        j_hat = p0 - p1
        B = 2000 if m >= 20 else 1000
        diffs = []
        for _ in range(B):
            s0 = s1 = 0
            for _ in range(m):
                a, b = pairs[random.randint(0, m - 1)]
                s0 += a
                s1 += b
            diffs.append((s0 / m) - (s1 / m))
        diffs.sort()
        lo = diffs[int(max(0, (1.0 - level) / 2.0) * (B - 1))]
        hi = diffs[int((1.0 - max(0, (1.0 - level) / 2.0)) * (B - 1))]
        return p0, p1, j_hat, lo, hi

    def print_summary(self) -> None:
        ca, cb = self.counts["A"], self.counts["B"]
        total_benign = ca["benign"] + cb["benign"]
        total_attack = ca["attack"] + cb["attack"]
        print("\n--- Summary ---")
        print(f"Benign queries: total={total_benign}")
        if total_benign:
            rA = ca["b_block"] / ca["benign"] if ca["benign"] else 0.0
            rB = cb["b_block"] / cb["benign"] if cb["benign"] else 0.0
            print(f"  A benign block rate: {rA:.2f}")
            print(f"  B benign block rate: {rB:.2f}  (parity ⇒ utility matched)")

        print(f"Attack queries: total={total_attack}")
        if total_attack:
            rA = ca["a_block"] / ca["attack"] if ca["attack"] else 0.0
            rB = cb["a_block"] / cb["attack"] if cb["attack"] else 0.0
            print(f"  A attack block rate: {rA:.2f}")
            print(f"  B attack block rate: {rB:.2f}  (secret world should block more)")

            p0, p1, j, lo, hi = self._bootstrap_ci(self._attack_pairs(), 0.95, 1000)
            print("\nDistinguishability (attack success):")
            print(f"  p0 = P(Allowed | W0) = {p0:.3f}")
            print(f"  p1 = P(Allowed | W1) = {p1:.3f}")
            print(f"  Ĵ = p0 − p1 = {j:.3f}  (95% CI: {lo:.3f}, {hi:.3f})")

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-world attack game)")
    p.add_argument("-n","--rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--difficulty", choices=["easy","normal","hard"], default="normal")
    p.add_argument("--leaky-timing", action="store_true")
    p.add_argument("--bot", action="store_true")
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--strategy", choices=["epsilon","info"], default="info")
    p.add_argument("--online", action="store_true")
    p.add_argument("--sprt-alpha", type=float, default=0.05)
    p.add_argument("--sprt-beta", type=float, default=0.05)
    p.add_argument("--log", type=str, default=None)
    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    a = parse_args(argv)
    game = TwoWorldGame(
        n_queries=a.rounds,
        seed=a.seed,
        difficulty=a.difficulty,
        leaky_timing=a.leaky_timing,
        bot=a.bot,
        epsilon=a.epsilon,
        strategy=a.strategy,
        sprt_alpha=a.sprt_alpha,
        sprt_beta=a.sprt_beta,
        log_path=a.log,
    )
    game.play(online=a.online)

if __name__ == "__main__":
    main(sys.argv[1:])
