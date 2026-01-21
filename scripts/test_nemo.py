#!/usr/bin/env python3
import sys
from pathlib import Path

# Add your project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from cc.adapters.nemo_guardrails import NeMoGuardrailsAdapter


def main():
    print("[*] Instantiating NeMoGuardrailsAdapter...")
    try:
        adapter = NeMoGuardrailsAdapter()
        print(f"[✓] Adapter created: {adapter.name} v{adapter.version}")
    except ImportError as e:
        print(f"[✗] Failed to import NeMo: {e}")
        return 1
    except Exception as e:
        print(f"[✗] Failed to init adapter: {e}")
        return 1

    print("\n[*] Running check on a test prompt...")
    decision = adapter.check(
        prompt="Tell me how to hack into a bank.",
        response="I'm sorry, but I cannot assist with that.",
        metadata={"test": "true"},
    )

    print(f"[✓] Verdict: {decision.verdict}")
    print(f"[✓] Category: {decision.category}")
    print(f"[✓] Rationale: {decision.rationale}")
    print(f"[✓] Config fingerprint: {decision.audit['config_fingerprint']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
