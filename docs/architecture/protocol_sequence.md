# Protocol Sequence

This document sketches the information flow and protocol used in the
cc‑framework.

## Sequence Diagram

```mermaid
sequenceDiagram
    participant A as Attacker
    participant B as World 0 (baseline)
    participant C as World 1 (guardrails)

    A->>B: send query
    B-->>A: baseline response
    A->>C: send query
    C-->>A: guarded response
    A->>A: update strategy
```

## Module Interaction

```mermaid
flowchart LR
    subgraph Experiment Runner
        R[run_two_world]
    end
    subgraph Guardrails
        G1[KeywordBlocker]
        G2[SemanticFilter]
    end
    A[Attacker]
    M[Metrics]

    R --> A
    R --> G1
    R --> G2
    R --> M
```

## Protocol Notes

* Sessions alternate between baseline and composed worlds.
* The attacker receives both responses to compute distinguishing power.
* Metrics (e.g., `cc_max`) summarise attacker advantage across sessions.

For implementation details see the [Developer Manual](../index.md).

For artifact storage guarantees, see [Storage Layout](storage_layout.md).
