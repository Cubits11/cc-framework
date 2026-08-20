# cc-kernel-v1 — Normative Specification

A CC composition kernel bounds the probability of a composed binary event from
the marginal probabilities of its parts, under unknown dependence.

This document is **normative and self-contained**. An implementer works from
this document and [`cases/`](cases/), never from the Python source. That is the
point: an implementation written by reading `cc.compose` would inherit its
mistakes, and agreement between it and the reference would establish nothing.

> **What passing establishes.** That an implementation computes the same
> intervals as the reference on the cases in this corpus.
>
> **What passing does not establish.** That either implementation is correct —
> two implementations can share a misreading of this spec. That the corpus
> covers the input space; it is a curated census whose size is an authoring
> decision. That any bound means a system is safe, a threshold is right, or a
> marginal is valid.

---

## 1. Objects

### 1.1 Event

A named binary indicator. `1` denotes occurrence of the event being bounded —
by CC convention, a **guardrail failure or unsafe pass**, not a success.

An event is a name and a marginal probability. It has no threshold, no
operating curve, and no false-positive rate. A deterministic refusal rule and a
tuned classifier are the same kind of input here.

### 1.2 Marginals

A mapping from event name to probability.

- Names MUST be non-empty strings.
- Probabilities MUST be finite and in `[0, 1]`.
- The mapping MUST contain at least one entry.
- Order MUST NOT affect any output value. It MAY affect `binding_event` only
  through the tie rule in §4.3, which is order-independent by construction.

### 1.3 Event kind

| Canonical | Aliases | Meaning |
|---|---|---|
| `all` | `and`, `AND`, `intersection` | `P(every event occurs)` — a conjunction |
| `any` | `or`, `OR`, `union` | `P(at least one occurs)` — a union |

Any other value MUST be refused (§5, `unknown_event_kind`).

### 1.4 Dependence assumption

| Value | Meaning |
|---|---|
| `unconstrained` | Nothing assumed. Returns the sharp interval. The honest default. |
| `independent` | A point value, returned as a degenerate interval. |
| `comonotone` | The upper Fréchet corner, as a degenerate interval. |
| `countermonotone` | **Defined only for exactly two events.** See §6. |

Any other value MUST be refused (`unknown_dependence`). An implementation MUST
NOT default an unrecognized value to `unconstrained`.

---

## 2. The unconstrained interval

For marginals `p_1 … p_n`:

**Conjunction** (`event = "all"`):

```
lower = max(0, (Σ p_i) − (n − 1))
upper = min_i p_i
```

**Union** (`event = "any"`):

```
lower = max_i p_i
upper = min(1, Σ p_i)
```

These are the Fréchet–Hoeffding inequalities. The mathematics is classical —
Fréchet 1935, Hoeffding 1940 — and this specification claims no novelty for it.

Both endpoints are **pointwise sharp**: for any fixed marginals, some joint
distribution attains each. Neither can be tightened without dependence evidence.

Both endpoints MUST be clipped to `[0, 1]` after computation, to absorb
floating-point excursions at the boundary.

### 2.1 The finding this encodes

Under a conjunction the upper bound is `min_i p_i`, which does **not** depend on
`n`. A conjunction of controls is no stronger than its single strongest member.
Adding a further control cannot raise the upper bound and can only lower the
floor. Case family `cliff-homogeneous-m*` pins this: ten events at `p = 0.1`
still bound to `[0, 0.1]` while independence predicts `1e-10`.

---

## 3. The independence baseline

Computed and reported on **every** result, whatever the dependence assumption:

```
independence_point = Π p_i                       (conjunction)
independence_point = 1 − Π (1 − p_i)             (union)
```

This is a **comparison baseline, never an answer.** It exists so the gap it
understates stays visible. An implementation MUST report it even when
`dependence = "independent"`, where it coincides with the interval.

Derived quantities:

```
width                   = max(0, upper − lower)
independence_regret     = upper − independence_point
understatement_factor   = upper / independence_point,  undefined when the point is 0
```

`understatement_factor` MUST serialize as JSON `null` when
`independence_point` is zero. JSON has no infinity, and a null here means the
ratio is *undefined*, not large.

---

## 4. `binding_event`

The name of the event whose marginal the upper bound turns on, or `null`.

This field is the actionable part of the result: it names where effort belongs.

### 4.1 Conjunction

The upper bound is `min_i p_i`, so the argmin binds — improving any other event
moves the upper bound not at all.

### 4.2 Union

The upper bound is `min(1, Σ p_i)`, a sum rather than a single event. No event
binds, and `binding_event` MUST be `null` — **except** when exactly one event is
present, where it is that event.

### 4.3 Ties

If two or more events attain the minimum (within tolerance), `binding_event`
MUST be `null`. Improving either leaves the bound pinned by the other, so naming
one of them would misdirect effort.

### 4.4 Constrained regimes

When `dependence` is anything other than `unconstrained`, `binding_event` MUST
be `null`. The interval is then a stipulated point, not a bound turning on an
event.

---

## 5. Refusals

An implementation MUST refuse, not default, in each case below. Refusal reasons
are the identifiers used by `expect_refusal` in
[`cases/adversarial.json`](cases/adversarial.json).

| Reason | Condition |
|---|---|
| `marginal_out_of_range` | a marginal outside `[0, 1]` |
| `marginal_not_finite` | a marginal that is NaN or infinite |
| `no_events` | an empty marginals mapping |
| `unknown_event_kind` | an event kind not in §1.3 |
| `unknown_dependence` | a dependence value not in §1.4 |
| `countermonotone_undefined` | `countermonotone` with more than two events (§6) |

**JSON note.** JSON has no NaN literal. In `cases/adversarial.json` the
non-finite marginal is encoded as the **string** `"NaN"`. An implementation
running the corpus MUST map that string to its language's NaN before calling the
kernel — and MUST NOT accept a string as a probability in normal operation.

---

## 6. Countermonotonicity is bivariate

`dependence = "countermonotone"` with `n > 2` MUST be refused.

Countermonotonicity is a strictly bivariate concept. Two events can be perfectly
negatively dependent — one occurs exactly when the other does not — but three
cannot all be pairwise mutually exclusive and exhaustive in that way, and there
is no `n`-dimensional countermonotonic structure for `n > 2`.

The matching fact about the bound: the Fréchet–Hoeffding lower bound
`max(0, Σ p_i − (n−1))` **is not a copula in dimension ≥ 3**. It remains
*pointwise sharp* — for any fixed marginals some joint distribution attains it —
but no single dependence structure attains it everywhere. So there is no
countermonotone regime to tabulate beside independence and comonotonicity.

An implementation that returns a number here is wrong. Silently offering a
"countermonotone" option for four events produces a value with the form of a
dependence regime and none of the content.

For `n = 2`:

```
countermonotone value = max(0, p_1 + p_2 − 1)      (conjunction)
countermonotone value = min(1, p_1 + p_2)          (union)
```

---

## 7. Tolerance

Numeric comparisons use an absolute tolerance of **1e-12**, declared in
[`manifest.json`](manifest.json).

This sits far above the worst disagreement measured between the reference
closed form and the finite-atom LP (3.3e-16 over 600 randomized cases), and far
below any difference that would change a reported bound. An implementation
needing a looser tolerance to pass has a defect, not a rounding difference.

---

## 8. Running the corpus

```
cases/composition.json   accept cases — reproduce every field in `expect`
cases/adversarial.json   reject cases — refuse with the named reason
manifest.json            digests, tolerance, case count, non-claims
```

For each accept case: call the kernel with `input`, compare every key in
`expect` within tolerance. `binding_event` is compared exactly.

For each reject case: call the kernel with `input` and require a refusal. An
implementation SHOULD distinguish the reason; a corpus runner MAY accept any
refusal if the implementation does not expose typed reasons, but MUST record
that it did so.

A runner MUST report `pass/fail` counts as exact integers. **No confidence
interval may be attached to a pass rate over this corpus** — it is the whole
population, not a sample, and its size is an authoring decision.

---

## 9. Versioning

`cc-kernel-v1` is stable. Adding a case is a **new corpus version**, not an edit
to this one — a passing implementation must not start failing because the corpus
grew underneath it.

Fixing a case whose expected value was **wrong** is a defect correction, and
MUST be recorded in the corpus changelog with the reason. Regenerating the
corpus to match a changed implementation, without such a record, is the failure
mode this discipline exists to prevent.

**Never weaken a case to make an implementation pass.**
