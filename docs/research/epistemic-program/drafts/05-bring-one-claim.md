# Draft 05 — Bring one claim that matters

**Surface:** the invitation. **Points at:** claim intake.

---

Bring one sentence your team says in public and cannot fully defend in private.

A benchmark number. A security property. A "proven". A "guaranteed". A 95%.

You do not get back `TRUE` or `FALSE`. A verdict would be the same error this
whole thing is about — a narrow analysis inflated into a broad judgment. You get
back the claim taken apart into the pieces that decide whether it holds, with
the unresolved pieces left visibly unresolved.

---

Take a real one:

> Our agent catches 95% of policy violations.

**Claim.** 95% of *what*? Caught out of violations that occurred, that were
labelled, or that the agent was shown? Is that recall, precision, or accuracy?
Most claims lose half their apparent strength here, before any evidence is
examined.

**Population.** Measured over which distribution, and what does it not
represent? Adversarial or organic traffic? Which policies, which locales, which
model version? A number from a curated benchmark and a number from production
share a digit and nothing else.

**Evidence.** Which artifact holds the measurement, who produced it, when, under
what code version, and can it be re-derived? An unreproducible number isn't
disqualified — it's relabelled as a report of a past observation rather than a
property of the system.

**Alternative explanations.** Could leakage explain it? Could selection explain
it — failures filtered before counting, an evaluation window chosen after seeing
results? Could the metric explain it — a class imbalance that makes 95% the
score of a system that always says "allow"?

**Smallest challenge.** Not the best study. The cheapest experiment whose
negative result would actually change what you say in public.

**Result.** Supported, contradicted, or inconclusive — at the scope defined in
step one. A process that never returns inconclusive isn't measuring anything.

**Boundary.** What larger claim remains unestablished, stated in the same
document, at the same size as the result.

---

What comes back: a claim map, an evidence trail, ranked alternative
explanations, one smallest useful test, and a limitation statement in plain
English that you can publish as written.

What does not come back: validation, compliance, a safety assertion, a badge, or
a score. If what you need is a badge, this is the wrong studio, and I'd rather
say so in the first email.

`docs/research/epistemic-program/claim-intake.md`
