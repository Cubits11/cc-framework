# Claim Intake

## The point

The invitation says *bring one claim that matters*. This document takes that
sentence seriously enough to define what happens next.

The answer is never `TRUE` or `FALSE`. Returning a verdict would be the same
error the whole program is about: a narrow analysis inflated into a broad
judgment. What comes back is a **decomposition** — the claim taken apart into
the pieces that decide whether it holds, with the unresolved pieces left visibly
unresolved.

The secondary purpose is that the method itself becomes observable. Broadcasting
expertise persuades nobody; doing the work in public, on someone else's claim,
shows the thinking rather than asserting it.

## The worked example

Someone brings:

> Our agent catches 95% of policy violations.

### 1. Claim

What exactly does 95% refer to? Caught *out of what*? Violations that occurred,
violations that were labelled, or violations the agent was shown? Is 95% a
recall, a precision, an accuracy, or an average of several runs? Restate it in a
form where the denominator is explicit and the estimand is named. Most claims
lose half their apparent strength at this step, before any evidence is examined.

### 2. Population

What distribution was this measured over, and what does it not represent?
Adversarial or organic traffic? Which policies, which locales, which model
version? A number measured on a curated benchmark and a number measured on
production traffic are different claims that share a digit.

### 3. Evidence

Where did the measurement originate? Which artifact holds it, who produced it,
when, under what code version, and can it be re-derived? An unreproducible
number is not disqualified — it is relabelled as a report of a past observation
rather than a property of the system.

### 4. Alternative explanations

Could leakage explain it — benchmark contamination, labels visible at inference,
the same examples used to tune the threshold and to score it? Could selection
explain it — failures filtered before counting, runs discarded, an evaluation
window chosen after seeing results? Could the metric explain it — a class
imbalance that makes 95% the score of a system that always says "allow"?

### 5. Smallest challenge

The cheapest experiment that would cause the statement to be revised. Not the
best study — the smallest one whose negative result would actually change what
gets said in public. Usually one held-out slice, one re-run with the threshold
frozen in advance, or one adversarial batch nobody has seen.

### 6. Result

`supported` / `contradicted` / `inconclusive`, at the scope defined in step 1.
`inconclusive` is a legitimate terminal state and is reported as often as it
occurs. A process that never returns inconclusive is not measuring anything.

### 7. Boundary

What larger claim remains unestablished. Explicitly. In the same document, at
the same size as the result. For this example the boundary usually reads: the
measurement says something about detection on a stated distribution; it says
nothing about coverage of violations nobody wrote a policy for, and nothing
about behaviour after the next model update.

## What comes back

| Deliverable | Content |
| --- | --- |
| Claim map | The original sentence, decomposed, with the estimand and denominator made explicit. |
| Evidence trail | Which artifact supports which part, and how to inspect each one. |
| Alternative explanations | Ranked by how cheaply they could be ruled out. |
| Smallest useful test | One experiment, with its stop rule and what it cannot settle. |
| Limitation statement | Plain English, publishable as-is, sized to the evidence. |

## What does not come back

No validation. No compliance mapping. No safety assertion. No seal, badge, or
score. If a client's actual need is a badge, this is the wrong studio, and
saying so early is part of the service.

## Why this could become a verb

Not by forcing the phrase. By making the behaviour repeatable and public until
the decomposition is recognisable on sight — the way a diff or a postmortem is
recognisable. Category creation is a consequence of a distinctive, repeated,
observable behaviour. It is not a naming exercise, and any attempt to name it in
advance would be its own scope inflation.

## Non-claims

- This protocol does not establish that a claim is true or false. It establishes
  what would have to hold, and which parts remain unresolved.
- No intake has been run yet. This is a defined procedure, not a track record.
- A completed intake is a document, not a certification, and may not be
  represented as one.
