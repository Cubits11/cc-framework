# Epistemic Prompt Library

## How to use these

These prompts produce research artifacts, not predictions. Replace bracketed
text before use. Require citations, source dates, uncertainty, and an explicit
non-claim in every answer.

Two standing constraints apply to every prompt below.

- A model's output is a draft artifact, not evidence. It enters the record the
  same way any other draft does: predeclared, reviewed, and bounded.
- If a prompt's answer would change a financial, legal, medical, safety, or
  otherwise consequential decision, the answer is insufficient on its own.

The prompts map onto the [eight-week plan](eight-week-plan.md) roughly in order:
1 for Week 1, 2 and 3 for Week 2, 4 and 5 for Week 3, 6 and 7 for Weeks 3 to 5,
8 for Week 4, 9 for Week 6, 10 for Week 8, and 11 whenever a prior statement
stops being supportable.

## 1. Narrative-to-study translator

```text
You are an epistemic editor. Analyze the following narrative without validating
its predictions or metaphysical claims:

[PASTE TEXT]

For every substantive statement, label it as:
REFLECTION, IDENTITY AFFIRMATION, FORECAST, CAUSAL ASSERTION,
ACTION HEURISTIC, EMPIRICAL CLAIM, or MARKETING FUNNEL.

Then produce:
1. a charitable interpretation;
2. a falsifiable translation, if one exists;
3. the minimum observation needed to test it;
4. one alternative explanation;
5. a non-claim;
6. whether using it for a financial, legal, medical, or consequential decision
   would be inappropriate.

Do not infer personal facts, future outcomes, or hidden causes.
```

## 2. Claim-to-proof boundary auditor

```text
You are a hostile technical reviewer. For each claim below, identify exactly
what the mechanism proves, what it assumes, and what it cannot establish.

[CLAIMS]
[ARCHITECTURE OR CODE LINKS]

Return a table:
claim | cryptographic/operational object | trust root | verifier input |
attack or failure mode | evidence currently present | evidence missing |
allowed wording | prohibited wording | explicit non-claim

Reject any leap from:
- app integrity to physical-world truth;
- enclave measurement to correct execution;
- signed assertion to assertion truth;
- provenance to authenticity;
- one successful run to universal reliability.
```

## 3. Prior-art destroyer

```text
Act as a prior-art researcher trying to falsify this proposed novelty claim:

[CLAIM]

Search primary sources, standards, vendor documentation, peer-reviewed papers,
public incident reports, and active products. Separate:
- already solved;
- partially solved;
- adjacent but materially different;
- commercially occupied;
- genuinely unresolved.

For every source, provide URL, publication date, source type, exact relation to
the claim, and a confidence rating. End with:
"what remains narrow enough to investigate honestly."

Do not use search-result snippets as evidence.
```

## 4. Experiment preregistration architect

```text
Turn this question into a preregistered experiment:

[QUESTION]
[DECISION IT MUST CHANGE]
[AVAILABLE DATA / ACCESS LIMITS]

Specify:
- hypothesis and falsifier;
- population and what it does not represent;
- sampling frame or census rationale;
- primary measure and denominator;
- control arm or alternative explanation;
- required positive-control/discriminator test;
- exclusions and unavailable-arm treatment;
- repeat count and stability plan;
- stop, scale, hold, and kill criteria;
- ethics/privacy requirements;
- exact non-claim.

Do not choose thresholds after viewing results.
```

## 5. Signal-versus-traction designer

```text
Convert this intuitive "signal" into a predeclared traction measure:

[OBSERVATION OR INTUITION]

Return:
1. the observable event;
2. the baseline or comparison;
3. the collection method;
4. the time window;
5. what would count as noise;
6. what result changes the decision;
7. the smallest reversible next action;
8. what must not be inferred.

Explicitly distinguish attention, inquiries, conversion, repeat behavior,
revenue, margin, and retention. Never substitute one for another.
```

## 6. Customer-evidence interviewer

```text
Design a neutral customer-discovery protocol for this decision:

[QUEUE / DELIVERY / GROUPS / ROUTES / PRICING / OTHER]

Create:
- recruitment criteria;
- non-leading interview questions;
- exact operational observations to collect;
- consent and privacy boundary;
- disconfirming answers to look for;
- how to avoid treating compliments as demand;
- a decision rule tied to the existing experiment registry;
- a summary template that separates direct quotes, observations, inference,
  and non-claim.

Do not promise services, ask for sensitive data, or create a marketing claim
from a single interview.
```

## 7. B2B consequence-search protocol

```text
Research whether this technical failure class has caused a real consequence:

[FAILURE CLASS]

Search CVEs, advisories, standards errata, incident postmortems, court filings,
and vendor security reports. For each candidate, classify:
- mechanism;
- affected representation or parser;
- attacker capability;
- observed consequence;
- evidence quality;
- remediation;
- relation to our corpus;
- whether the incident supports prevalence, possibility only, or neither.

Report null findings and search limitations. Do not turn a mechanism into a
market claim unless consequence evidence supports that step.
```

## 8. Instrument-sensitivity challenge

```text
You are validating a research instrument, not its preferred answer.

[INSTRUMENT]
[INTENDED DETECTION OR MEASUREMENT]

Propose:
- one deliberate defect the instrument must detect;
- one negative control it must not flag;
- one boundary case;
- one unavailable-arm condition;
- one mutation that should change the result;
- one mutation that should not;
- a pass/fail criterion for each;
- the consequence if the instrument fails any test.

Explain why a green result without these controls would be uninformative.
```

## 9. Independent replay handoff

```text
Create a handoff for an independent reviewer to reproduce this result:

[CLAIM]
[ARTIFACTS]
[COMMANDS]
[ENVIRONMENT]

Include:
- exact inputs and hashes;
- expected output and failing output;
- tool versions;
- trust roots and keys that may be public;
- no-secret setup;
- likely failure modes;
- how to report disagreement;
- what agreement would and would not establish.

The handoff must allow the reviewer to disagree safely and visibly.
```

## 10. Evidence-to-decision board

```text
Given this evidence packet:

[PACKET]

Make a four-way decision memo:
CONTINUE, HOLD, NARROW, or STOP.

For each option, state:
- facts supporting it;
- facts against it;
- assumptions still carrying the conclusion;
- the next cheapest high-information test;
- irreversible actions that must wait;
- claims that must be removed or downgraded now;
- the condition that would reverse the recommendation.

Do not reward activity, novelty, or optimism. Reward only decision-relevant
evidence.
```

## 11. Retraction and revision writer

```text
An earlier statement is no longer supportable:

[OLD STATEMENT]
[NEW EVIDENCE]

Write:
1. a precise retraction or downgrade;
2. what remains true;
3. what was not measured;
4. the artifact or source revision that changed the conclusion;
5. the new allowed wording;
6. the prohibited wording;
7. the next experiment, if any.

Never silently delete the old claim or rewrite history as if it was never made.
```

## Non-claims

- These prompts do not validate their own output. A completed table is a draft,
  not a finding.
- A model asked to search may return plausible sources that do not exist or do
  not say what the summary says. Every citation is checked against the primary
  source before it enters an artifact.
- Prompt 1 classifies statements; it does not determine whether a narrative
  source is meaningful to the person who brought it.
- Prompts 6 and 7 describe protocols. Running them against real people or
  external systems requires separate explicit authorization and, for human
  contact, appropriate consent.
