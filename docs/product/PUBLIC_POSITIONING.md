# Public Positioning

## Status

Positioning and communication guidance. This document is a C0 repository/strategy
artifact. It creates no claim, promotes no surface into Paper Core, and is not
evidence for anything. Where it describes offers or a schedule, those are
intentions, not commitments already met.

## The idea

```text
Before a green check becomes a claim, make its assumptions, evidence,
and limits inspectable.
```

That is not safety branding. It is a public method for turning impressive
technical promises into questions that can genuinely fail.

## The category

Not a generic security company. A **public laboratory for doubt**.

The advantage is not "we have the answer." The advantage is **"we make the
answer show its work."**

## Lockup and voice

```text
Pranav Bhave / Cubits11
Make the claim smaller. Make the evidence stronger.
```

Flagship:

```text
A story can start a question.
Evidence must finish the answer.

Cubits11
Keep the wonder. Check the claim.
```

Bio:

```text
I build research software and public experiences that turn vague technical
claims into inspectable evidence.
```

## Name boundaries

| Name | Public role | Boundary |
| --- | --- | --- |
| **Pranav Bhave** | The author, builder, and speaker. | The human voice. |
| **Cubits11** | Independent studio for evidence design, research software, and education. | The commercial home. Sells clarity, never certainty. |
| **CC-Framework** | Research on what stacked guardrail evidence supports when failures may be correlated. | Not a safety score and not a certification. |
| **Ghost-Ark** | A separate institutional research artifact on bounded receipts and verifiable evidence. | Not a Cubits11 product and not a university endorsement. |

**The Ghost-Ark rule is strict.** It stays out of Cubits11 sales material
entirely. Its own public-interface rules prohibit commercial planning and
unapproved institutional representation. Reference it as a clearly labelled
research case study or not at all. The film's `ghost-ark` cut carries no
Cubits11 mark for exactly this reason.

## The film is not an ad

The strongest thing in the 15-second film is not the notebook, the grid, or the
light. It is this:

> The boldest thing a technical brand can say is `INCONCLUSIVE`.

The film says it on purpose, and when it has no verifier output to show it
stamps `ILLUSTRATION` on its own footage rather than imply a result it was not
given. That behavior is enforced in the source, not in a style guide: see
`visual_identity/before_you_see_it/`.

## The release loop

```text
Film → curiosity → evidence page → interactive check → external scrutiny
  ↑                                                        ↓
  └────── next experiment, including failures and limits ──┘
```

The canonical page carries all five required elements: the film and its
transcript, a source ledger, a visible "what this does not establish" strip, one
real interactive check, and the invitation. It lives at
`visual_identity/canonical_page/index.html`.

**Never fake a verifier result.** If a screen is illustrative, label it on the
screen. If it is real, link the fixture, the command, and the limitation. The
canonical page's check recomputes SHA-256 in the visitor's browser over a real
capsule fixture and compares it against the digest recorded in
`examples/claim_governance_capsule/manifest.expected.json`. Nothing is
hard-coded, and the limitation is printed directly beneath it: integrity only.

## What the studio sells

Small, fixed-scope offers.

1. **Claim-to-Evidence Sprint.** A team brings one important product, research,
   or AI-system claim. They get back a claim map, evidence gaps, alternative
   explanations, the smallest useful test, and a plain-English limitation
   statement.
2. **Evidence Narrative Package.** Dense technical work becomes a film, an
   evidence page, a source trail, and a limitation section.
3. **Evidence Literacy Lab.** A workshop built on
   `story → claim → test → result → limitation → next decision`.
4. **Research-to-Product Evidence Design.** For teams with real technical work
   and no coherent way to show what is implemented, measured, assumed, or merely
   planned.

**Not sold:** validation, compliance, "safe AI", or a seal of approval. What is
sold is better questions, stronger evidence, clearer public communication, and
more useful failure boundaries.

Likely first customers: applied-AI founders, research groups, technical product
teams, security and evaluation teams, and public-interest organizations. Not
buyers shopping for a compliance badge.

## The ninety days

| Window | Work |
| --- | --- |
| **1-10 — establish the flag** | Publish the film and one canonical page. Publish the fixed bio and the method. Put one interactive artifact beside the film. Make authorship, sources, and limitations visible. |
| **11-30 — become useful in public** | Publish "an intuition can start a question; it cannot finish it." Release three short clips: *a signature is not truth*, *a test must be able to fail*, *inconclusive is a result*. Write five genuinely useful technical responses to people already working on evaluation, provenance, security, and reproducibility. No spam. |
| **31-60 — invite scrutiny** | Run a small live Claim Clinic: five to ten people, one claim each. Ask three independent technical people to try to reproduce or break one narrow artifact. Publish what they found, including the uncomfortable parts. |
| **61-90 — turn trust into work** | Offer the first paid Claim-to-Evidence Sprints. Publish one approved or anonymized lesson. Package repeated work into the workshop. Consider hosted software only after repeated demand for the same workflow. |

## What to measure

Independent reproductions. External issues, corrections, and contributions.
Invitations to teach or speak. Qualified technical conversations. Paid pilots.
Citations and references.

Views and followers are not evidence that anyone trusts the work. Do not confuse
the two, and do not report them as traction.

## Where scrutiny lives

- Communities where scrutiny is the culture, entered with a narrow artifact or a
  practical question rather than a pitch:
  [OpenSSF working groups](https://openssf.org/community/openssf-working-groups/),
  which include AI/ML security and supply-chain integrity.
- Provenance can eventually become part of the art itself. Use
  [C2PA](https://spec.c2pa.org/specifications/specifications/1.1/specs/C2PA_Specification.html)
  only to show signed origin and edit history, never to imply that a video is
  factually true. That distinction is the same one this whole method rests on.
- Model the research culture on artifact availability and reproducibility rather
  than prestige. The
  [USENIX Security artifact evaluation program](https://secartifacts.github.io/usenixsec2026/)
  is a useful benchmark for that mindset.

## The sentence that matters

Not "Pranav Bhave is brilliant." This one:

> "When Pranav Bhave publishes something, I can see what it claims, what it does
> not claim, and how to challenge it."

## Non-claims

- This document does not establish that the method works, that the studio has
  customers, or that any offer has been delivered.
- The ninety-day schedule is a plan. Publishing it is not evidence of executing
  it.
- Nothing here authorizes representing Ghost-Ark, Penn State, or any other
  institution commercially.
- No superlative in this document ("first", "only", "best") is licensed for
  public use. Any such wording requires a named evidentiary standard under
  `docs/claims/CLAIM_BOUNDARY_MANIFEST.md` before it ships.
