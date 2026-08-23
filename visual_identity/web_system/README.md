# Web system — the observatory grammar for evidence surfaces

Status: documented port · 2026-08-23. This directory records the shared
visual grammar used by CC-Framework's web-facing surfaces and by
[cubits11.github.io](https://cubits11.github.io/) (whose
`assets/site.css` is the canonical running instance). The two repositories
stay loosely coupled on purpose: this document is the contract, not a
shared build.

**Boundary.** Visual components consume kernel outputs; they never
re-implement the kernel. Nothing in this directory defines mathematical
semantics, and nothing here is evidence for any claim — the theorem
ledger and validation matrix carry those.

## 1 · Semantic color law

Inherited from `visual_identity/claim_observatory/` ("Cyan/white light
marks evidence and replayable structure. Amber marks required review. Red
is reserved for expiration/invalidation.") and from the film color law in
`visual_identity/before_you_see_it/DIRECTORS_CUT.md`.

| Token | Meaning | Dark | Light |
| --- | --- | --- | --- |
| field / ink | ground and readable truth claims | `#0B0F0A` / `#EDE8DA` | `#F1EDE2` / `#14170F` |
| `--evidence` | public evidence, replayable structure, supported bindings | `#7FC4CF` | `#175F6B` |
| `--review` | review required, assumptions doing work, freshness warnings | `#E9A23B` | `#7E4E12` |
| `--invalid` | contradiction, invalidation, expiry, hard refusal | `#E4796F` | `#993127` |
| `--muted` | unknown, inactive, context | `#9AA391` | `#565E4E` |
| `--gold` | identity accent only — never a state | `#C9A15E` | `#755A2C` |

Rules:

- **Green carries no semantic duty.** Never a green "safe" badge — the
  claim-observatory world bible forbids it, and so does this system.
- **No state is encoded by color alone.** Every colored state pairs with a
  label, a shape (solid dot / hollow dashed dot / square), or both.
- **Red is earned.** Expiry, contradiction, falsification — never
  decoration, never urgency theater.
- Every foreground/background pair ships with a computed WCAG ratio
  (≥ 4.5:1); the site's DESIGN.md records the current numbers.

## 2 · Typography

Editorial serif for propositions and questions; restrained grotesk for
explanation; mono **only** for artifact IDs, revisions, hashes, commands,
and metrics — the mono layer is the instrument-marking voice, and diluting
it with decorative use destroys its meaning. Self-hosted fonts only; no
third-party requests from page code.

## 3 · Epistemic primitives

| Primitive | Encodes | Never |
| --- | --- | --- |
| Evidence chip (solid `--evidence` dot) | link opens a public artifact | used on unlinked text |
| Attested chip (dashed, hollow dot) | owner-attested, dated, no public artifact | dressed up as verification |
| Claim envelope (8 fields) | proposition · scope · support · challenge · test design · status · boundary · freshness | rendered partially filled without saying which fields are absent |
| Status pill | one claim's evidential status | aggregated into a health score |
| Stamp (rotated mono) | a verdict that was actually recorded | implying a verdict that was not given |
| Non-claims wall / box | the boundary, stated before anyone asks | hidden behind hover |
| Decay clock | review-window consumption from registry fields | a countdown implying automatic invalidation |
| Binding (`repo @ sha`) | an immutable, **ref-reachable** support revision | a mutable link styled as a binding |

## 4 · Motion vocabulary

Motion verbs only: **reveal · branch · constrain · compare · rotate
representation · collapse · lock · expire.** If an animation cannot name
its verb, it does not ship. Everything gated on `prefers-reduced-motion`;
no autoplay loops (user-driven instruments instead — the site's
feasible-worlds slider replaced an infinite loop); no essential
information only in motion or only on hover.

## 5 · Figure compression rules

A film may merge logical operations; when it does, the compression is
**labeled in the caption and recorded in a machine manifest** the tests
hold prose to. Reference implementation:
`docs/assets/epistemic_machine.manifest.json` (six logical rows, five
cinematic panels, explicit `represents_rows` mapping) guarded by
`tests/unit/docs/test_epistemic_machine_manifest.py`. The same pattern at
smaller scale: ghost-ark's `figure-data.json` ("a figure cannot drift from
the number it draws") and the site's `scripts/verify_figures.py`
(committed geometry assertions to 1e-9).

## 6 · Accessibility rules

Keyboard-first (native controls; visible focus); complete `<title>/<desc>`
on argumentative SVGs — the description must carry the argument, not
describe the decoration; reduced-motion users get a static state that is
**epistemically neutral** (the site's instrument defaults to an endpoint
witness with the full interval drawn — never the independence world);
wide figures scroll in their own container at narrow widths; heading
order is semantic; 320 px is a first-class layout.

## 7 · Artifact manifest schema (minimum)

Any web-facing visualization of repository results ships a manifest with:

```
source_artifacts:   the kernel/ledger files it draws from (paths + revisions)
semantic_mappings:  which visual channel encodes which recorded field
forbidden_interpretations:  what a viewer must not conclude
non_claim:          what the render is not evidence of
```

`visual_identity/claim_observatory/visual_world_manifest_v3.json` is the
full-scale example; the epistemic-machine manifest is the minimal one.

## 8 · Anti-patterns

- A meter, gauge, or percentage ring over claim statuses (the
  evidence-cards manifest bans aggregates explicitly: "any surface that
  computes one from this file is misusing it").
- "not-run" rendered as pending, in-progress, or healthy.
- Certification badges without a certification.
- Decorative red, decorative countdowns, decorative hashes ("the timestamp
  and hash are set dressing" belongs in the caption when true).
- Visual complexity as authority: if an element answers none of the
  epistemic questions (what is claimed / assumed / feasible / falsifying /
  unestablished / inspectable / expiring), it is removed.

## Non-claims

This directory documents conventions. It does not certify that any page
follows them (the site's own CI does that for the site), does not make any
rendered surface evidence, and does not replace the repository's claim
governance.
