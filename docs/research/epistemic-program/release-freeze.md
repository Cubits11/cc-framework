# Release Freeze

Written last, on purpose. No new ideas after this point — only a record of what
exists and a decision about what ships.

Temperance gets the session. The Emperor gets the last ten minutes.

## What exists now

| Layer | Artifact | State |
| --- | --- | --- |
| Research | Finite-atom kernel, Fréchet bounds, endpoint witnesses, sample complexity | Release-candidate, tested |
| Governance | Role ontology, confirmatory protocol, decay, receipts, claim governance | Implemented, tested |
| Capsule | Deterministic claim-governance capsule | Reproducible; boundary demonstrated by CH-001 |
| Method | Future-expansion package: source ledger, eight-week plan, prompt library | Written |
| Film | *Before You See It*, three cuts, 15.000s each | Rendered |
| Page | Canonical page with live WebCrypto check | Built and browser-tested |
| Program | Inventory, research graph, scope inflation, challenges, manifold, packet, intake | Written this session |
| Guards | Drift guard, artifact boundary, language quarantine, claim manifest validator | Passing |

## The decisions

**What ships.**
The canonical page and the film. Nothing else in the same release.

**The central demonstration.**
The browser integrity check. One object, operable by a stranger in under a
minute, breakable on purpose.

**The sentence that accompanies it.**

```text
Bytes verified. Claim unresolved.
```

**What is explicitly not being claimed.**

- Not that any system is safe.
- Not that a digest match says anything about the value inside the file.
- Not that reproducibility from declared inputs establishes that those inputs
  were measured — CH-001 shows it does not.
- Not that the method has been validated. It has been defined and applied here.
- Not that any offer has been delivered or any intake has been run.

**Where someone inspects the source.**

```text
page      visual_identity/canonical_page/index.html
fixture   examples/claim_governance_capsule/expected/calibration.json
digest    examples/claim_governance_capsule/manifest.expected.json
guard     tests/unit/docs/test_canonical_page_fixture.py
film      visual_identity/before_you_see_it/
finding   docs/research/epistemic-program/challenges.md   (CH-001)
```

**The one action a visitor should take.**

Change one character in the box and watch it fail. Everything else — the film,
the ledger, the invitation — is optional. That single interaction is the whole
argument, and it is the only thing the release asks for.

## What is deliberately not in this release

- The three short clips. Ready to derive, not ready to ship.
- Posts 3–5 of the packet. Post 3 waits for CH-001's write-up and the page's
  updated limitation to be live, so it lands on an artifact that already
  reflects the finding.
- Any paid offer. The offers are defined; none has been sold or delivered.
- CH-002 through CH-005. Open, unattempted, and published as open.

## The next artifact

CH-002: close rung 4, or establish that no repository-local mechanism can. It is
the highest-value open question in the program, it came out of attacking our own
work, and either outcome is publishable.

## Frozen

No redesigning the philosophy at minute 178.
