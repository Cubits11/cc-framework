# Claim Observatory World V3

## Purpose

V3 turns the original single-room Claim Observatory into a modular world. Each chamber corresponds to one stage of claim integrity: bounded claim body, evidence roles, typed support, non-claims, decay, challenge, replay, review, ledger lifecycle, and Frechet endpoint witnesses.

## Core Thesis

```text
Not safety scores.
Evidence-bound claims.
```

## Why This Is A World, Not A Dashboard

A dashboard implies live operational control or a safety score. This world uses physical spaces instead: capsule, vault, orrery, wall, clock, range, engine, tribunal, tower, and mathematical garden. The visual grammar teaches constraints, not confidence theater.

## Chamber Map

- `01_ArrivalHall`: thesis entry.
- `02_ClaimCapsuleChamber`: central claim body.
- `03_EvidenceVault`: role-specific evidence tablets.
- `04_SupportGraphOrrery`: typed support edges.
- `05_NonClaimsWall`: semantic firewall.
- `06_DecayClockRoom`: Fresh / Degraded / Expired mechanism.
- `07_ChallengeRange`: boundary-testing probes.
- `08_ReplayManifestEngine`: manifest file floor engine.
- `09_HumanReviewTribunal`: scoped review authority.
- `10_LedgerTower`: future public claim lifecycle.
- `11_FrechetAtomGarden`: finite-atom endpoint witnesses.

## Object-To-Artifact Map

- Claim Capsule: `examples/claim_governance_capsule/expected/claim_envelope.json`
- Governance status labels: `examples/claim_governance_capsule/expected/claim_governance_audit.json`
- Report plate: `examples/claim_governance_capsule/expected/cc_report.json`
- Replay tiles: `examples/claim_governance_capsule/manifest.expected.json`
- Decay clock: `examples/claim_governance_capsule/expected/decay_policy.json`
- Frechet witnesses: `examples/claim_governance_capsule/expected/extremal_lower.json`, `examples/claim_governance_capsule/expected/extremal_upper.json`
- Confirmatory evidence: `examples/claim_governance_capsule/expected/confirmatory_protocol.json`, `examples/claim_governance_capsule/expected/confirmatory_failure_matrix.json`

## Camera List

- `Camera_WorldHero`
- `Camera_Arrival`
- `Camera_ClaimCapsule`
- `Camera_EvidenceVault`
- `Camera_SupportGraphOrrery`
- `Camera_NonClaimsWall`
- `Camera_DecayClockRoom`
- `Camera_ChallengeRange`
- `Camera_ReplayManifestEngine`
- `Camera_HumanReviewTribunal`
- `Camera_LedgerTower`
- `Camera_FrechetAtomGarden`

## Render Commands

From the repository root:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python visual_identity/claim_observatory/create_claim_observatory_world_v3.py \
  -- --render-stills
```

To regenerate the `.blend` and manifest without rendering:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python visual_identity/claim_observatory/create_claim_observatory_world_v3.py
```

## How To Regenerate

The generator loads the current checked-in capsule artifacts from `examples/claim_governance_capsule/expected/` and `examples/claim_governance_capsule/manifest.expected.json`. It writes:

- `claim_observatory_world_v3.blend`
- `visual_world_manifest_v3.json`
- `renders/world_v3_*.png`

## Text Legibility Rules

Large text must be readable in the hero camera. Medium text is used for chamber status and caveats. Small text is reserved for close-up cameras and should not carry the hero-frame meaning.

The hero shot should teach the thesis in one second. Detail shots can teach the machinery.

## No-Overclaim Rules

- Do not show a green safety badge.
- Do not show a final approval checkmark.
- Do not imply deployment safety.
- Do not turn receipt integrity into statistical validity.
- Do not let a PASS verdict visually erase non-claims.
- Do not show human review as magic.

## What This World Does Not Claim

This render is not evidence.
This render does not claim AI deployment safety.
This render does not replace the claim-governance audit.
This render is a visual explanation of the evidence-bound claim architecture.
This world is a teaching artifact, not a verifier.

## Next Visual Passes

V4 should add an animation path and align the still-world grammar with any future dashboard visual language. The priority should be camera motion, chamber-to-chamber transitions, and better close-up typography.

