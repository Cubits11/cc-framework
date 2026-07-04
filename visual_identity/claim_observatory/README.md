# Claim Observatory Blender Scene

Editable cinematic identity scene for `cc-framework / Ghost Protocol / Claim Observatory`.

Core thesis:

```text
Not safety scores. Evidence-bound claims.
```

## Outputs

- `claim_observatory.blend` - editable Blender 5.1 scene with modular named collections.
- `renders/claim_observatory_hero.png` - 1920x1080 final hero frame for repo hero, demo intro, or thumbnail use.
- `create_claim_observatory_scene.py` - deterministic scene generator.

## Build

From the repository root:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python visual_identity/claim_observatory/create_claim_observatory_scene.py \
  -- --render-still
```

Run without `-- --render-still` to regenerate only the `.blend`.

## Scene Collections

- `ClaimCapsule`
- `EvidenceArtifacts`
- `SupportGraph`
- `NonClaimsWall`
- `DecayClock`
- `ReplayManifest`
- `HumanReview`
- `CameraRig`
- `Lighting`

## Shot Plan

The timeline is set to 360 frames at 24 fps, about 15 seconds. Timeline markers are attached to the main cinematic camera:

1. Wide reveal of the dark observatory.
2. Push toward the glowing Claim Capsule.
3. Orbit around support graph edges.
4. Pan to the Non-Claims Wall.
5. Close-up on the Decay Clock.
6. Final hero frame with the thesis text.

## Visual Semantics

Cyan/white light marks evidence and replayable structure. Amber marks required review. Red is reserved for expiration/invalidation. The scene intentionally avoids green safety badges, fake certification imagery, and generic SaaS dashboard framing.
