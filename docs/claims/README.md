# Claims

This folder contains the claim-boundary artifacts for CC-Framework.

- [Claim Boundary Manifest](CLAIM_BOUNDARY_MANIFEST.md) is the human-readable
  map of active public claims, validation lanes, supporting files, tests or
  commands, and explicit non-claims.
- [claim_boundary_manifest.v0.1.json](claim_boundary_manifest.v0.1.json) is
  the machine-readable version of the same boundary map.

The JSON manifest is intended for future validators, docs checks, release
checks, or website generation. It documents current claim boundaries; it does
not create new authority, certify deployment safety, or promote experimental
surfaces into the paper core.
