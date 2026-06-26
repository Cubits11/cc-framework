# Private Transparency Log Security Model

This design replaces a local hash chain with a private transparency log:

1. Audit records are appended to an RFC 6962-style Merkle tree.
2. The run attestation signs a checkpoint containing the Merkle root, tree size,
   and a fresh `run_nonce`.
3. When configured, an external witness co-signs the same root checkpoint. A
   verifier treats that witness signature as the externally anchored root.

The local JSONL files are still useful evidence, but they are not trusted by
themselves. Verification is against signed and, when configured, witnessed
checkpoints.

## Cryptographic Commitments

`src/cc/evidence/merkle_log.py` uses domain-separated hashes:

* leaf: `SHA256(0x00 || canonical_record_json)`
* internal node: `SHA256(0x01 || left_child || right_child)`
* empty tree: `SHA256(b"")`

An inclusion proof shows that one audit record is committed under a trusted root.
A consistency proof shows that a later root is an append-only extension of an
earlier root. These checks are verifier-side functions and do not require trust
in the log host.

`src/cc/evidence/anchoring.py` provides a witness interface. The implemented
Ed25519 witness signs a checkpoint:

* `log_id`
* `tree_size`
* `root_hash`
* `run_nonce`
* `issued_at`
* optional `previous_root_hash`

In production, the witness private key must be controlled outside the log host.
The in-repo Ed25519 witness implementation is a protocol implementation and test
fixture, not a claim that a local key file is independent.

## Attestation Binding

Evidence bundle attestations use `cc/evidence-attestation.v2`. A signed
attestation covers:

* `run_id`
* `run_nonce`
* manifest hash
* metrics hash
* Merkle `root_hash`
* Merkle `tree_size`
* optional witness anchor

The `run_nonce` is also written into `manifest.json`, so replaying a valid old
attestation into a different run context changes the manifest hash or nonce
check and fails verification.

Unsigned bundles are allowed only when explicitly requested with `unsigned=True`
or `--unsigned`. No private key is generated into an output directory. Signed
bundles require an externally supplied Ed25519 private key path outside the
bundle output directory.

## Adversary Model

### Attacker With Filesystem Write Access

This attacker can edit, delete, reorder, or replace local files, including
`results.jsonl`, `transparency_log.jsonl`, `ledger.jsonl`, and
`attestation.json`.

They can:

* rewrite the whole local log and recompute a self-consistent Merkle root;
* delete evidence that has not yet been externally copied or anchored;
* roll the directory back to an older set of files.

They cannot, assuming the signing key and witness key are unavailable:

* produce a valid signature for a changed attestation;
* produce a valid witness signature for a changed Merkle root;
* make a rewritten log verify against a previously witnessed root;
* prove append-only consistency from a witnessed old root to a rewritten history
  with a different prefix.

Detection depends on the verifier holding, fetching, or trusting an external
checkpoint or witness public key. If no root has ever been signed, witnessed, or
copied outside the host, a full rewrite before first anchoring is not
cryptographically detectable.

### Attacker Who Controls the Signing Key

This attacker can sign new bundle attestations.

They can:

* create a valid signed attestation for a false or selective local log;
* sign an attestation that omits a witness anchor if verifier policy does not
  require anchoring;
* create a new run with a new nonce and valid signature.

They cannot, assuming the witness key is independent and verifier policy
requires a witness anchor:

* forge the witness signature on a rewritten root;
* make an unwitnessed rewritten root verify when `require_anchor=True`;
* make an old attestation verify in a new run context without also preserving
  the old manifest hash and nonce.

The signing key authenticates the bundle producer. It does not by itself prove
global append-only history. That property comes from consistency proofs against
previously anchored roots.

### Attacker Who Controls Both Signing Key and Log Host

This attacker can rewrite local history and sign new attestations.

They can:

* create internally consistent, signed false histories after the point of
  compromise;
* omit records from a new run and sign that run;
* delete local evidence.

They cannot, if an independent witness anchor from before the rewrite remains
available and the witness key is not compromised:

* rewrite history before that anchored root without failing anchor or
  consistency verification;
* forge inclusion under the old anchored root for records that were not present;
* silently replace an anchored root with a different root.

If this attacker also controls the witness key, or can make verifiers trust a
new malicious witness key, the cryptographic append-only guarantee is lost from
that point forward. Existing checkpoints already replicated to independent
verifiers may still expose equivocation or rollback.

## Verification Policy

For high-assurance verification:

1. Require signed attestations.
2. Require witness anchors for production runs.
3. Pin trusted witness public keys by witness id.
4. Store root checkpoints outside the log host.
5. Verify Merkle inclusion for records of interest.
6. Verify Merkle consistency from older anchored roots to newer anchored roots.

The tests in `tests/unit/evidence/test_transparency_log_adversarial.py` encode
the required attack failures: full-log rewrite and rehash, proof forgery,
attestation replay under a new nonce, and anchoring bypass when a witness is
configured.
