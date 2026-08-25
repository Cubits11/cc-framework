# E2 Field Test 001 — activation audit addendum

**Recorded:** 2026-08-24T15:42:19Z  
**Scope:** evidence-retention correction only. The study status remains
**NOT ACTIVATED** and no score, observation, or statistical result exists.

The first activation record and its checksum remain preserved as originally
written. They are local, uncommitted, hash-attested files—not immutable,
externally timestamped, signed, or independently reproduced evidence. File
permissions and SHA-256 sidecars help detect later changes but do not establish
when the files were created or who controlled them.

The observed no-input PromptPurify loader failure remains sufficient for the
protocol's stop rule: the frozen runner verified the tarball and declared
work-directory, then called `createL5eRunner()` without its documented
`modelDir` option. The released package instead searched relative to its own
installation and reported the missing model. No repair, relocation, wrapper,
retry, prompt, or ProtectAI loader run followed.

Evidence-retention gaps that prevent a stronger provenance claim:

- The local record does not retain a per-file PromptPurify extraction manifest,
  NPM lockfile copy, Python wheel list, archive listing, or command/stdout/stderr
  transcript with its own digest.
- It has no external timestamp or independent storage anchor showing the frozen
  manifest predates artifact retrieval.
- The protocol header's statement that no candidate had been installed or loaded
  is historical. A no-input local loader attempt occurred; it was not a
  prediction or measurement.

Accordingly, this packet supports only a local, self-attested activation-gate
finding. It does not independently prove artifact provenance, execution timing,
or the absence of any prompt outside the retained record. Publishing or auditing
this event would require a newly specified, externally anchored evidence packet;
it must not be retrofitted into this frozen study.
