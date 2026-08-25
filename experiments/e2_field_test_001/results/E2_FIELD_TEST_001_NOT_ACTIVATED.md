# E2 Field Test 001 — NOT ACTIVATED

The frozen local pilot stopped at its first no-input loader gate. The exact
PromptPurify release tarball, model files, NPM package, peer runtime, and
declared work directory all verified, but the released package's
`createL5eRunner()` resolved the model only relative to its own installation.
The frozen runner calls that API without its documented `modelDir` override,
so it could not load the artifact under the frozen `<workdir>/models/l5e`
location.

Moving or linking the model into the package directory, changing the runner,
or using a wrapper would change the frozen execution condition. The protocol
therefore requires **NOT ACTIVATED** rather than a repair. No prompt was sent
to either classifier and no E2 observation or derived measurement exists.

See `E2_FIELD_TEST_001_ACTIVATION_RECORD.json` for the immutable evidence
record, pinned artifact digests, runtime locks, and exact loader error.
