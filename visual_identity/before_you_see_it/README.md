# Before You See It

A 15-second film about the moment a seductive story is forced to become a test.

```text
A story can start a question.
Evidence must finish the answer.
```

**Keep the wonder. Check the claim.**

| | |
| --- | --- |
| Treatment | [DIRECTORS_CUT.md](DIRECTORS_CUT.md) - beats, color law, voiceover, sound, production rules |
| Source | [film.html](film.html) - the film itself, one self-contained file |
| Render | [render_film.py](render_film.py) - deterministic frame capture and encode |
| Clip | [render_check_clip.py](render_check_clip.py) - Surface B, the ten-second `BYTES MATCH → FAIL CLOSED` clip, driven against the real page |
| Chamber | Chamber 00 of the [Claim Observatory](../claim_observatory/WORLD_BIBLE_V3.md) |
| Method | [Future Expansion](../../docs/research/future-expansion/README.md) - the research package the film dramatizes |
| Page | [Canonical page](../canonical_page/index.html) - the film, transcript, ledger, limits strip, and one real check |

## Watch it

Open `film.html` in any browser. It plays on load; click to replay.

## Render it

```bash
python3 -m pip install playwright          # Chromium is already present in CI images
python3 visual_identity/before_you_see_it/render_film.py --cut cc-framework
python3 visual_identity/before_you_see_it/render_film.py --cut ghost-ark
```

Outputs land in `renders/`: an H.264 `.mp4`, a VP9 `.webm`, and a poster frame.

The film is a **pure function of time**. The page exposes `window.__seek(t)` and
the renderer walks the timeline frame by frame rather than recording playback, so
the same commit produces the same frames on any machine. Nothing depends on
wall-clock speed, and there is no font download - only faces already present in a
standard Linux render environment are used.

## The two cuts

Same film, same timing, same frames. Only the final lockup changes.

| Cut | Wordmark | Tagline |
| --- | --- | --- |
| `cc-framework` | CC-FRAMEWORK | Keep the wonder. Check the claim. |
| `ghost-ark` | GHOST-ARK | Evidence you can inspect. |
| `cubits11` | CUBITS11 | Keep the wonder. Check the claim. |

## The verdict rule

The result card shows `window.__VERDICT` and nothing else. With no verdict
injected it falls back to an illustrative `INCONCLUSIVE` **and stamps
`ILLUSTRATION` in the corner of the frame** for the whole beat.

A film about not overclaiming cannot display an unearned result, so it labels its
own unverified moment on screen. To show a real one, hand it a real verifier's
output:

```bash
python3 visual_identity/before_you_see_it/render_film.py \
    --cut ghost-ark --verdict verdict.json
```

```json
{ "state": "FAIL CLOSED", "detail": "mutated receipt rejected\nverifier: independent", "color": "#D9534F" }
```

For the Ghost-Ark cut that is the intended path: run the malicious-corpus
verifier, take its actual output, and let the film show a real mutation failing
closed rather than a decorative checkmark.

## Placing the Ghost-Ark cut

The Ghost-Ark cut is built to live in `PSUCyberSecurityLab/ghost-ark`. It was not
pushed there from this repository - that session could not attach a repository
outside the `Cubits11` owner - so it ships here, ready to move:

```bash
# from a session or checkout that has ghost-ark
mkdir -p docs/media
cp renders/before_you_see_it__ghost-ark.mp4  docs/media/
cp renders/poster__ghost-ark.png             docs/media/
```

Then in the Ghost-Ark README:

```markdown
https://github.com/PSUCyberSecurityLab/ghost-ark/raw/main/docs/media/before_you_see_it__ghost-ark.mp4
```

Regenerating it there needs only `film.html` and `render_film.py`; both are
self-contained and have no repository-specific dependencies.

## Non-claims

- The film is a statement of method. It is not evidence and reports no
  measurement.
- The result card is an illustration unless a verdict was injected.
- The timestamp and hash in the protocol bar are set dressing at fixed values.
  They are not a receipt over a real artifact.
- Neither lockup asserts that the named project has proven anything. Both name a
  standard the project holds itself to.
