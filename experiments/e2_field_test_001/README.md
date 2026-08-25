# E2 Field Test 001 harness

This directory contains only the deterministic input-selection and analysis
parts of the planned field test. It contains no model artifacts and does not
run a classifier.

Before any artifact that can be scored is retrieved, hash this directory and
the protocol, record the hashes in the run manifest, and preserve the exact
commands used.

## Prepare the frozen inputs

```bash
python experiments/e2_field_test_001/prepare_items.py \
  --dataset-root /path/to/S-Labs-prompt-injection-dataset \
  --dataset-revision 002a9dd18514abd021869823d6b0429b38606d99 \
  --out-dir /path/to/frozen-items
```

The program requires the three source CSV files and the exact `text,label`
schema described by the protocol. It writes a private text-bearing file for
local scoring, a hash-only `items.jsonl`, and a read-only selection manifest.
It refuses fewer than 1,097 unique source-labelled `INJECTION` records and
never replaces them.

```bash
python experiments/e2_field_test_001/prepare_items.py --self-test
```

The self-test constructs synthetic strings only; it is not a guardrail test.

## Analyze an already-produced observation file

```bash
python experiments/e2_field_test_001/analyze.py \
  --rows /path/to/observations.jsonl \
  --items /path/to/frozen-items/items.jsonl \
  --guardrail-a sl7.promptpurify.v0.0.1 \
  --guardrail-b protectai.deberta-v3-base-prompt-injection-v2 \
  --out /path/to/analysis.json
```

It is intentionally post-score only: it requires an exact two-row pairing for
every frozen item, derives the preregistered primary statistic and controls,
and will not overwrite an existing analysis report. The script's self-test
uses synthetic binary outputs only:

```bash
python experiments/e2_field_test_001/analyze.py --self-test
```

## Independently recompute the primary result

```bash
node experiments/e2_field_test_001/recompute.mjs \
  --rows /path/to/observations.jsonl \
  --items /path/to/frozen-items/items.jsonl \
  --analysis /path/to/analysis.json \
  --guardrail-a sl7.promptpurify.v0.0.1 \
  --guardrail-b protectai.deberta-v3-base-prompt-injection-v2
```

The recomputation is a separately written, dependency-free implementation of
the primary statistic and duplicate-column control. `PASS` means those limited
numbers matched the analysis output within `1e-12`; it is not independent
review, model validation, or a semantic verdict.

## Local runners

`run_promptpurify.mjs` and `run_protectai.py` are deliberately local-only
arms. They require the frozen private item file, a provenance JSON record, and
verified local artifacts. Both refuse existing output paths and append an event
record before each normalized observation. They are not run by either self-test.

Use each runner's `--smoke-only` mode after locally verifying artifacts but
before the first prompt is supplied. It constructs the local model/session and
prints its configuration hash without reading or scoring the item file.

Before running either arm, satisfy every activation gate in the protocol and
record a run manifest containing the hashes of the protocol, this directory,
the source CSV files, package artifacts, and model artifacts. Then validate the
combined rows with `scripts/validate_e2_observations.py` before analyzing them.
