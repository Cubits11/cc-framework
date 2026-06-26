# Active Red-Team Dependence Search Bounds

`cc.redteam.dependence_search` is a bounded discovery tool for finding inputs
that make a composed guardrail stack fail together. It is intended for defensive
measurement of dependence cliffs, not for open-ended adversarial content
generation.

## Perturbation Boundary

Search starts from an existing prompt corpus. Candidate prompts can only be made
by applying transformations declared in the run config:

- `synonym_substitution`: replace configured source terms with configured
  replacements, capped per prompt.
- `phrase_rewrite`: apply configured phrase rewrites only when the rewritten
  prompt stays within the configured token-Jaccard similarity budget.
- `public_prompt_injection`: wrap the prompt with a configured `{prompt}`
  template whose config names an already-public benchmark and public reference.

The engine does not call an LLM, web service, thesaurus, or model-based
paraphraser to invent new candidate content. It also enforces maximum candidate
length, maximum total transformations, and per-transformation caps.

## Safety Gate

Every candidate is checked by `ContentSafetyGate` before guardrails are evaluated
or the search objective is computed. The gate is not configurable to disabled.

The default gate blocks:

- candidate text over `max_candidate_chars`;
- configured blocked terms for high-risk synthetic probes, such as requests for
  explosives, malware, credential theft, ransomware, phishing kits, and explicit
  self-harm instruction requests;
- configured regular expressions covering the same classes with simple lexical
  variation.

When the gate blocks a candidate, the candidate is discarded and counted. It is
not sent to the guardrail stack and does not reach the objective function.
Held-out baseline prompts are checked with the same gate; an unsafe baseline
prompt fails the run because there is no safe replacement for a held-out record.

## Objective And Report

The objective estimates whether discovered candidates have higher shared
pass-through/miss dependence than a held-out passive baseline. It reports:

- average pairwise binary Kendall-style dependence across guardrail miss
  indicators;
- multiway joint co-failure rate;
- joint tail co-failure rate, normalized by the rarest individual miss rate;
- search-discovered versus passive-baseline shifts;
- a `cliff_certificate()` result for the discovered joint tail co-failure rate.

Artifacts include the resolved search config, seed, corpus hashes, discovered
input hashes, certificate, and file hashes. Raw discovered prompt text is omitted
by default and is written only with the explicit
`--include-raw-discoveries` opt-in or the matching Python API flag.

## Limits

The safety gate is a deterministic lexical filter. It is deliberately hard and
pre-objective, but it is not a complete content moderation system and does not
understand every paraphrase, language, code word, or context. The perturbation
boundary is therefore part of the safety design: declared transformations must be
reviewed before use, especially public prompt-injection templates and synonym or
phrase tables.

The discovered-cliff report is evidence of dependence behavior under the
declared corpus, perturbation space, guardrail stack, and seed. It is not a
general safety certificate for a deployed system and should be included in an
evidence bundle with the config, hashes, version metadata, and any external
review notes.
