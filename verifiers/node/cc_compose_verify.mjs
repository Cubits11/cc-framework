#!/usr/bin/env node
/**
 * ═══════════════════════════════════════════════════════════════════════════
 * INDEPENDENT CC COMPOSITION KERNEL — Node, zero dependencies
 *
 * Written from `conformance/cc-kernel-v1/SPEC.md` and the JSON corpus. It does
 * not import, transcribe, or consult `cc.compose`. That constraint is the
 * whole value: an implementation written by reading the Python would inherit
 * its mistakes, and agreement between the two would establish nothing.
 *
 * ── What agreement here does and does not establish ────────────────────────
 *
 * Agreement establishes that two implementations, working from a written
 * specification, compute the same intervals on the corpus. It does NOT
 * establish that either is correct. Both were authored in the same project,
 * and a specification that is wrong produces two implementations that are
 * wrong together. This is a DIFFERENTIAL TESTING INSTRUMENT, not an
 * independent replication.
 *
 * The genuinely independent check is `--external-oracle`, which reproduces
 * numbers published by a separate project that implemented these bounds for
 * its own purposes, without reference to this corpus. See §"external oracle"
 * below.
 *
 *   node verifiers/node/cc_compose_verify.mjs                  run the corpus
 *   node verifiers/node/cc_compose_verify.mjs --json           machine-readable
 *   node verifiers/node/cc_compose_verify.mjs --external-oracle
 *   node verifiers/node/cc_compose_verify.mjs --batch < cases.json
 * ═══════════════════════════════════════════════════════════════════════════
 */

import { readFileSync, readSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const CORPUS = join(HERE, '..', '..', 'conformance', 'cc-kernel-v1');

/* ── SPEC §1.3 · event kinds ─────────────────────────────────────────────── */

const EVENT_ALIASES = new Map([
  ['all', 'all'], ['and', 'all'], ['AND', 'all'], ['intersection', 'all'],
  ['any', 'any'], ['or', 'any'], ['OR', 'any'], ['union', 'any']
]);

/* ── SPEC §1.4 · dependence assumptions ──────────────────────────────────── */

const DEPENDENCE = new Set([
  'unconstrained', 'independent', 'comonotone', 'countermonotone'
]);

/** A refusal carrying the SPEC §5 reason identifier. */
class Refusal extends Error {
  constructor(reason, detail) {
    super(`${reason}: ${detail}`);
    this.reason = reason;
  }
}

const clip01 = (x) => Math.min(1, Math.max(0, x));

/* ── SPEC §1.2 · marginal validation ─────────────────────────────────────── */

function validateMarginals(marginals) {
  if (marginals === null || typeof marginals !== 'object' || Array.isArray(marginals)) {
    throw new Refusal('no_events', 'marginals must be a mapping of name to probability');
  }
  const names = Object.keys(marginals);
  if (names.length === 0) throw new Refusal('no_events', 'at least one event is required');

  const out = [];
  for (const name of names) {
    if (typeof name !== 'string' || name.length === 0) {
      throw new Refusal('no_events', `event name must be a non-empty string: ${name}`);
    }
    const p = marginals[name];
    if (typeof p !== 'number' || !Number.isFinite(p)) {
      throw new Refusal('marginal_not_finite', `${name} = ${p}`);
    }
    if (p < 0 || p > 1) {
      throw new Refusal('marginal_out_of_range', `${name} = ${p}`);
    }
    out.push([name, p]);
  }
  return out;
}

/* ── SPEC §2 · the unconstrained interval ────────────────────────────────── */

function classical(values, kind) {
  const n = values.length;
  const sum = values.reduce((a, b) => a + b, 0);
  if (kind === 'all') {
    return [clip01(Math.max(0, sum - (n - 1))), clip01(Math.min(...values))];
  }
  return [clip01(Math.max(...values)), clip01(Math.min(1, sum))];
}

/* ── SPEC §3 · the independence baseline ─────────────────────────────────── */

function independencePoint(values, kind) {
  if (kind === 'all') return values.reduce((a, b) => a * b, 1);
  return clip01(1 - values.reduce((a, b) => a * (1 - b), 1));
}

/* ── SPEC §4 · binding_event ─────────────────────────────────────────────── */

const TIE_TOL = 1e-12;

function bindingEvent(pairs, kind) {
  if (kind === 'all') {
    const min = Math.min(...pairs.map(([, p]) => p));
    const atMin = pairs.filter(([, p]) => Math.abs(p - min) <= TIE_TOL);
    // §4.3 — a tie means no single event binds.
    return atMin.length === 1 ? atMin[0][0] : null;
  }
  // §4.2 — the union upper bound is a sum, not a single event.
  return pairs.length === 1 ? pairs[0][0] : null;
}

/* ── the kernel ──────────────────────────────────────────────────────────── */

export function composeBounds(marginals, { event = 'all', dependence = 'unconstrained' } = {}) {
  const kind = EVENT_ALIASES.get(event);
  if (kind === undefined) throw new Refusal('unknown_event_kind', String(event));
  if (!DEPENDENCE.has(dependence)) throw new Refusal('unknown_dependence', String(dependence));

  const pairs = validateMarginals(marginals);
  const values = pairs.map(([, p]) => p);

  // §6 — countermonotonicity is strictly bivariate: exactly two, never more,
  // never fewer. Without this the n=1 branch reads values[1] as undefined and
  // returns NaN rather than refusing. Found by differential fuzz, seed 1.
  if (dependence === 'countermonotone' && values.length !== 2) {
    throw new Refusal(
      'countermonotone_undefined',
      values.length < 2
        ? `${values.length} event(s); countermonotonicity is a relation between ` +
          'two events, so there is nothing to be countermonotone with'
        : `${values.length} events; there is no n-dimensional countermonotonic ` +
          'structure for n > 2, and the FH lower bound is not a copula in ' +
          'dimension >= 3 (though it stays pointwise sharp)'
    );
  }

  const indep = independencePoint(values, kind);
  let [lower, upper] = classical(values, kind);

  if (dependence === 'independent') {
    lower = upper = indep;
  } else if (dependence === 'comonotone') {
    lower = upper = kind === 'all' ? clip01(Math.min(...values)) : clip01(Math.max(...values));
  } else if (dependence === 'countermonotone') {
    lower = upper = kind === 'all'
      ? clip01(Math.max(0, values[0] + values[1] - 1))
      : clip01(Math.min(1, values[0] + values[1]));
  }

  // §4.4 — no event binds in a stipulated regime.
  const binding = dependence === 'unconstrained' ? bindingEvent(pairs, kind) : null;

  return {
    lower,
    upper,
    width: Math.max(0, upper - lower),
    event: kind,
    dependence,
    independence_point: indep,
    independence_regret: upper - indep,
    // §3 — JSON has no infinity; null means undefined, not large.
    understatement_factor: indep > 0 ? upper / indep : null,
    binding_event: binding
  };
}

/* ── corpus runner ───────────────────────────────────────────────────────── */

const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'));

function runAccept(tolerance) {
  const doc = readJson(join(CORPUS, 'cases', 'composition.json'));
  const failures = [];
  for (const c of doc.cases) {
    let got;
    try {
      got = composeBounds(c.input.marginals, {
        event: c.input.event,
        dependence: c.input.dependence
      });
    } catch (err) {
      failures.push({ id: c.id, field: '<refused>', detail: err.message });
      continue;
    }
    for (const [field, want] of Object.entries(c.expect)) {
      const have = got[field];
      if (field === 'binding_event' || want === null || have === null) {
        // §4 — compared exactly, including the null cases.
        if (have !== want) failures.push({ id: c.id, field, want, have });
        continue;
      }
      if (Math.abs(have - want) > tolerance) {
        failures.push({ id: c.id, field, want, have, delta: Math.abs(have - want) });
      }
    }
  }
  return { total: doc.cases.length, failures };
}

function runReject() {
  const doc = readJson(join(CORPUS, 'cases', 'adversarial.json'));
  const failures = [];
  for (const c of doc.cases) {
    // §5 JSON note — the corpus encodes NaN as the string "NaN".
    const marginals = {};
    for (const [k, v] of Object.entries(c.input.marginals)) {
      marginals[k] = v === 'NaN' ? Number.NaN : v;
    }
    try {
      composeBounds(marginals, { event: c.input.event, dependence: c.input.dependence });
      failures.push({ id: c.id, detail: 'accepted, but the corpus requires a refusal' });
    } catch (err) {
      if (!(err instanceof Refusal)) {
        failures.push({ id: c.id, detail: `threw ${err.constructor.name}, not a typed refusal` });
      } else if (err.reason !== c.expect_refusal) {
        failures.push({ id: c.id, want: c.expect_refusal, have: err.reason });
      }
    }
  }
  return { total: doc.cases.length, failures };
}

/* ── external oracle ─────────────────────────────────────────────────────────
 *
 * Numbers PUBLISHED by a separate project that implemented these bounds
 * independently, for its own purposes, before this corpus existed. Reproducing
 * them is the one check here that is not same-author: neither this file nor
 * the Python reference was consulted when those numbers were produced.
 *
 * Source: a four-control ledger composition study. Detection rates are ASSUMED
 * by that project — zero records had been issued — so these are bounds over
 * assumed inputs, never measurements.
 */
const EXTERNAL_ORACLE = [
  {
    name: 'four-control ledger, small-chapter scenario',
    // Published as detection rates; evasion is 1 - detection.
    detection: { SELF_REPORTED: 0.70, STALE: 0.99, IMMUTABLE: 0.99, SEPARATION: 0.60 },
    published: { lower: 0.0, upper: 0.01, width: 0.01, independence_point: 0.000012 },
    // Published rounded, as "understates the admissible worst case by 833x".
    publishedFactorRounded: 833
  }
];

function runExternalOracle(tolerance) {
  const failures = [];
  for (const o of EXTERNAL_ORACLE) {
    const marginals = {};
    for (const [k, d] of Object.entries(o.detection)) marginals[k] = 1 - d;
    const got = composeBounds(marginals, { event: 'all' });
    for (const [field, want] of Object.entries(o.published)) {
      if (Math.abs(got[field] - want) > tolerance) {
        failures.push({ oracle: o.name, field, want, have: got[field] });
      }
    }
    const factor = Math.round(got.understatement_factor);
    if (factor !== o.publishedFactorRounded) {
      failures.push({
        oracle: o.name, field: 'understatement_factor(rounded)',
        want: o.publishedFactorRounded, have: factor
      });
    }
  }
  return { total: EXTERNAL_ORACLE.length, failures };
}

/* ── batch mode, for the differential harness ────────────────────────────────
 *
 * Reads {"cases":[{"id","marginals","event","dependence"}]} on stdin and emits
 * one result or one typed refusal per case. The harness on the other side
 * compares; this side never sees the reference answers, so it cannot converge
 * on them.
 */
/**
 * Read all of stdin. `readFileSync(0)` returns only the first chunk when stdin
 * is a pipe, which silently truncated large batches at 64 KiB.
 */
function readStdin() {
  const chunks = [];
  const buf = Buffer.alloc(1 << 16);
  for (;;) {
    let n;
    try {
      n = readSync(0, buf, 0, buf.length, null);
    } catch (err) {
      if (err.code === 'EAGAIN') continue;
      if (err.code === 'EOF') break;
      throw err;
    }
    if (n === 0) break;
    chunks.push(Buffer.from(buf.subarray(0, n)));
  }
  return Buffer.concat(chunks).toString('utf8');
}

function runBatch() {
  const doc = JSON.parse(readStdin());
  const results = doc.cases.map((c) => {
    try {
      const got = composeBounds(c.marginals, {
        event: c.event ?? 'all',
        dependence: c.dependence ?? 'unconstrained'
      });
      return { id: c.id, ok: true, ...got };
    } catch (err) {
      return {
        id: c.id,
        ok: false,
        refusal: err instanceof Refusal ? err.reason : `untyped:${err.constructor.name}`
      };
    }
  });
  process.stdout.write(JSON.stringify({ results }));
  return 0;
}

/* ── main ────────────────────────────────────────────────────────────────── */

function main(argv) {
  if (argv.includes('--batch')) return runBatch();
  const wantJson = argv.includes('--json');
  const oracleOnly = argv.includes('--external-oracle');
  const manifest = readJson(join(CORPUS, 'manifest.json'));
  const tol = manifest.tolerance;

  // The oracle numbers are published to 1e-6, not to the corpus tolerance.
  const oracle = runExternalOracle(1e-9);
  const accept = oracleOnly ? { total: 0, failures: [] } : runAccept(tol);
  const reject = oracleOnly ? { total: 0, failures: [] } : runReject();

  const failed = accept.failures.length + reject.failures.length + oracle.failures.length;
  const report = {
    corpus: manifest.corpus,
    tolerance: tol,
    implementation: 'verifiers/node/cc_compose_verify.mjs',
    derived_from: 'conformance/cc-kernel-v1/SPEC.md',
    accept: { total: accept.total, failed: accept.failures.length, failures: accept.failures },
    reject: { total: reject.total, failed: reject.failures.length, failures: reject.failures },
    external_oracle: { total: oracle.total, failed: oracle.failures.length, failures: oracle.failures },
    // Provenance: exact counts, no interval. The corpus is the whole
    // population, not a sample.
    provenance: 'census',
    agreement_non_claim:
      'Agreement establishes that two implementations of one specification ' +
      'compute the same values. It does not establish that either is correct: ' +
      'both were authored in the same project and can share a misreading. ' +
      'Only external_oracle is not same-author.'
  };

  if (wantJson) {
    console.log(JSON.stringify(report, null, 2));
    return failed ? 1 : 0;
  }

  console.log('independent CC composition kernel — corpus agreement');
  console.log(`  corpus     ${manifest.corpus}  (tolerance ${tol})`);
  console.log(`  derived    ${report.derived_from}`);
  console.log('');
  if (!oracleOnly) {
    console.log(`  accept     ${accept.total - accept.failures.length}/${accept.total} agree`);
    console.log(`  reject     ${reject.total - reject.failures.length}/${reject.total} refuse correctly`);
  }
  console.log(`  oracle     ${oracle.total - oracle.failures.length}/${oracle.total} external published results reproduced`);
  console.log('');
  for (const f of [...accept.failures, ...reject.failures, ...oracle.failures]) {
    console.log(`  FAIL ${JSON.stringify(f)}`);
  }
  console.log(failed ? `FAIL: ${failed} disagreement(s).` : 'PASS: full agreement.');
  console.log('\n  ' + report.agreement_non_claim.replace(/(.{72}) /g, '$1\n  '));
  return failed ? 1 : 0;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  // NOT process.exit(): it discards pending stdout writes to a pipe, which
  // truncated batch output at the 64 KiB pipe buffer. Setting exitCode lets
  // Node flush and exit on its own.
  process.exitCode = main(process.argv.slice(2));
}
