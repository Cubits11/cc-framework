#!/usr/bin/env node
/**
 * Independently recompute the primary E2 Field Test 001 quantities.
 *
 * This is intentionally small and dependency-free. It does not invoke a model;
 * it reads a completed observation JSONL and compares the independently derived
 * primary statistic and duplicate-column control with analyze.py's JSON output.
 */

import fs from "node:fs";

const STUDY_ID = "E2-FIELD-TEST-001";
const TOLERANCE = 1e-12;

function parseArgs(argv) {
  const args = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error("usage: recompute.mjs --rows observations.jsonl --items items.jsonl --analysis analysis.json --guardrail-a ID --guardrail-b ID [--expected-items 1097]");
    }
    args.set(key.slice(2), value);
  }
  for (const required of ["rows", "items", "analysis", "guardrail-a", "guardrail-b"]) {
    if (!args.has(required)) throw new Error(`missing --${required}`);
  }
  return {
    rows: args.get("rows"),
    items: args.get("items"),
    analysis: args.get("analysis"),
    guardrailA: args.get("guardrail-a"),
    guardrailB: args.get("guardrail-b"),
    expectedItems: Number(args.get("expected-items") ?? "1097"),
  };
}

function parseItems(path, expectedItems) {
  const items = new Map();
  const lines = fs.readFileSync(path, "utf8").split(/\r?\n/);
  lines.forEach((line, zeroBasedLine) => {
    if (!line.trim()) return;
    let item;
    try {
      item = JSON.parse(line);
    } catch {
      throw new Error(`items line ${zeroBasedLine + 1}: invalid JSON`);
    }
    if (typeof item.item_id !== "string" || typeof item.item_hash !== "string" || item.source_label !== 1) {
      throw new Error(`items line ${zeroBasedLine + 1}: invalid frozen item record`);
    }
    if (items.has(item.item_id)) throw new Error(`duplicate frozen item ${item.item_id}`);
    items.set(item.item_id, item.item_hash);
  });
  if (items.size !== expectedItems) throw new Error(`expected ${expectedItems} frozen items; found ${items.size}`);
  return items;
}

function parseRows(path, items, guardrailA, guardrailB) {
  const byItem = new Map();
  const lines = fs.readFileSync(path, "utf8").split(/\r?\n/);
  lines.forEach((line, zeroBasedLine) => {
    if (!line.trim()) return;
    let row;
    try {
      row = JSON.parse(line);
    } catch {
      throw new Error(`line ${zeroBasedLine + 1}: invalid JSON`);
    }
    if (row.study_id !== STUDY_ID || row.replicate_id !== 0) {
      throw new Error(`line ${zeroBasedLine + 1}: unexpected study_id or replicate_id`);
    }
    if (![guardrailA, guardrailB].includes(row.guardrail_id) || typeof row.item_id !== "string") {
      throw new Error(`line ${zeroBasedLine + 1}: unexpected guardrail or item id`);
    }
    if (!byItem.has(row.item_id)) byItem.set(row.item_id, new Map());
    const rows = byItem.get(row.item_id);
    if (rows.has(row.guardrail_id)) throw new Error(`duplicate row for ${row.item_id}/${row.guardrail_id}`);
    rows.set(row.guardrail_id, row);
  });
  if (byItem.size !== items.size || [...byItem.keys()].some((itemId) => !items.has(itemId))) {
    throw new Error("observation rows do not exactly match the frozen item manifest");
  }
  const valuesA = [];
  const valuesB = [];
  for (const [itemId, rows] of [...byItem.entries()].sort(([a], [b]) => a.localeCompare(b))) {
    if (rows.size !== 2 || !rows.has(guardrailA) || !rows.has(guardrailB)) {
      throw new Error(`unpaired item ${itemId}`);
    }
    const a = rows.get(guardrailA);
    const b = rows.get(guardrailB);
    if (a.item_hash !== b.item_hash || a.item_hash !== items.get(itemId)) {
      throw new Error(`item hash mismatch for ${itemId}`);
    }
    const aObserved = a.missingness_code === "observed";
    const bObserved = b.missingness_code === "observed";
    for (const [name, row, observed] of [[guardrailA, a, aObserved], [guardrailB, b, bObserved]]) {
      if (observed) {
        if (row.execution_status !== "success" || ![0, 1].includes(row.normalized_outcome)
          || row.reference_label !== 1 || row.normalizer_version !== "h.unsafe_pass.e2-field-test-001.v1") {
          throw new Error(`${itemId}/${name}: invalid observed row`);
        }
      } else if (row.normalized_outcome !== null || row.execution_status === "success" || row.reference_label !== 1) {
        throw new Error(`${itemId}/${name}: invalid missing row`);
      }
    }
    if (aObserved && bObserved) {
      valuesA.push(a.normalized_outcome);
      valuesB.push(b.normalized_outcome);
    }
  }
  if (!valuesA.length) throw new Error("no complete pairs");
  return { valuesA, valuesB, itemCount: byItem.size };
}

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function stats(valuesA, valuesB) {
  const pA = mean(valuesA);
  const pB = mean(valuesB);
  const p11 = mean(valuesA.map((value, index) => value * valuesB[index]));
  return { p_a: pA, p_b: pB, p_11: p11, delta: p11 - pA * pB };
}

function close(left, right) {
  return Math.abs(left - right) <= TOLERANCE;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const items = parseItems(args.items, args.expectedItems);
  const { valuesA, valuesB, itemCount } = parseRows(
    args.rows,
    items,
    args.guardrailA,
    args.guardrailB,
  );
  const independent = stats(valuesA, valuesB);
  const duplicate = stats(valuesA, valuesA);
  const expectedDuplicate = independent.p_a * (1 - independent.p_a);
  const supplied = JSON.parse(fs.readFileSync(args.analysis, "utf8"));
  const suppliedPrimary = supplied.primary;
  const suppliedDuplicate = supplied.duplicate_column_control;
  if (!suppliedPrimary || !suppliedDuplicate) throw new Error("analysis JSON lacks primary/control outputs");
  const primaryMatches = ["p_a", "p_b", "p_11", "delta"].every((key) => close(independent[key], suppliedPrimary[key]));
  const duplicateMatches = close(duplicate.delta, suppliedDuplicate.delta)
    && close(expectedDuplicate, suppliedDuplicate.expected)
    && close(duplicate.delta, expectedDuplicate);
  const result = {
    study_id: STUDY_ID,
    item_count: itemCount,
    complete_pairs: valuesA.length,
    primary: independent,
    duplicate_column_control: { delta: duplicate.delta, expected: expectedDuplicate },
    tolerance: TOLERANCE,
    primary_matches_analysis: primaryMatches,
    duplicate_control_matches_analysis: duplicateMatches,
    verdict: primaryMatches && duplicateMatches ? "PASS" : "DIVERGENCE",
  };
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (result.verdict !== "PASS") process.exitCode = 2;
}

try {
  main();
} catch (error) {
  process.stderr.write(`recompute: ${error.message}\n`);
  process.exitCode = 2;
}
