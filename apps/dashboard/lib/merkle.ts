import type {
  ClientVerificationResult,
  ConsistencyProof,
  EnterpriseBundle,
  InclusionProof,
} from "./types";

const HASH_ALGORITHM = "sha256-rfc6962";
const EMPTY_ROOT_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

export async function verifyEnterpriseBundle(
  bundle: EnterpriseBundle,
): Promise<ClientVerificationResult> {
  const errors: string[] = [];
  const verification = bundle.verification;

  if (verification.hash_algorithm !== HASH_ALGORITHM) {
    errors.push(`unsupported hash algorithm: ${verification.hash_algorithm}`);
  }

  const leafResults = await Promise.all(
    verification.records.map(async (record, index) => {
      const actual = await leafHash(record, verification.canonical_records?.[index]);
      return actual === verification.leaf_hashes[index];
    }),
  );
  const leafHashesOk = leafResults.every(Boolean);
  if (!leafHashesOk) {
    errors.push("one or more record leaf hashes changed");
  }

  const inclusionResults = await Promise.all(
    verification.records.map((record, index) =>
      verifyInclusion(
        record,
        verification.inclusion_proofs[index],
        verification.trusted_root,
        verification.canonical_records?.[index],
      ),
    ),
  );
  const inclusionOk = inclusionResults.every(Boolean);
  if (!inclusionOk) {
    errors.push("one or more inclusion proofs failed");
  }

  const consistencyOk = await verifyConsistency(verification.consistency_proof);
  if (!consistencyOk) {
    errors.push("consistency proof failed");
  }

  return {
    ok: errors.length === 0,
    inclusionOk,
    consistencyOk,
    leafHashesOk,
    checkedRecords: verification.records.length,
    rootHash: verification.trusted_root,
    treeSize: verification.tree_size,
    errors,
  };
}

export async function verifyInclusion(
  record: unknown,
  proof: InclusionProof | undefined,
  rootHash?: string,
  canonicalRecord?: string,
): Promise<boolean> {
  try {
    if (!proof || proof.hash_algorithm !== HASH_ALGORITHM) {
      return false;
    }
    const trustedRoot = rootHash ?? proof.root_hash;
    if (rootHash !== undefined && proof.root_hash !== rootHash) {
      return false;
    }
    if (proof.tree_size <= 0 || proof.record_id < 0 || proof.record_id >= proof.tree_size) {
      return false;
    }
    if ((await leafHash(record, canonicalRecord)) !== proof.leaf_hash) {
      return false;
    }
    const expectedSides = expectedInclusionSides(proof.record_id, proof.tree_size);
    const actualSides = proof.proof.map((step) => step.side);
    if (JSON.stringify(expectedSides) !== JSON.stringify(actualSides)) {
      return false;
    }
    let current = proof.leaf_hash;
    for (const step of proof.proof) {
      if (step.side === "left") {
        current = await hashNodeHex(step.hash, current);
      } else if (step.side === "right") {
        current = await hashNodeHex(current, step.hash);
      } else {
        return false;
      }
    }
    return current === trustedRoot;
  } catch {
    return false;
  }
}

export async function verifyConsistency(
  proof: ConsistencyProof | undefined,
  oldRoot?: string,
  newRoot?: string,
): Promise<boolean> {
  try {
    if (!proof || proof.hash_algorithm !== HASH_ALGORITHM) {
      return false;
    }
    const trustedOld = oldRoot ?? proof.old_root;
    const trustedNew = newRoot ?? proof.new_root;
    if (proof.old_size < 0 || proof.new_size < proof.old_size) {
      return false;
    }
    if (oldRoot !== undefined && proof.old_root !== oldRoot) {
      return false;
    }
    if (newRoot !== undefined && proof.new_root !== newRoot) {
      return false;
    }
    if (proof.old_size === proof.new_size) {
      return (
        proof.old_root === proof.new_root &&
        proof.old_root === trustedOld &&
        proof.new_root === trustedNew &&
        proof.proof.length === 0
      );
    }
    if (proof.old_size === 0) {
      return proof.old_root === trustedOld && trustedOld === EMPTY_ROOT_HASH && proof.proof.length === 0;
    }

    let fn = proof.old_size - 1;
    let sn = proof.new_size - 1;
    const hashes = proof.proof.map(hexToBytes);

    while ((fn & 1) === 1) {
      fn >>= 1;
      sn >>= 1;
    }

    let oldHash: Uint8Array;
    let newHash: Uint8Array;
    if (fn === 0) {
      oldHash = hexToBytes(proof.old_root);
      newHash = hexToBytes(proof.old_root);
    } else {
      const first = hashes.shift();
      if (!first) {
        return false;
      }
      oldHash = first;
      newHash = first;
    }

    while (fn !== 0) {
      if ((fn & 1) === 1) {
        const sibling = hashes.shift();
        if (!sibling) {
          return false;
        }
        oldHash = await hashNodeBytes(sibling, oldHash);
        newHash = await hashNodeBytes(sibling, newHash);
      } else if (fn < sn) {
        const sibling = hashes.shift();
        if (!sibling) {
          return false;
        }
        newHash = await hashNodeBytes(newHash, sibling);
      }
      fn >>= 1;
      sn >>= 1;
    }

    while (sn !== 0) {
      const sibling = hashes.shift();
      if (!sibling) {
        return false;
      }
      newHash = await hashNodeBytes(newHash, sibling);
      sn >>= 1;
    }

    return bytesToHex(oldHash) === trustedOld && bytesToHex(newHash) === trustedNew && hashes.length === 0;
  } catch {
    return false;
  }
}

export async function leafHash(record: unknown, canonicalRecord?: string): Promise<string> {
  const payload = canonicalRecord === undefined ? encodeCanonical(record) : new TextEncoder().encode(canonicalRecord);
  const bytes = new Uint8Array(payload.length + 1);
  bytes[0] = 0;
  bytes.set(payload, 1);
  return bytesToHex(await sha256(bytes));
}

function expectedInclusionSides(recordId: number, treeSize: number): Array<"left" | "right"> {
  function rec(start: number, end: number, index: number): Array<"left" | "right"> {
    const size = end - start;
    if (size === 1) {
      return [];
    }
    const split = largestPowerOfTwoLessThan(size);
    const mid = start + split;
    if (index < mid) {
      return [...rec(start, mid, index), "right"];
    }
    return [...rec(mid, end, index), "left"];
  }
  return rec(0, treeSize, recordId);
}

function largestPowerOfTwoLessThan(n: number): number {
  return 1 << (Math.ceil(Math.log2(n)) - 1);
}

async function hashNodeHex(left: string, right: string): Promise<string> {
  return bytesToHex(await hashNodeBytes(hexToBytes(left), hexToBytes(right)));
}

async function hashNodeBytes(left: Uint8Array, right: Uint8Array): Promise<Uint8Array> {
  const bytes = new Uint8Array(1 + left.length + right.length);
  bytes[0] = 1;
  bytes.set(left, 1);
  bytes.set(right, 1 + left.length);
  return sha256(bytes);
}

async function sha256(bytes: Uint8Array): Promise<Uint8Array> {
  const payload = new Uint8Array(bytes.byteLength);
  payload.set(bytes);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", payload as BufferSource);
  return new Uint8Array(digest);
}

function encodeCanonical(value: unknown): Uint8Array {
  return new TextEncoder().encode(stableStringify(value));
}

function stableStringify(value: unknown): string {
  if (value === null || typeof value === "number" || typeof value === "boolean" || typeof value === "string") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(",")}]`;
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record).filter((key) => record[key] !== undefined).sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`).join(",")}}`;
  }
  return "null";
}

function hexToBytes(hex: string): Uint8Array {
  if (hex.length % 2 !== 0) {
    throw new Error("invalid hex string");
  }
  const bytes = new Uint8Array(hex.length / 2);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return bytes;
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}
