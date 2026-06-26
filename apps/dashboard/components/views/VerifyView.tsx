import { CheckCircle2, Upload, XCircle } from "lucide-react";
import type { ChangeEvent } from "react";
import { useEffect, useState } from "react";
import { verifyEnterpriseBundle } from "../../lib/merkle";
import type { ClientVerificationResult, EnterpriseBundle } from "../../lib/types";

export function VerifyView({
  bundle,
  onBundleLoaded,
}: {
  bundle?: EnterpriseBundle;
  onBundleLoaded: (bundle: EnterpriseBundle) => void;
}) {
  const [result, setResult] = useState<ClientVerificationResult | undefined>();
  const [error, setError] = useState<string | undefined>();

  useEffect(() => {
    let mounted = true;
    setError(undefined);
    if (!bundle) {
      setResult(undefined);
      return;
    }
    verifyEnterpriseBundle(bundle)
      .then((next) => {
        if (mounted) {
          setResult(next);
        }
      })
      .catch((err: unknown) => {
        if (mounted) {
          setError(err instanceof Error ? err.message : "client verification failed");
        }
      });
    return () => {
      mounted = false;
    };
  }, [bundle]);

  const onFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setError(undefined);
    try {
      const parsed = JSON.parse(await readFileText(file)) as EnterpriseBundle;
      onBundleLoaded(parsed);
    } catch (err) {
      setError(err instanceof Error ? err.message : "could not read bundle JSON");
    }
  };

  return (
    <div className="panel">
      <div className="section-head">
        <div>
          <h2>Verify</h2>
          <p>Merkle inclusion and consistency proofs are recomputed locally from the uploaded bundle.</p>
        </div>
        {bundle ? <span className="run-chip">{bundle.bundle_id}</span> : null}
      </div>

      <div className="verify-panel">
        <div className="upload-row">
          <label className="file-label">
            <Upload size={17} />
            <span>Evidence bundle JSON</span>
            <input accept="application/json,.json" aria-label="Evidence bundle JSON" onChange={onFileChange} type="file" />
          </label>
          {error ? <span className="hash">{error}</span> : null}
        </div>

        {result ? (
          <>
            <div className="status-grid">
              <Status label="Client verification" ok={result.ok} value={result.ok ? "passed" : "failed"} />
              <Status label="Inclusion proofs" ok={result.inclusionOk} value={`${result.checkedRecords} records`} />
              <Status label="Consistency proof" ok={result.consistencyOk} value={result.consistencyOk ? "append-only" : "invalid"} />
              <Status label="Leaf hashes" ok={result.leafHashesOk} value={result.leafHashesOk ? "unchanged" : "changed"} />
            </div>
            <div>
              <span className="node-kicker">Merkle root</span>
              <p className="hash">{result.rootHash}</p>
            </div>
            {result.errors.length ? <p className="hash">{result.errors.join("; ")}</p> : null}
          </>
        ) : (
          <div className="empty">Awaiting bundle data.</div>
        )}
      </div>
    </div>
  );
}

function readFileText(file: File): Promise<string> {
  if (typeof file.text === "function") {
    return file.text();
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("could not read file"));
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.readAsText(file);
  });
}

function Status({ label, ok, value }: { label: string; ok: boolean; value: string }) {
  return (
    <div className={`status ${ok ? "ok" : "fail"}`}>
      <span>{label}</span>
      <strong>
        {ok ? <CheckCircle2 aria-hidden="true" size={16} /> : <XCircle aria-hidden="true" size={16} />} {value}
      </strong>
    </div>
  );
}
