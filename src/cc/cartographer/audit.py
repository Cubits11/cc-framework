import hashlib
import json
import os
import time


def make_record(cfg, JA, JA_ci, JB, JB_ci, Jc, Jc_ci, CC, Dadd, decision, figures):
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg,
        "metrics": {
            "J_A": round(JA, 4),
            "CI_J_A": list(JA_ci),
            "J_B": round(JB, 4),
            "CI_J_B": list(JB_ci),
            "J_comp": round(Jc, 4),
            "CI_J_comp": list(Jc_ci),
            "CC_max": round(CC, 4),
            "Delta_add": round(Dadd, 4),
        },
        "decision": decision,
        "figures": figures,
    }


def append_jsonl(path, rec):
    prev = None
    if os.path.exists(path):
        with open(path, "rb") as f:
            try:
                last = f.read().strip().splitlines()[-1]
                prev = json.loads(last).get("sha256")
            except Exception:
                prev = None
    rec["prev_sha256"] = prev
    sha = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    rec["sha256"] = sha
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return sha


def verify_chain(path):
    with open(path) as f:
        prev = None
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            exp = obj.get("sha256")
            material = {k: v for k, v in obj.items() if k != "sha256"}
            got = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
            if got != exp:
                raise RuntimeError(f"Line {i}: SHA mismatch")
            if obj.get("prev_sha256") != prev:
                raise RuntimeError(f"Line {i}: chain break")
            prev = exp
