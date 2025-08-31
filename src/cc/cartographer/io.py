import json
import os

import numpy as np
import requests
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_scores(cfg, n=None):
    # TODO: replace this RNG with your real two-world outputs.
    rng = np.random.default_rng(1337)
    k = n or 200
    return {
        "A0": rng.normal(0, 1, k),
        "A1": rng.normal(0.4, 1, k),
        "B0": rng.normal(0, 1, k),
        "B1": rng.normal(0.3, 1, k),
        "Comp0": None,
        "Comp1": None,
        "rocA": np.stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50)], 1),
        "rocB": np.stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50)], 1),
    }


def post_github_comment(ctx_json, body):
    try:
        ctx = json.loads(ctx_json)
        repo = ctx["repository"]["full_name"]
        num = ctx["issue"]["number"]
        url = f"https://api.github.com/repos/{repo}/issues/{num}/comments"
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            return
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        requests.post(url, headers=headers, json={"body": body})
    except Exception:
        # Avoid hard failing in CI comment posting
        pass
