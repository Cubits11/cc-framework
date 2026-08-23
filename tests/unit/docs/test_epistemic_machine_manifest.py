"""README's epistemic-machine section must match the machine's manifest.

The logical machine has six rows (the README table); the cinematic machine
has five panels (the poster/gif/mp4). The compression between them is
deliberate and labeled — the bridge caption at README.md's figure. Before
this manifest existed, the panel titles' only source of truth was pixels,
so a rename in the table (or an embed without the caption) could silently
reopen the five/six drift the caption closes. These tests pin README.md to
docs/assets/epistemic_machine.manifest.json instead.

What these tests do NOT establish: they compare committed text with
committed text. They cannot read the rendered pixels, so a regenerated
poster whose painted titles diverge from the manifest would pass here —
regenerating the media requires re-checking the manifest's poster fields
by eye, and the manifest says so.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
README = (ROOT / "README.md").read_text(encoding="utf-8")
MANIFEST = json.loads(
    (ROOT / "docs" / "assets" / "epistemic_machine.manifest.json").read_text(encoding="utf-8")
)


def test_manifest_shape() -> None:
    assert len(MANIFEST["logical_rows"]) == 6
    assert len(MANIFEST["cinematic_panels"]) == 5
    assert [r["n"] for r in MANIFEST["logical_rows"]] == list(range(1, 7))
    assert [p["n"] for p in MANIFEST["cinematic_panels"]] == list(range(1, 6))


def test_compression_covers_every_row_exactly_once() -> None:
    represented = [
        row for panel in MANIFEST["cinematic_panels"] for row in panel["represents_rows"]
    ]
    assert sorted(represented) == list(range(1, 7))
    merged = [p for p in MANIFEST["cinematic_panels"] if len(p["represents_rows"]) > 1]
    assert [p["n"] for p in merged] == [4], "the only labeled merge is bounds+witnesses in panel 4"
    assert merged[0]["represents_rows"] == [4, 5]


def test_readme_table_matches_logical_rows() -> None:
    for row in MANIFEST["logical_rows"]:
        cell = f"| **{row['name']}** | {row['contributes']} |"
        assert cell in README, f"README table lost logical row: {row['name']}"
    positions = [README.index(f"| **{row['name']}**") for row in MANIFEST["logical_rows"]]
    assert positions == sorted(positions), "README table rows out of order"


def test_readme_alt_text_names_every_panel() -> None:
    alt = re.search(
        r'<img src="docs/assets/epistemic-machine\.png"[^>]*alt="([^"]+)"',
        README,
    )
    assert alt, "epistemic-machine poster embed (with alt text) missing"
    alt_text = alt.group(1)
    positions = []
    for panel in MANIFEST["cinematic_panels"]:
        assert panel["alt_title"] in alt_text, f"alt text lost panel: {panel['alt_title']}"
        positions.append(alt_text.index(panel["alt_title"]))
    assert positions == sorted(positions), "alt-text panels out of order"


def test_readme_carries_the_bridge_caption() -> None:
    normalized = re.sub(r"<[^>]+>", "", README)
    normalized = re.sub(r"\s+", " ", normalized)
    bridge = re.sub(r"\s+", " ", MANIFEST["bridge_caption"])
    assert bridge in normalized, (
        "the bridge caption (six rows, five panels, same machine) must accompany the media embed"
    )
    assert (
        f"{len(MANIFEST['logical_rows'])} rows".capitalize().replace("6 rows", "Six rows")
        in MANIFEST["bridge_caption"]
    ), "caption numerals drifted from the manifest"


def test_media_files_exist() -> None:
    for media in MANIFEST["media"]:
        assert (ROOT / media["path"]).exists(), f"missing media: {media['path']}"
