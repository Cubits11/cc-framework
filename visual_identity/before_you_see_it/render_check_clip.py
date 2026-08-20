#!/usr/bin/env python3
"""Render Surface B: a ten-second silent clip of the canonical page's check.

    BYTES MATCH -> one character changes -> FAIL CLOSED -> restore -> BYTES MATCH

The clip is driven against the real page in a real browser and captured frame by
frame, so it cannot drift from the artifact it depicts. Nothing is re-staged,
mocked up, or animated separately: every digest on screen was computed by
WebCrypto during the capture.

Usage:
    python3 visual_identity/before_you_see_it/render_check_clip.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from render_film import find_ffmpeg  # noqa: E402  (path set above; shared helper)

PAGE = HERE.parents[1] / "visual_identity" / "canonical_page" / "index.html"
DURATION = 10.0

# (at second, action). "select" highlights the digit; "type" replaces it.
SCRIPT: tuple[tuple[float, str, str], ...] = (
    (2.00, "select", ""),
    (2.40, "type", "9"),
    (5.40, "select", ""),
    (5.80, "type", "1"),
)

SELECT_DIGIT = """
() => {
  const t = document.getElementById('doc');
  const i = t.value.indexOf('0.041') + 4;
  t.focus();
  t.setSelectionRange(i, i + 1);
}
"""

TYPE_DIGIT = """
(ch) => {
  const t = document.getElementById('doc');
  const s = t.selectionStart, e = t.selectionEnd;
  t.value = t.value.slice(0, s) + ch + t.value.slice(e);
  t.setSelectionRange(s + 1, s + 1);
  t.dispatchEvent(new Event('input', { bubbles: true }));
}
"""


def capture(frames_dir: Path, fps: int) -> int:
    from playwright.sync_api import sync_playwright

    total = round(DURATION * fps)
    with sync_playwright() as pw:
        launch: dict = {}
        root = Path("/opt/pw-browsers")
        for exe in sorted(root.glob("chromium-*/chrome-linux/chrome")):
            launch["executable_path"] = str(exe)
            break
        browser = pw.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
        page.goto(PAGE.as_uri(), wait_until="load")
        page.wait_for_function("document.getElementById('actual').textContent.length === 64")
        page.locator("#doc").scroll_into_view_if_needed()
        page.wait_for_timeout(200)

        head = page.locator("section:has(#doc) h3").bounding_box()
        box = page.locator("section:has(#doc) .check").bounding_box()
        pad = 34
        top = head["y"] - pad
        clip = {
            "x": int(box["x"] - pad) & ~1,
            "y": int(top) & ~1,
            "width": int(box["width"] + 2 * pad) & ~1,
            "height": int(box["y"] + box["height"] + 12 - top) & ~1,
        }
        print(f"  framing {clip['width']}x{clip['height']} at ({clip['x']},{clip['y']})")

        pending = list(SCRIPT)
        for i in range(total):
            now = i / fps
            while pending and pending[0][0] <= now:
                _, action, arg = pending.pop(0)
                if action == "select":
                    page.evaluate(SELECT_DIGIT)
                else:
                    page.evaluate(TYPE_DIGIT, arg)
                page.wait_for_timeout(30)
            page.screenshot(
                path=str(frames_dir / f"f{i:05d}.png"), clip=clip, animations="disabled"
            )
            if i % 60 == 0:
                print(f"  frame {i}/{total}", flush=True)
        browser.close()
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out-dir", type=Path, default=HERE / "renders")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out_dir / ".frames-check-clip"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    ffmpeg = find_ffmpeg()
    print(f"capturing the check at {args.fps}fps -> {frames_dir}")
    capture(frames_dir, args.fps)

    mp4 = args.out_dir / "check__bytes_match_fail_closed.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frames_dir / "f%05d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not args.keep_frames:
        shutil.rmtree(frames_dir)
    print(f"  {mp4.relative_to(HERE.parents[1])}  {mp4.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
