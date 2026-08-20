#!/usr/bin/env python3
"""Render `film.html` to a deterministic 15-second film.

The film is a pure function of time: the page exposes ``window.__seek(t)`` and
this script walks the timeline frame by frame, so two renders of the same commit
produce byte-identical frames. Nothing is recorded in real time.

Usage:
    python3 visual_identity/before_you_see_it/render_film.py --cut cc-framework
    python3 visual_identity/before_you_see_it/render_film.py --cut ghost-ark \
        --verdict path/to/verdict.json

`--verdict` takes a JSON object ``{"state": ..., "detail": ..., "color": ...}``
produced by a real verifier run. Without it the result card is rendered with an
on-screen ILLUSTRATION tag: the film does not display a verdict it was not given.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILM = HERE / "film.html"
WIDTH, HEIGHT = 1920, 1080


def _has_x264(binary: str) -> bool:
    try:
        out = subprocess.run(
            [binary, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "libx264" in out.stdout


def find_ffmpeg() -> str:
    """First ffmpeg on the box that can actually encode H.264.

    Playwright ships an ffmpeg built only for VP8 screen recording, so it is
    tried last and usually rejected here.
    """
    candidates: list[str] = []
    if os.environ.get("FFMPEG"):
        candidates.append(os.environ["FFMPEG"])
    on_path = shutil.which("ffmpeg")
    if on_path:
        candidates.append(on_path)
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    root = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers"))
    candidates += [str(p) for p in sorted(root.glob("ffmpeg-*/ffmpeg-linux"))]

    for candidate in candidates:
        if _has_x264(candidate):
            return candidate
    raise SystemExit(
        "no ffmpeg with libx264 found. `pip install imageio-ffmpeg`, or set "
        "FFMPEG=/path/to/ffmpeg (the Playwright bundle is VP8-only)."
    )


def find_chromium() -> str | None:
    root = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers"))
    for candidate in sorted(root.glob("chromium-*/chrome-linux/chrome")):
        return str(candidate)
    return None


def capture(frames_dir: Path, cut: str, fps: int, verdict: dict | None) -> int:
    from playwright.sync_api import sync_playwright

    total = round(15.0 * fps)
    url = f"{FILM.as_uri()}?capture=1&cut={cut}"
    launch: dict = {"args": ["--force-color-profile=srgb", "--font-render-hinting=none"]}
    exe = find_chromium()
    if exe:
        launch["executable_path"] = exe

    with sync_playwright() as pw:
        browser = pw.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)
        page.goto(url, wait_until="load")
        if verdict is not None:
            page.evaluate("v => { window.__VERDICT = v; }", verdict)
        page.wait_for_function("typeof window.__seek === 'function'")
        for i in range(total):
            page.evaluate("t => window.__seek(t)", i / fps)
            page.screenshot(path=str(frames_dir / f"f{i:05d}.png"), animations="disabled")
            if i % 120 == 0:
                print(f"  frame {i}/{total}", flush=True)
        browser.close()
    return total


def encode(ffmpeg: str, frames_dir: Path, out_dir: Path, cut: str, fps: int) -> list[Path]:
    stem = f"before_you_see_it__{cut}"
    mp4 = out_dir / f"{stem}.mp4"
    webm = out_dir / f"{stem}.webm"
    common = [ffmpeg, "-y", "-framerate", str(fps), "-i", str(frames_dir / "f%05d.png")]
    subprocess.run(
        [
            *common,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "19",
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
    subprocess.run(
        [
            *common,
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "0",
            "-crf",
            "32",
            "-pix_fmt",
            "yuv420p",
            "-row-mt",
            "1",
            str(webm),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return [mp4, webm]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cut", choices=["cc-framework", "ghost-ark", "cubits11"], default="cc-framework"
    )
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--out-dir", type=Path, default=HERE / "renders")
    ap.add_argument(
        "--poster-at",
        type=float,
        default=7.6,
        help="seconds; the poster frame is pulled from this moment",
    )
    ap.add_argument(
        "--verdict",
        type=Path,
        default=None,
        help="JSON file holding a real verifier result to display",
    )
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument(
        "--encode-only",
        action="store_true",
        help="reuse frames already on disk instead of re-capturing",
    )
    args = ap.parse_args(argv)

    verdict = json.loads(args.verdict.read_text()) if args.verdict else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out_dir / f".frames-{args.cut}"
    ffmpeg = find_ffmpeg()

    if args.encode_only:
        total = len(list(frames_dir.glob("f*.png")))
        if not total:
            raise SystemExit(f"--encode-only: no frames in {frames_dir}")
        print(f"reusing {total} frames in {frames_dir}")
    else:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True)
        print(f"capturing {args.cut} at {args.fps}fps -> {frames_dir}")
        total = capture(frames_dir, args.cut, args.fps, verdict)

    outputs = encode(ffmpeg, frames_dir, args.out_dir, args.cut, args.fps)
    poster_idx = min(total - 1, round(args.poster_at * args.fps))
    poster = args.out_dir / f"poster__{args.cut}.png"
    shutil.copyfile(frames_dir / f"f{poster_idx:05d}.png", poster)
    outputs.append(poster)

    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    for path in outputs:
        print(f"  {path.relative_to(HERE.parents[1])}  {path.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
