"""Compression benchmark: synthesised grain vs a clean encode vs grain baked into the pixels.

COMFY_ROOT=/path/to/ComfyUI python tests/benchmark.py [--size 1280x720] [--frames 48] [--crf 28] [--preset 6]

Everything is measured on this machine: file size, bitrate, whether the bitstream carries film grain
parameters (read back through dav1d), and the grain a viewer sees -- the standard deviation of the
decoded luma over a flat region of a clip that has no noise of its own, in 8-bit code values.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _harness

# Read our arguments before the harness replaces sys.argv for ComfyUI.
_ARGV = sys.argv[1:]
_harness.load()

import importlib

import clip
import numpy as np

grain = importlib.import_module(f"{_harness.PACK_NAME}.grain")
av1enc = importlib.import_module(f"{_harness.PACK_NAME}.av1enc")


def decoded_flat_noise(path: str, flat, frames: int = 8) -> float:
    """Std of decoded luma over the flat region, per frame, averaged (what a viewer sees)."""
    stds = []
    for frame, _ in av1enc.decode(path, limit=frames):
        y = frame.reformat(format="gray16le").to_ndarray().astype(np.float32) / 257.0
        stds.append(float(y[flat].std()))
    return float(np.mean(stds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--crf", type=int, default=28)
    parser.add_argument("--preset", type=int, default=6)
    a = parser.parse_args(_ARGV)
    w, h = (int(v) for v in a.size.split("x"))
    flat = clip.regions(h, w)["flat"]
    tmp = tempfile.mkdtemp(prefix="av1grain_bench_")

    external = os.path.join(tmp, "external grain.tbl")
    with open(external, "w") as fh:  # a table file on disk, read back as a user's would be
        fh.write(grain.dumps(grain.preset("Coarse 16mm", w, h)))
    with open(external) as fh:
        external_table = grain.loads(fh.read())

    cases = [
        ("Clean AV1", av1enc.Grain(), None),
        ("AV1 automatic FGS (film-grain=12)", av1enc.Grain("estimate", estimation_level=12), None),
        (
            "AV1 photon noise (ISO 3200, phone-size sensor)",
            av1enc.Grain("table", table=grain.dumps(grain.preset("High ISO 3200", w, h))),
            None,
        ),
        (
            "AV1 external FGS table (Coarse 16mm)",
            av1enc.Grain("table", table=grain.dumps(external_table)),
            None,
        ),
        (
            "AV1 Vintage Anime preset",
            av1enc.Grain("table", table=grain.dumps(grain.preset("Vintage Anime", w, h))),
            None,
        ),
        ("AV1 baked grain (matched to Coarse 16mm)", av1enc.Grain(), "baked"),
    ]
    rows = []
    target = None
    try:
        for name, g, special in cases:
            source = clip.frames(a.frames, h, w)
            if special == "baked":
                rng = np.random.default_rng(1)
                sigma = target / 255.0

                def baked(frames=source, sigma=sigma, rng=rng):
                    for f in frames:
                        yield np.clip(
                            f + rng.normal(0, sigma, f.shape[:2])[..., None], 0, 1
                        ).astype(np.float32)

                source = baked()
            settings = av1enc.Settings(crf=a.crf, preset=a.preset, grain=g)
            out = os.path.join(tmp, f"{len(rows)}.mkv")
            t0 = time.perf_counter()
            r = av1enc.encode(source, out, settings, size=(w, h), log=lambda _m: None)
            seconds = time.perf_counter() - t0
            noise = decoded_flat_noise(out, flat)
            if name.startswith("AV1 external"):
                target = noise  # bake the same amount of grain into the pixels for the last row
            rows.append((name, r, noise, seconds))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    clean_size = rows[0][1].file_size
    print(
        f"\nclean synthetic clip {w}x{h}, {a.frames} frames at 24 fps, SVT-AV1 crf {a.crf} preset {a.preset}\n"
    )
    print(
        "| Encoding method | File size | vs clean | Bitrate | Grain params in bitstream | Decoded grain (std, 8-bit) | Encode time |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, r, noise, seconds in rows:
        print(
            f"| {name} | {r.file_size / 1000:.1f} kB | {r.file_size / clean_size:.2f}x | {r.bitrate_kbps:.0f} kb/s "
            f"| {r.grain_frames}/{r.verified_frames} frames | {noise:.2f} | {seconds:.1f} s |"
        )


if __name__ == "__main__":
    main()
