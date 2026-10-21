"""The CRT display: beam scanlines, phosphor mask, brightness, persistence, and the streamed chain.

COMFY_ROOT=/path/to/ComfyUI python tests/test_crt.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _harness

_harness.load()

import importlib
from dataclasses import replace

import torch

crt = importlib.import_module(f"{_harness.PACK_NAME}.crt")
nodes = importlib.import_module(f"{_harness.PACK_NAME}.nodes")

H, W = 240, 320
BARE = crt.Display(mask="none", glow=0.0, halation=0.0, convergence=0.0, lines=60, scanlines=1.0)


def flat(level, n=1, h=H, w=W):
    return torch.full((n, h, w, 3), float(level))


def test_shape_range_and_black():
    for name, (_, d) in crt.PRESETS.items():
        out, last = crt.apply(torch.rand(2, H, W, 3), replace(d, curvature=0.4))
        assert out.shape == (2, H, W, 3) and out.dtype == torch.float32, name
        assert float(out.min()) >= 0 and float(out.max()) <= 1, name
        assert last.shape == (3, H, W), name
        assert float(crt.apply(flat(0.0), d)[0].max()) < 1e-6, f"{name}: black must stay black"


def test_average_brightness_is_kept():
    # Mask and lines darken a tube; it is brought back up so a mid-grey screen stays mid-grey.
    for name, (_, d) in crt.PRESETS.items():
        out = crt.apply(flat(0.5), d)[0]
        lin = lambda x: float(((x + 0.055) / 1.055).pow(2.4).mean())  # noqa: E731
        ratio = lin(out) / lin(flat(0.5))
        assert 0.8 < ratio < 1.15, (name, ratio)


def test_scanlines_have_the_line_pitch_and_dark_gaps():
    rows = crt.apply(flat(0.3), BARE)[0][0, :, W // 2, 1]
    rows = ((rows + 0.055) / 1.055).pow(2.4)  # light, not code values
    assert crt.line_count(BARE, H) == 60
    period = H // 60
    assert torch.allclose(rows[period : 2 * period], rows[2 * period : 3 * period], atol=1e-4)
    assert float(rows.min()) < 0.65 * float(rows.max()), "gaps between lines on a dim screen"


def test_bright_lines_bloom_into_their_gaps():
    def depth(level):
        rows = crt.apply(flat(level), BARE)[0][0, 40:200, W // 2, 1]
        return float((rows.max() - rows.min()) / rows.max())

    assert depth(0.15) > depth(0.9) + 0.1, (depth(0.15), depth(0.9))


def test_line_count_snaps_to_whole_rows():
    d = crt.Display(lines=480)
    assert crt.line_count(d, 1248) == 416  # 3 rows a line, not 2.6
    assert crt.line_count(d, 960) == 480
    assert crt.line_count(replace(d, lines=0), 1248) == 1248


def test_masks_repeat_at_the_triad_pitch():
    for kind in ("aperture grille", "slot mask", "shadow mask"):
        m = crt.phosphor_mask(kind, 40.0, H, W, "cpu")  # 8 px triads at 320 wide
        assert m.shape == (3, H, W), kind
        row = m[1, H // 2 + 1]
        assert torch.allclose(row[:-8], row[8:], atol=0.05) or kind != "aperture grille", kind
        # Each gun's phosphors sit a third of a triad from the next.
        r, g = m[0].mean(0), m[1].mean(0)
        assert int(r[:8].argmax()) != int(g[:8].argmax()), kind
    stripes = crt.phosphor_mask("aperture grille", 40.0, H, W, "cpu")
    assert float((stripes[:, 0] - stripes[:, H // 2]).abs().max()) < 1e-6, "a grille has no breaks"
    slots = crt.phosphor_mask("slot mask", 40.0, H, W, "cpu")
    assert float(slots[1].std(0).max()) > 0.05, "a slot mask is broken vertically"


def test_persistence_leaves_a_trail():
    frames = torch.zeros(4, H, W, 3)
    for i in range(4):
        frames[i, 100:140, 40 + 60 * i : 80 + 60 * i] = 1.0
    d = replace(BARE, scanlines=0.0, persistence=0.6)
    trail = crt.apply(frames, d)[0]
    none = crt.apply(frames, replace(d, persistence=0.0))[0]
    assert float(trail[3, 100:140, 40:80].mean()) > 0.2, "the first spot still glows"
    assert float(none[3, 100:140, 40:80].mean()) < 1e-3


def test_hum_bar_rolls_down_the_screen():
    d = replace(BARE, scanlines=0.0, hum_bar=1.0, hum_speed=0.05)
    x = flat(0.6, n=3)
    rows = crt.apply(x, d)[0][..., W // 2, 1]
    darkest = [int(r.argmin()) for r in rows]
    step = 0.05 * H
    assert abs(darkest[1] - darkest[0] - step) <= 2 and abs(darkest[2] - darkest[1] - step) <= 2, (
        darkest
    )
    assert float(rows[0].min()) < 0.85 * float(rows[0].max()), "a visible band"
    later = crt.apply(x[:1], d, frame_offset=1)[0][0, :, W // 2, 1]
    assert torch.allclose(later, rows[1], atol=1e-5), (
        "the band follows the frame number, not the chunk"
    )
    assert float(crt.apply(x, replace(d, hum_bar=0.0))[0].std()) < 1e-5


def test_classic_scanlines_are_just_lines():
    d = crt.PRESETS["classic scanlines"][1]
    out = crt.apply(flat(0.4, h=480, w=640), replace(d, convergence=0.0))[0][0]
    assert float(out[:, 200:440].std(1).max()) < 1e-3, "no mask: every row is flat across"
    rows = ((out[:, 320, 1] + 0.055) / 1.055).pow(2.4)
    assert float(rows.min()) < 0.4 * float(rows.max()), "deep gaps between the lines"


def test_chain_matches_node_across_chunks():
    torch.manual_seed(1)
    x = torch.rand(20, H, W, 3) * 0.8
    args = ("arcade (240p)", 1.0, 1.0, 1.0, 0.3, 0.5, 0.7)  # hum bar: frame numbers must line up
    batch = nodes.CRTDisplay.execute(*args, images=x).result[0]
    fx = nodes.CRTDisplay.execute(*args).result[1]
    assert fx.describe() == ["crt"]
    gen, _, total = nodes._stream(x, 24.0, "same as source", 180.0, fx)
    streamed = torch.stack([torch.from_numpy(f) for f in gen])
    assert total == 20
    assert float((streamed - batch).abs().max()) < 2e-4


if __name__ == "__main__":
    raise SystemExit(_harness.run(dict(globals())))
