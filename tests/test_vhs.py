"""VHS / NTSC: the signal chain behaves like the formats' specifications, and its noise reaches AV1.

COMFY_ROOT=/path/to/ComfyUI python tests/test_vhs.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _harness

_harness.load()

import importlib
from dataclasses import replace

import numpy as np
import torch

vhs = importlib.import_module(f"{_harness.PACK_NAME}.vhs")
grain = importlib.import_module(f"{_harness.PACK_NAME}.grain")
av1enc = importlib.import_module(f"{_harness.PACK_NAME}.av1enc")
nodes = importlib.import_module(f"{_harness.PACK_NAME}.nodes")

H, W = 240, 400
CLEAN = vhs.Tape(
    composite=False,
    tape=False,
    saturation=1.0,
    jitter_us=0,
    edge_wave_us=0,
    head_switching=False,
    dropouts=0,
    luma_noise=0,
    chroma_noise=0,
)


def flat(rgb):
    return torch.tensor(rgb).view(1, 1, 1, 3).expand(1, H, W, 3).contiguous()


def sweep(mhz):
    """Luma stripes at ``mhz`` on the NTSC line (52.6 us across the width)."""
    x = torch.arange(W) / W * vhs.ACTIVE_US * 1e-6
    row = 0.5 + 0.35 * torch.sin(2 * np.pi * mhz * 1e6 * x)
    return row.view(1, 1, W, 1).expand(1, H, W, 3).contiguous()


def contrast(img):
    y = img[0, H // 4 : 3 * H // 4, W // 10 : -W // 10].mean(-1)
    return float(y.std())


def test_a_clean_chain_is_nearly_transparent():
    x = flat((0.2, 0.5, 0.8))
    y = vhs.apply(x, CLEAN)
    assert (y - x)[0, 20:-20, 30:-30].abs().max() < 0.01


def test_composite_keeps_flat_colour():
    x = flat((0.7, 0.3, 0.2))
    for decoder in ("notch", "comb"):
        y = vhs.apply(x, replace(CLEAN, composite=True, decoder=decoder))
        assert (y - x)[0, 20:-20, 40:-40].abs().mean() < 0.02, decoder


def test_luma_bandwidth_follows_the_format():
    """Stripes below the tape's luma bandwidth survive; above it they turn grey."""
    sp = replace(
        CLEAN,
        tape=True,
        luma_mhz=3.0,
        sharpen=0.0,
        chroma_delay_us=0.0,
        chroma_vertical=0.0,
        head_switching=False,
    )
    assert contrast(vhs.apply(sweep(1.0), sp)) > 0.7 * contrast(sweep(1.0))
    assert contrast(vhs.apply(sweep(4.0), sp)) < 0.35 * contrast(sweep(4.0))
    svhs = replace(sp, luma_mhz=4.6)
    assert contrast(vhs.apply(sweep(3.6), svhs)) > 1.5 * contrast(vhs.apply(sweep(3.6), sp))


def test_dot_crawl_moves_and_comb_reduces_it():
    """A saturated edge leaks chroma into luma (dot crawl), alternating frame to frame.

    Measured on the NTSC raster: the crawl's ~4-sample pattern averages away once a frame is scaled
    down to a small output, as it would on a small screen.
    """
    x = torch.zeros(1, H, W, 3)
    x[..., : W // 2, 0] = 0.8
    x[..., W // 2 :, 2] = 0.8
    raster = vhs.to_raster(x)[0]
    edge = slice(vhs.PAD + vhs.SAMPLES // 2 - 20, vhs.PAD + vhs.SAMPLES // 2 + 20)
    flat_red = slice(vhs.PAD + 100, vhs.PAD + 200)
    a, b = (vhs.composite(raster.clone(), f, "notch")[0] for f in (0, 1))
    assert (a - b)[40:-40, edge].abs().mean() > 0.01, "the crawl moves between frames"
    assert (a - b)[40:-40, flat_red].abs().mean() < 1e-3, "flat colour does not crawl"
    comb = vhs.composite(raster.clone(), 0, "comb")[0]
    ripple = lambda y: float((y[40:-40, edge][:, 1:] - y[40:-40, edge][:, :-1]).abs().mean())  # noqa: E731
    assert ripple(comb) < ripple(a), "a comb decoder separates colour better"


def test_chroma_lags_luma_on_tape():
    x = torch.zeros(1, H, W, 3)
    x[..., W // 2 :, :] = torch.tensor([0.8, 0.2, 0.2])
    t = replace(
        CLEAN,
        tape=True,
        luma_mhz=4.0,
        chroma_mhz=0.5,
        chroma_delay_us=0.5,
        sharpen=0,
        chroma_vertical=0,
    )
    y = vhs.apply(x, t)[0, H // 2]
    redness = y[:, 0] - y[:, 1]
    edge = int((redness > 0.3).nonzero()[0])
    lag_px = edge - W // 2
    expected = 0.5 / vhs.ACTIVE_US * W
    assert lag_px > expected * 0.5, (lag_px, expected)


def test_head_switching_only_at_the_bottom():
    x = torch.rand(1, H, W, 3)
    base = replace(
        CLEAN,
        tape=True,
        sharpen=0,
        chroma_delay_us=0,
        chroma_vertical=0,
        luma_mhz=6.0,
        chroma_mhz=6.0,
    )
    y0 = vhs.apply(x, base)
    y1 = vhs.apply(x, replace(base, head_switching=True))
    rows = (y1 - y0).abs().mean((0, 2, 3))
    assert rows[: int(H * 0.9)].max() < 1e-4 and rows[-3:].min() > 0.02


def test_frames_are_seeded_and_chunk_safe():
    t = vhs.PRESETS["VHS EP (worn)"][1]
    x = torch.rand(6, H, W, 3)
    whole = vhs.apply(x, t, bake_noise=True)
    parts = torch.cat(
        [
            vhs.apply(x[:3], t, frame_offset=0, bake_noise=True),
            vhs.apply(x[3:], t, frame_offset=3, bake_noise=True),
        ]
    )
    assert torch.allclose(whole, parts, atol=1e-5)
    assert not torch.allclose(whole, vhs.apply(x, replace(t, seed=9), bake_noise=True))


def test_crt_bulges_with_rounded_corners_and_streams_like_batch():
    x = torch.full((1, H, W, 3), 0.6)
    flat = vhs.crt(x, 0.0)
    assert torch.equal(flat, x)
    y = vhs.crt(x, 1.0, 0.6)[0].mean(-1)
    assert y[0, 0] < 0.01 and y[-1, -1] < 0.01, "black outside the tube's rounded corners"
    assert y[H // 2, W // 2] > 0.55, "the centre is untouched"
    assert y[H // 2, 2] > 0.2 and y[2, W // 2] > 0.2, (
        "the edge midpoints still show picture: it bulges out"
    )
    # Straight lines bow: a vertical line near the edge comes out curved.
    line = torch.zeros(1, H, W, 3)
    line[:, :, W // 8] = 1.0
    bent = vhs.crt(line, 1.0)[0].mean(-1)
    top, mid = int(bent[H // 8].argmax()), int(bent[H // 2].argmax())
    assert top > mid, "barrel distortion pulls the line inwards towards the corners"
    clip = torch.rand(4, H, W, 3)
    t = vhs.PRESETS["VHS SP"][1]
    batch = nodes.VHSTape.execute("VHS SP", 0.0, 0.0, "baked into pixels (exact, larger file)", True, 0,
                                  images=clip, crt_curvature=0.7).result[0]  # fmt: skip
    fx = nodes.VHSTape.execute(
        "VHS SP", 0.0, 0.0, "baked into pixels (exact, larger file)", True, 0, crt_curvature=0.7
    ).result[2]
    gen, _, _ = nodes._stream(clip, 24.0, "same as source", 180.0, fx)
    streamed = torch.stack([torch.from_numpy(f) for f in gen])
    assert (streamed - batch).abs().max() < 2e-4
    assert t is not None


def test_noise_table_is_valid_luma_and_chroma_grain():
    for name, (_, t) in vhs.PRESETS.items():
        table = vhs.noise_table(t, 1280, 720)
        assert grain.loads(grain.dumps(table)) == table, name
        p = table.entries[0].params
        assert p.scaling_points_cb and p.scaling_points_cr and not p.chroma_scaling_from_luma
        assert p.scaling_points_cb[0][1] > p.scaling_points_y[0][1], (
            "VHS chroma noise is stronger than luma"
        )
    assert vhs.noise_table(replace(CLEAN), 640, 360) is None


def test_vhs_node_to_encoder_with_av1_chroma_grain():
    import folder_paths

    folder_paths.set_output_directory(tempfile.mkdtemp(prefix="vhs out "))
    x = torch.rand(6, H, W, 3) * 0.5 + 0.25
    images, table = nodes.VHSTape.execute(
        "VHS SP", 0.3, 0.2, "AV1 grain (small file)", True, 1, images=x
    ).result[:2]
    assert images.shape == x.shape and table is not None
    res = nodes.AV1FilmGrainEncode.execute(
        images,
        24.0,
        "vhs/test",
        {"encoder": "svt-av1", "crf": 30, "preset": 10},
        {"grain_mode": "from VHS node", "grain": table},
        container="webm",
    )
    report = json.loads(res.result[1])
    assert report["grain_frames"] == report["frames"] == 6
    path = res.result[0].get_stream_source()
    on, off = av1enc.decode_rgb(path), av1enc.decode_rgb(path, apply_grain=False)
    d = on - off
    chroma = (d[..., 0] - d[..., 1]).std()
    assert chroma > 2 / 255, f"chroma noise must reach the decoded picture, got {chroma * 255:.2f}"
    _baked, none = nodes.VHSTape.execute(
        "VHS SP", 0.3, 0.2, "baked into pixels (exact, larger file)", True, 1, images=x
    ).result[:2]
    assert none is None
    try:
        nodes.build_settings(
            {"encoder": "svt-av1", "crf": 30, "preset": 10},
            {"grain_mode": "from VHS node", "grain": None},
            W,
            H,
        )
    except ValueError as exc:
        assert "baked" in str(exc)
    else:
        raise AssertionError("an empty VHS grain must be refused with a hint")


if __name__ == "__main__":
    raise SystemExit(_harness.run(dict(globals())))
