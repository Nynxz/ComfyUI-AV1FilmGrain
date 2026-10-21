"""Streaming effects through the encoder: the same frames as the batch nodes, at a fraction of the memory.

COMFY_ROOT=/path/to/ComfyUI python tests/test_stream.py
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

import numpy as np
import torch

fxchain = importlib.import_module(f"{_harness.PACK_NAME}.fxchain")
av1enc = importlib.import_module(f"{_harness.PACK_NAME}.av1enc")
nodes = importlib.import_module(f"{_harness.PACK_NAME}.nodes")

H, W, N = 144, 256, 20
LOOK = ({"look": "warm print", "strength": 1.0}, "none", 1.0, 0.35, 0.15, 0.3, 0.25)
DAMAGE = (24.0, "Super 8", 0.8, 1.0, 1.0, 0.5, 0.5, 0.5, True, "none", 3)
VHS = ("VHS EP (worn)", 0.3, 0.3, "baked into pixels (exact, larger file)", True, 5)
ENC = {"encoder": "svt-av1", "crf": 30, "preset": 10}


def clip(n=N):
    torch.manual_seed(0)
    return torch.rand(n, H, W, 3) * 0.6 + 0.2


def test_retime_plan_follows_the_shutter():
    plan = fxchain.retime_plan(48, 48.0, 24.0, 180.0)
    assert len(plan) == 24 and plan[0] == (0, 1) and plan[5] == (10, 11), (
        "180 deg at 24 fps spans 1/48 s"
    )
    assert fxchain.retime_plan(48, 48.0, 24.0, 0.0)[5] == (10,), "0 deg just picks frames"
    assert fxchain.retime_plan(60, 60.0, 24.0, 180.0)[1] in ((2, 3), (3,), (2, 3, 4)), (
        "60 -> 24 averages ~1.25 frames"
    )
    assert len(fxchain.retime_plan(383, 48.0, 24.0, 180.0)) == 191


def test_retime_averages_in_linear_light():
    a, b = torch.zeros(1, 4, 4, 3), torch.ones(1, 4, 4, 3)
    blended = next(fxchain.frames(torch.cat([a, b]), [(0, 1)]))
    # The mean of black and white light is 0.5 linear = 0.735 sRGB, not 0.5.
    assert abs(float(blended.mean()) - 0.7354) < 0.002


def test_chain_gives_the_batch_nodes_frames():
    x = clip()
    batch = nodes.FilmLook.execute(*LOOK, images=x).result[0]
    batch = nodes.FilmDamage.execute(*DAMAGE, images=batch).result[0]
    batch = nodes.VHSTape.execute(*VHS, images=batch).result[0]

    fx = nodes.FilmLook.execute(*LOOK).result[2]
    fx = nodes.FilmDamage.execute(*DAMAGE, fx=fx).result[1]
    fx = nodes.VHSTape.execute(*VHS, fx=fx).result[2]
    assert fx.describe() == ["look", "damage", "vhs"]
    gen, fps, total = nodes._stream(x, 24.0, "same as source", 180.0, fx)
    streamed = torch.stack([torch.from_numpy(f) for f in gen])
    assert total == N and fps == 24.0
    assert (streamed - batch).abs().max() < 2e-4, float((streamed - batch).abs().max())


def test_streaming_nodes_skip_the_frames():
    look = nodes.FilmLook.execute(*LOOK).result
    assert look[0] is None and look[1] is None and look[2].steps
    vhs_out = nodes.VHSTape.execute("VHS SP", 0.0, 0.0, "AV1 grain (small file)", True, 0).result
    assert vhs_out[0] is None and vhs_out[1] is not None, "the grain table needs no frames"


def test_encode_streams_effects_and_retimes():
    import folder_paths

    folder_paths.set_output_directory(tempfile.mkdtemp(prefix="stream out "))
    x = clip(24)
    fx = nodes.FilmLook.execute(*LOOK).result[2]
    fx = nodes.FilmDamage.execute(*DAMAGE, fx=fx).result[1]
    res = nodes.AV1FilmGrainEncode.execute(
        x, 48.0, "stream/t", ENC, {"grain_mode": "preset", "preset": "Fine 35mm", "intensity": 0.6, "chroma_grain": False},
        effects=fx, frame_rate="24 (film)", shutter_angle=180.0,
    )  # fmt: skip
    report = json.loads(res.result[1])
    assert report["frames"] == 12 and abs(report["fps"] - 24.0) < 1e-6
    assert report["grain_frames"] == 12
    decoded = av1enc.decode_rgb(res.result[0].get_stream_source(), apply_grain=False)
    assert decoded.shape[0] == 12


def test_preview_uses_the_chain():
    x = clip()
    fx = nodes.FilmLook.execute(*LOOK).result[2]
    with_grain, without, _ = nodes.AV1FilmGrainPreview.execute(
        x,
        4,
        3,
        24.0,
        ENC,
        {"grain_mode": "preset", "preset": "Fine 35mm", "intensity": 1.0, "chroma_grain": False},
        effects=fx,
    ).result
    plain = nodes.AV1FilmGrainPreview.execute(
        x,
        4,
        3,
        24.0,
        ENC,
        {"grain_mode": "preset", "preset": "Fine 35mm", "intensity": 1.0, "chroma_grain": False},
    ).result[1]
    assert with_grain.shape == (3, H, W, 3)
    assert (without - plain).abs().mean() > 0.01, "the look must be in the preview"


def test_share_copy_is_h264_with_the_grain_and_fits_its_limit():
    import av

    folder_paths = importlib.import_module("folder_paths")
    folder_paths.set_output_directory(tempfile.mkdtemp(prefix="share out "))
    x = clip(24)
    res = nodes.AV1FilmGrainEncode.execute(
        x, 24.0, "share/t", ENC, {"grain_mode": "preset", "preset": "Vintage Anime", "intensity": 1.0, "chroma_grain": False},
        share_copy="Discord (10 MB)",
    )  # fmt: skip
    data = json.loads(res.result[1])
    share = data["share_copy"]["path"]
    with av.open(share) as f:
        v = f.streams.video[0]
        assert v.codec_context.name == "h264" and v.codec_context.pix_fmt == "yuv420p"
    assert data["share_copy"]["file_size"] <= 10 * 2**20
    # The grain is drawn in: the H.264 frames match the AV1 decoded WITH grain, not the clean ones.
    av1 = res.result[0].get_stream_source()
    with_grain = av1enc.decode_rgb(av1, apply_grain=True, limit=6)
    clean = av1enc.decode_rgb(av1, apply_grain=False, limit=6)
    with av.open(share) as f:
        h264 = (
            np.stack([fr.to_ndarray(format="rgb24") for fr in f.decode(video=0)][:6]).astype(
                np.float32
            )
            / 255
        )
    assert np.abs(h264 - with_grain).mean() < np.abs(h264 - clean).mean() * 0.7
    assert len(res.ui.as_dict()["images"]) == 2


def test_share_copy_shrinks_to_a_tight_limit():
    av1enc.SHARE_TARGETS["tight (0.15 MB)"] = 0.15
    try:
        import folder_paths

        folder_paths.set_output_directory(tempfile.mkdtemp(prefix="share tiny "))
        res = nodes.AV1FilmGrainEncode.execute(
            clip(48), 24.0, "share/tiny", ENC, {"grain_mode": "preset", "preset": "Vintage Anime", "intensity": 1.0, "chroma_grain": False},
            share_copy="tight (0.15 MB)",
        )  # fmt: skip
        report = json.loads(res.result[1])["share_copy"]
        assert (
            report["file_size"] <= 0.15 * 2**20
            and report["attempts"] > 1
            and "kb/s" in report["mode"]
        )
    finally:
        del av1enc.SHARE_TARGETS["tight (0.15 MB)"]


if __name__ == "__main__":
    raise SystemExit(_harness.run(dict(globals())))
