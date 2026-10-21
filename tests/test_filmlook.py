"""The film look: each stage does what it claims, and nothing when it is off.

COMFY_ROOT=/path/to/ComfyUI python tests/test_filmlook.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _harness

_harness.load()

import importlib

import clip
import torch

fl = importlib.import_module(f"{_harness.PACK_NAME}.filmlook")
nodes = importlib.import_module(f"{_harness.PACK_NAME}.nodes")

H, W = 144, 256


def frames(n=3):
    return torch.from_numpy(clip.frame(0, H, W)).unsqueeze(0).repeat(n, 1, 1, 1)


def ramp():
    x = torch.linspace(0, 1, 256)
    return x.view(1, 1, 256, 1).expand(1, 8, 256, 3).contiguous()


def test_everything_off_is_the_identity():
    x = frames()
    y = fl.apply(x, fl.Grade(), fl.Optics())
    assert (y - x).abs().max() < 2e-5, float((y - x).abs().max())


def test_tone_curve_keeps_black_and_white_and_is_monotonic():
    x = torch.linspace(0, 1, 1001)
    for name, (_, g) in fl.LOOKS.items():
        y = fl.tone_curve(x, g)
        assert y[0] < 1e-5 and y[-1] > 1 - 1e-5, name
        assert (y[1:] - y[:-1] >= -1e-6).all(), f"{name} must not invert tones"


def test_rolloff_softens_the_shoulder_and_toe_softens_the_shadows():
    def slope(g, at):
        x = torch.tensor([at - 0.005, at + 0.005])
        y = fl.tone_curve(x, g)
        return float((y[1] - y[0]) / 0.01)

    plain = fl.Grade(contrast=1.2)
    soft = fl.Grade(contrast=1.2, highlight_rolloff=1.0, shadow_toe=1.0)
    # The shoulder bends the curve into white: flatter just below white than without one.
    assert slope(soft, 0.985) < 0.6 * slope(plain, 0.985)
    # The toe softens only the very bottom: flatter at black, but shadow detail keeps its slope.
    assert slope(soft, 0.006) < slope(plain, 0.006)
    assert slope(soft, 0.15) > 0.7 * slope(plain, 0.15)


def test_low_key_footage_is_not_crushed():
    """Power-law contrast keeps shadow ratios: a dark face stays a darker face, not black.

    Checked from 5 %: below ~3 % the toe softens into black on purpose.
    """
    x = torch.tensor([0.05, 0.1, 0.2])
    for name, (_, g) in fl.LOOKS.items():
        if name in ("none", "vivid slide", "bleach bypass"):  # those two are meant to be dense
            continue
        y = fl.grade(fl.tone_curve(x[:, None].expand(3, 3).contiguous(), g), g).mean(
            -1
        )  # what the viewer sees
        assert (y > 0.6 * x).all(), (name, y.tolist())
        assert (y[1:] - y[:-1] > 0.4 * (x[1:] - x[:-1])).all(), f"{name} flattens shadow detail"


def test_fade_lifts_black():
    y = fl.apply(torch.zeros(1, 4, 4, 3), fl.LOOKS["faded 70s"][1], fl.Optics())
    assert 0.06 < y.mean() < 0.12


def test_looks_shift_colour_the_way_they_say():
    grey = torch.full((1, 8, 8, 3), 0.5)

    def rgb(look):
        return fl.apply(grey, fl.LOOKS[look][1], fl.Optics())[0, 0, 0]

    warm, cool = rgb("warm print"), rgb("cool night")
    assert warm[0] > warm[2] + 0.03, warm
    assert cool[2] > cool[0] + 0.03, cool
    x = frames(1)
    sat = lambda t: float((t - t.mean(-1, keepdim=True)).abs().mean())  # noqa: E731
    assert sat(fl.apply(x, fl.LOOKS["bleach bypass"][1], fl.Optics())) < sat(x)
    assert sat(fl.apply(x, fl.LOOKS["vivid slide"][1], fl.Optics())) > sat(x)


def test_look_strength_blends_towards_no_grade():
    g = fl.LOOKS["teal & orange"][1]
    assert fl.blend_grade(g, 0.0) == fl.Grade()
    assert fl.blend_grade(g, 1.0) == g
    half = fl.blend_grade(g, 0.5)
    assert abs(half.contrast - (1 + (g.contrast - 1) / 2)) < 1e-9


def test_halation_is_a_red_glow_around_highlights_only():
    x = torch.full((1, 64, 64, 3), 0.1)
    x[:, 28:36, 28:36] = 1.0
    y = fl.apply(x, fl.Grade(), fl.Optics(halation=1.0, halation_size=0.06))
    ring = y[0, 24, 32] - x[0, 24, 32]  # just outside the bright square
    far = y[0, 2, 2] - x[0, 2, 2]
    assert ring[0] > 0.05 and ring[0] > 2 * ring[2], f"red-orange glow expected, got {ring}"
    assert far.abs().max() < 0.01, "dark areas far away must not glow"
    dim = torch.full((1, 64, 64, 3), 0.2)
    assert (fl.apply(dim, fl.Grade(), fl.Optics(halation=1.0)) - dim).abs().max() < 0.01, (
        "no highlights, no halation"
    )


def test_vignette_darkens_corners_not_centre():
    x = torch.full((1, 90, 160, 3), 0.6)
    y = fl.apply(x, fl.Grade(), fl.Optics(vignette=1.0))
    assert abs(y[0, 45, 80, 0] - 0.6) < 0.01 and y[0, 0, 0, 0] < 0.45


def test_softness_scales_with_resolution():
    """The same setting blurs a 2x larger frame by 2x the pixels: the look holds at any size."""

    def edge_width(h):
        x = torch.zeros(1, h, h, 3)
        x[:, :, h // 2 :] = 1.0
        row = fl.apply(x, fl.Grade(), fl.Optics(softness=1.0))[0, h // 2, :, 0]
        return int(((row > 0.05) & (row < 0.95)).sum())

    small, large = edge_width(540), edge_width(1080)
    assert large >= small * 1.6, (small, large)


def test_gate_weave_moves_frames_differently_and_repeatably():
    x = torch.from_numpy(clip.frame(0, H, W)).unsqueeze(0).repeat(6, 1, 1, 1)
    o = fl.Optics(gate_weave=1.0, seed=3)
    a, b = fl.apply(x, fl.Grade(), o), fl.apply(x, fl.Grade(), o)
    assert torch.equal(a, b), "same seed, same weave"
    assert (a[0] - a[3]).abs().mean() > 1e-3, "frames must move relative to each other"
    # Chunked processing continues the same motion.
    parts = torch.cat(
        [
            fl.apply(x[:3], fl.Grade(), o, frame_offset=0, total_frames=6),
            fl.apply(x[3:], fl.Grade(), o, frame_offset=3, total_frames=6),
        ]
    )
    assert (parts - a).abs().max() < 1e-5


def test_cube_parsing_and_identity_lut():
    lut = fl.parse_cube(fl.identity_cube(9))
    assert lut.is_3d and lut.table.shape == (9, 9, 9, 3)
    x = frames(1)
    assert (fl.apply_lut(x, lut) - x).abs().max() < 1e-5


def test_lut_axis_order_red_varies_fastest():
    """A LUT that swaps red and blue must swap them: catches a transposed table."""
    size = 5
    lines = ["LUT_3D_SIZE 5"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(f"{b / 4} {g / 4} {r / 4}")
    lut = fl.parse_cube("\n".join(lines))
    px = torch.tensor([[[[0.9, 0.5, 0.1]]]])
    out = fl.apply_lut(px, lut)[0, 0, 0]
    assert torch.allclose(out, torch.tensor([0.1, 0.5, 0.9]), atol=1e-5), out


def _hald_identity(level: int) -> torch.Tensor:
    n = level * level
    t = torch.linspace(0, 1, n)
    b, g, r = torch.meshgrid(t, t, t, indexing="ij")
    return (torch.stack([r, g, b], -1).reshape(level**3, level**3, 3) * 255).round().to(torch.uint8)


def test_hald_clut_identity_and_layout():
    lut = fl.parse_hald(_hald_identity(8).numpy())
    assert lut.is_3d and lut.table.shape == (64, 64, 64, 3)
    x = frames(1)
    assert (fl.apply_lut(x, lut) - x).abs().max() < 1.5 / 255
    # Swap red and blue in the image: the LUT must swap them too (catches a transposed reading).
    swapped = fl.parse_hald(_hald_identity(4).flip(-1).numpy())
    out = fl.apply_lut(torch.tensor([[[[0.9, 0.5, 0.1]]]]), swapped)[0, 0, 0]
    assert torch.allclose(out, torch.tensor([0.1, 0.5, 0.9]), atol=0.01), out
    for bad in (
        torch.zeros(100, 100, 3, dtype=torch.uint8),
        torch.zeros(64, 512, 3, dtype=torch.uint8),
    ):
        try:
            fl.parse_hald(bad.numpy())
        except fl.LutError as exc:
            assert "L^3" in str(exc)
        else:
            raise AssertionError(bad.shape)


def test_hald_png_file_through_the_node():
    import folder_paths
    from PIL import Image

    folder = tempfile.mkdtemp(prefix="hald luts ")
    folder_paths.folder_names_and_paths[nodes.LUT_FOLDER] = ([folder], nodes.LUT_EXTENSIONS)
    Image.fromarray(_hald_identity(8).flip(-1).numpy()).save(os.path.join(folder, "swap rb.png"))
    x = frames(1)
    out = nodes.FilmLook.execute(
        {"look": "none"}, "swap rb.png", 1.0, 0.0, 0.0, 0.0, 0.0, images=x, lut_shadow_protect=0.0
    ).result[0]
    assert (out - x.flip(-1)).abs().max() < 2.0 / 255


def test_lut_shadow_protect_keeps_shadows_and_not_highlights():
    lut = fl.parse_hald(_hald_identity(4).flip(-1).numpy())  # swaps red and blue
    dark, bright = torch.tensor([[0.06, 0.02, 0.01]]), torch.tensor([[0.9, 0.5, 0.1]])
    assert torch.allclose(fl.apply_lut(dark, lut, 1.0, 1.0), dark, atol=0.01), (
        "protected shadows stay"
    )
    assert torch.allclose(fl.apply_lut(bright, lut, 1.0, 1.0), bright.flip(-1), atol=0.02), (
        "highlights get the LUT"
    )


def test_clean_shadows_smooths_flat_dark_patches_not_edges():
    h, w = 96, 128
    x = torch.full((1, h, w, 3), 0.03)
    x[:, :, w // 2 :] = 0.045  # a step between two flat dark patches, like source macroblocks
    stripes = torch.where(torch.arange(w).view(1, 1, w, 1) % 2 == 0, 0.6, 0.2).expand(
        1, h // 2, w, 3
    )
    x[:, h // 2 :] = stripes  # bright texture
    y = fl.apply(x, fl.Grade(), fl.Optics(clean_shadows=1.0))
    before = abs(float(x[0, 10, w // 2, 0] - x[0, 10, w // 2 - 1, 0]))
    after = float((y[0, 10, w // 2] - y[0, 10, w // 2 - 1]).abs().max())
    assert after < 0.5 * before, "the patch edge becomes a gradient"
    assert (y[0, h // 2 + 8 :, 20:-20] - x[0, h // 2 + 8 :, 20:-20]).abs().max() < 0.01, (
        "bright texture untouched"
    )


def test_1d_lut_and_domain():
    lut = fl.parse_cube("LUT_1D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n1 1 1\n0 0 0\n")
    assert torch.allclose(
        fl.apply_lut(torch.tensor([[0.25, 0.5, 1.0]]), lut), torch.tensor([[0.75, 0.5, 0.0]])
    )


def test_bad_cubes_are_refused_with_a_reason():
    for text, needle in (
        ("hello", "not a .cube"),
        ("LUT_3D_SIZE 2\n0 0 0\n", "needs 8 entries"),
        ("LUT_3D_SIZE 2\n" + "0 0\n" * 8, "expected 3 values"),
        ("LUT_3D_SIZE 300\n", "outside 2..256"),
        ("LUT_3D_SIZE 2\nDOMAIN_MIN 1 1 1\nDOMAIN_MAX 0 0 0\n" + "0 0 0\n" * 8, "DOMAIN_MIN"),
        ("LUT_3D_SIZE 2\n" + "0 0 x\n" * 8, "cannot read"),
    ):
        try:
            fl.parse_cube(text)
        except fl.LutError as exc:
            assert needle in str(exc), (needle, str(exc))
        else:
            raise AssertionError(f"accepted {text[:30]!r}")


def test_a_look_exported_as_cube_reproduces_the_grade():
    g = fl.LOOKS["warm print"][1]
    lut = fl.parse_cube(fl.grade_to_cube(g, size=33))
    x = frames(1)
    direct = fl.apply(x, g, fl.Optics())
    via_lut = fl.apply_lut(x, lut)
    assert (direct - via_lut).abs().mean() < 0.004, float((direct - via_lut).abs().mean())


# ---------------------------------------------------------------- nodes


def test_film_look_node_and_comparison():
    x = frames(5)
    result = nodes.FilmLook.execute(
        {"look": "warm print", "strength": 1.0},
        "none",
        1.0,
        0.35,
        0.15,
        0.3,
        0.25,
        compare_frame=2,
        images=x,
    )
    out, comparison, _fx = result.result
    assert out.shape == x.shape and comparison.shape == (1, H, W, 3)
    assert torch.equal(comparison[0, :, : W // 2 - 2], x[2, :, : W // 2 - 2]), (
        "left half is the original"
    )
    assert torch.allclose(comparison[0, :, W // 2 + 2 :], out[2, :, W // 2 + 2 :]), (
        "right half has the look"
    )


def test_custom_look_and_lut_files():
    import folder_paths

    folder = tempfile.mkdtemp(prefix="luts ")
    folder_paths.folder_names_and_paths[nodes.LUT_FOLDER] = ([folder], nodes.LUT_EXTENSIONS)
    custom = {
        "look": "custom", "contrast": 1.2, "highlight_rolloff": 0.5, "shadow_toe": 0.3, "fade": 0.05,
        "saturation": 0.9, "temperature": 0.2, "tint": 0.0, "crosstalk": 0.1, "bleach_bypass": 0.0,
    }  # fmt: skip
    saved = nodes.SaveFilmLookLUT.execute(custom, "my look").result[0]
    assert os.path.isfile(saved) and saved.endswith("my look.cube")
    try:
        nodes.SaveFilmLookLUT.execute(custom, "my look")
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("overwrote an existing LUT")
    x = frames(2)
    via_lut = nodes.FilmLook.execute(
        {"look": "none"}, "my look.cube", 1.0, 0.0, 0.0, 0.0, 0.0, images=x
    ).result[0]
    direct = nodes.FilmLook.execute(custom, "none", 1.0, 0.0, 0.0, 0.0, 0.0, images=x).result[0]
    assert (via_lut - direct).abs().mean() < 0.005
    for bad in ("../../etc/passwd", "missing.cube"):
        try:
            nodes.load_lut(bad)
        except ValueError as exc:
            assert "not found" in str(exc)
        else:
            raise AssertionError(bad)


def test_extension_exposes_all_nodes():
    import asyncio

    pack = sys.modules[_harness.PACK_NAME]
    ext = asyncio.run(pack.comfy_entrypoint())
    ids = sorted(n.define_schema().node_id for n in asyncio.run(ext.get_node_list()))
    assert ids == sorted(
        [
            "av1grain.FilmLook",
            "av1grain.FilmDamage",
            "av1grain.FilmSound",
            "av1grain.VHSTape",
            "av1grain.AV1FilmGrainEncode",
            "av1grain.AV1FilmGrainPreview",
            "av1grain.SaveFilmLookLUT",
        ]
    ), ids
    for n in asyncio.run(ext.get_node_list()):
        n.GET_NODE_INFO_V1()  # what /object_info serves


if __name__ == "__main__":
    raise SystemExit(_harness.run(dict(globals())))
