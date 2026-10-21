"""Film grain tables: the text format, AV1 validation and the generators. No encoder needed.

COMFY_ROOT=/path/to/ComfyUI python tests/test_grain.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _harness

_harness.load()

import importlib

grain = importlib.import_module(f"{_harness.PACK_NAME}.grain")

#: A table as libaom's writer emits it for luma-only grain: no cCb / cCr lines.
LIBAOM_STYLE = """filmgrn1
E 0 9223372036854775807 1 7391 1
\tp 0 6 0 8 0 1 0 0 0 0 0 0
\tsY 3  0 20 128 30 255 10
\tsCb 0
\tsCr 0
\tcY
"""

TWO_ENTRIES = """filmgrn1
E 0 10000000 1 100 1
\tp 1 7 0 9 0 1 128 192 256 128 192 256
\tsY 2 0 40 255 40
\tsCb 2 0 20 255 20
\tsCr 2 0 20 255 20
\tcY 1 2 3 4
\tcCb 0 0 0 0 0
\tcCr 0 0 0 0 0
E 10000000 20000000 1 200 0
"""


def expect_error(text: str, needle: str) -> None:
    try:
        grain.loads(text)
    except grain.TableError as exc:
        assert needle in str(exc), f"{needle!r} not in {exc}"
    else:
        raise AssertionError(f"accepted a table that should fail with {needle!r}")


def test_every_preset_round_trips_through_the_text_format():
    for name in grain.PRESETS:
        for chroma in (False, True):
            table = grain.preset(name, 1280, 720, chroma=chroma, seed=1234)
            assert grain.loads(grain.dumps(table)) == table, name
            assert grain.has_grain(table), name


def test_chroma_coefficients_are_always_written():
    """SvtAv1EncApp refuses a table without cCb/cCr lines, even for luma-only grain."""
    text = grain.dumps(grain.photon_noise(800, 640, 360))
    assert "\tcCb 0\n" in text and "\tcCr 0\n" in text


def test_libaom_style_table_without_chroma_coefficients_is_accepted():
    table = grain.loads(LIBAOM_STYLE)
    p = table.entries[0].params
    assert p.scaling_points_y == ((0, 20), (128, 30), (255, 10)) and p.ar_coeffs_cb == (0,)


def test_multi_entry_table_and_parameter_reuse():
    table = grain.loads(TWO_ENTRIES)
    first, second = (e.params for e in table.entries)
    assert second.random_seed == 200 and not second.update_parameters
    assert second.scaling_points_y == first.scaling_points_y and second.ar_coeffs_y == (1, 2, 3, 4)
    assert grain.loads(grain.dumps(table)) == table


def test_invalid_tables_are_refused_with_the_field_named():
    base = LIBAOM_STYLE
    expect_error("hello", "must start with")
    expect_error(base.replace("p 0 6", "p 4 6"), "ar_coeff_lag 4")
    expect_error(base.replace("p 0 6 0 8", "p 0 6 0 12"), "scaling_shift")
    expect_error(base.replace("p 0 6 0 8", "p 0 5 0 8"), "ar_coeff_shift")
    expect_error(
        base.replace("sY 3  0 20 128 30 255 10", "sY 3  0 20 200 30 128 10"), "strictly increase"
    )
    expect_error(base.replace("sY 3  0 20 128 30 255 10", "sY 2  0 20 255 300"), "0..255")
    expect_error(base.replace("sY 3", "sY 15"), "AV1 allows 0..14")
    with_chroma_coeffs = base + "\tcCb 0\n\tcCr 0\n"
    expect_error(with_chroma_coeffs.replace("sCb 0", "sCb 1 0 20"), "both have points")
    expect_error(
        with_chroma_coeffs.replace("p 0 6 0 8 0", "p 0 6 0 8 1")
        .replace("sCb 0", "sCb 1 0 20")
        .replace("sCr 0", "sCr 1 0 20"),
        "takes no chroma scaling points",
    )
    expect_error(base.replace("p 0 6", "p 1 6"), "cY coefficient")  # lag 1 needs 4 cY values
    expect_error(base.replace("sY 3  0 20", "sY 3  0 x"), "is not an integer")
    expect_error(TWO_ENTRIES.replace("cY 1 2 3 4", "cY 1 2 3 400"), "-128..127")
    expect_error(TWO_ENTRIES.replace("E 10000000 20000000", "E 5000000 20000000"), "overlap")
    expect_error(TWO_ENTRIES.replace("E 0 10000000 1 100 1", "E 0 10000000 1 70000 1"), "16 bits")
    expect_error(
        base.replace("E 0 9223372036854775807 1 7391 1", "E 0 9 1 7391 0"), "entry 1 must define"
    )


def test_photon_noise_grows_with_iso_resolution_and_smaller_sensors():
    def peak(*args, **kw):
        return grain.peak_strength(grain.photon_noise(*args, **kw))

    assert peak(800, 1920, 1080) < peak(1600, 1920, 1080) < peak(3200, 1920, 1080)
    assert peak(1600, 1280, 720) < peak(1600, 3840, 2160)
    assert peak(1600, 1920, 1080) < peak(1600, 1920, 1080, sensor="Micro Four Thirds (17.3x13 mm)")
    # Shot noise alone makes 4x the ISO 2x the noise; the fixed read noise adds a little on top.
    ratio = peak(6400, 1920, 1080, sensor=grain.PHONE_SENSOR) / peak(
        1600, 1920, 1080, sensor=grain.PHONE_SENSOR
    )
    assert 1.8 < ratio < 2.6, ratio


def test_photon_noise_matches_libaoms_full_frame_model():
    """libaom's model at 1080p ISO 1600 has mid-grey noise near 1 code value; check the scale."""
    p = grain.photon_noise(1600, 1920, 1080).entries[0].params
    mid = (
        dict(p.scaling_points_y)[118] / grain.PHOTON_SCALING_PER_STD / (1 << (p.scaling_shift - 8))
    )
    assert 0.5 < mid < 2.0, mid
    assert p.ar_coeff_lag == 0 and not p.chroma_scaling_from_luma


def test_scaling_shift_keeps_weak_grain_precise_and_strong_grain_unclipped():
    weak = grain.stylized(0.5, tone="flat").entries[0].params
    strong = grain.stylized(30, tone="flat").entries[0].params
    assert weak.scaling_shift == 11 and weak.scaling_points_y[0][1] >= 20
    assert strong.scaling_shift == 8 and all(y <= 255 for _, y in strong.scaling_points_y)


def test_scale_strength_doubles_the_grain():
    table = grain.preset("Fine 35mm", 1280, 720)
    double = grain.scale_strength(table, 2.0)
    assert abs(grain.peak_strength(double) / grain.peak_strength(table) - 2.0) < 0.05
    assert grain.peak_strength(grain.scale_strength(table, 0.0)) == 0 and not grain.has_grain(
        grain.scale_strength(table, 0.0)
    )
    huge = grain.scale_strength(table, 100.0).entries[0].params
    assert huge.scaling_shift == 8 and max(y for _, y in huge.scaling_points_y) == 255


def test_texture_coefficients_are_a_stable_low_pass():
    assert grain.texture_coeffs(0.0) == (0,) * 24
    coarse = grain.texture_coeffs(1.0)
    assert len(coarse) == 24 and all(c >= 0 for c in coarse)
    assert sum(coarse) / 256 < 1.0, "AR weights must sum below 1 to stay stable"
    gain = grain._ar_gain(coarse, 3, 8)
    assert 1.02 < gain < 1.5, gain  # a gentle low-pass: stronger shows the 32x32 block tiling


def test_tone_curves_shape_the_scaling_function():
    flat = [y for _, y in grain.stylized(5, tone="flat").entries[0].params.scaling_points_y]
    mids = [y for _, y in grain.stylized(5, tone="midtones").entries[0].params.scaling_points_y]
    assert len(set(flat)) == 1
    assert mids[len(mids) // 2] > mids[0] and mids[len(mids) // 2] > mids[-1]


if __name__ == "__main__":
    raise SystemExit(_harness.run(dict(globals())))
