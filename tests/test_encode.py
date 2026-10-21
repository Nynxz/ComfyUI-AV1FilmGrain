"""Real encodes: grain in the bitstream, synthesised by the decoder, and the two nodes end to end.

COMFY_ROOT=/path/to/ComfyUI python tests/test_encode.py

Needs PyAV with libsvtav1 + libdav1d (ComfyUI's own PyAV has both) and SvtAv1EncApp for the
table modes. Every claim is measured on dav1d's output: (decoded with grain) - (decoded without),
over a flat region of a clip that has no noise of its own.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _harness

_harness.load()

import importlib

import av
import clip
import numpy as np
import torch

grain = importlib.import_module(f"{_harness.PACK_NAME}.grain")
av1enc = importlib.import_module(f"{_harness.PACK_NAME}.av1enc")
nodes = importlib.import_module(f"{_harness.PACK_NAME}.nodes")

H, W, N = 192, 336, 12
FLAT = clip.regions(H, W)["flat"]
TMP = tempfile.mkdtemp(prefix="av1grain test ")  # a space, on purpose
CAPS = av1enc.capabilities()


def encode(grain_settings, name, *, frames=N, size=(W, H), **settings):
    settings.setdefault("preset", 10)
    s = av1enc.Settings(grain=grain_settings, **settings)
    out = os.path.join(TMP, name)
    report = av1enc.encode(
        clip.frames(frames, size[1], size[0]), out, s, size=size, log=lambda _m: None
    )
    return out, report


def table_grain(table):
    return av1enc.Grain("table", table=grain.dumps(table))


def flat_grain(path, frames=6):
    return av1enc.grain_difference(path, frames)[:, FLAT[0], FLAT[1]]


def neighbour_correlation(g):
    return float(np.mean([np.corrcoef(f[:, :-1].ravel(), f[:, 1:].ravel())[0, 1] for f in g]))


def test_capabilities_are_detected():
    assert CAPS.pyav_svt and CAPS.dav1d, CAPS.summary()
    assert "svt-av1" in CAPS.encoders()


def test_disabled_signals_no_grain_and_decodes_identically():
    path, report = encode(av1enc.Grain(), "clean.mkv")
    assert report.grain_frames == 0 and report.frames == N
    assert np.abs(av1enc.grain_difference(path, 4)).max() == 0


def test_table_grain_is_in_the_bitstream_and_synthesised_by_the_decoder():
    path, report = encode(table_grain(grain.stylized(4.0, tone="flat")), "flat4.mkv")
    assert report.route == "svt-app"
    assert report.grain_frames == report.verified_frames == N, (
        "every frame carries film grain params"
    )
    g = flat_grain(path)
    assert abs(g.std() - 4.0) < 0.6, f"calibrated strength 4, measured {g.std():.2f}"
    # Synthesised fresh every frame, not a static pattern.
    assert abs(np.corrcoef(g[0].ravel(), g[1].ravel())[0, 1]) < 0.1
    # The coded (clean) reconstruction itself is clean: grain comes from the parameters alone.
    clean = av1enc.decode_rgb(path, apply_grain=False, limit=2)[:, FLAT[0], FLAT[1]]
    assert clean.std(axis=(1, 2)).max() < 0.004


def test_presets_differ_in_strength_and_texture():
    measured = {}
    for name in ("Fine 35mm", "Coarse 16mm", "Vintage Anime"):
        path, _ = encode(table_grain(grain.preset(name, W, H)), f"{name}.mkv")
        g = flat_grain(path)
        measured[name] = (float(g.std()), neighbour_correlation(g))
    fine, coarse, anime = measured["Fine 35mm"], measured["Coarse 16mm"], measured["Vintage Anime"]
    assert fine[0] < coarse[0] < anime[0], measured
    assert coarse[1] > fine[1] + 0.12, f"coarse grain must be spatially clumpier: {measured}"


def test_iso_presets_grow_with_iso():
    stds = []
    for name in ("High ISO 800", "High ISO 1600", "High ISO 3200"):
        path, _ = encode(table_grain(grain.preset(name, W, H)), f"{name}.mkv")
        stds.append(float(flat_grain(path).std()))
    assert stds[0] < stds[1] < stds[2], stds


def test_chroma_grain_reaches_the_chroma_planes():
    def chroma_std(chroma):
        path, _ = encode(
            table_grain(grain.stylized(5.0, tone="flat", chroma=chroma)), f"chroma{chroma}.mkv"
        )
        on = av1enc.decode_rgb(path, limit=3)[:, FLAT[0], FLAT[1]]
        off = av1enc.decode_rgb(path, apply_grain=False, limit=3)[:, FLAT[0], FLAT[1]]
        d = on - off
        # Monochrome grain moves R, G and B together; chroma grain moves them apart.
        return float((d[..., 0] - d[..., 1]).std())

    assert chroma_std(True) > 3 * chroma_std(False)


def test_automatic_estimation_finds_little_grain_in_clean_footage():
    path, report = encode(av1enc.Grain("estimate", estimation_level=12), "estimate.mkv")
    assert report.route == "pyav-svt"
    g = flat_grain(path)
    assert g.std() < 1.0, f"estimation must not invent strong grain on clean footage: {g.std():.2f}"


def test_grain_costs_almost_nothing_and_baked_grain_costs_a_lot():
    _clean_path, clean = encode(av1enc.Grain(), "size_clean.mkv", frames=24, preset=8)
    grain_path, synth = encode(
        table_grain(grain.preset("Coarse 16mm", W, H)), "size_grain.mkv", frames=24, preset=8
    )
    rng = np.random.default_rng(0)
    std = float(flat_grain(grain_path).std()) / 255.0

    def baked():
        for f in clip.frames(24, H, W):
            yield np.clip(f + rng.normal(0, std, f.shape[:2])[..., None], 0, 1).astype(np.float32)

    s = av1enc.Settings(preset=8)
    baked_report = av1enc.encode(
        baked(), os.path.join(TMP, "size_baked.mkv"), s, size=(W, H), log=lambda _m: None
    )
    assert synth.file_size < clean.file_size * 1.25, (clean.file_size, synth.file_size)
    assert baked_report.file_size > synth.file_size * 3, (synth.file_size, baked_report.file_size)


def test_every_container_keeps_the_grain_and_the_audio():
    sr = 48000
    t = np.arange(int(sr * N / 24)) / sr
    audio = {
        "waveform": torch.from_numpy(
            np.stack([np.sin(2 * np.pi * 440 * t)] * 2)[None].astype(np.float32)
        ),
        "sample_rate": sr,
    }
    for container in ("mkv", "mp4", "webm"):
        s = av1enc.Settings(
            preset=10, container=container, grain=table_grain(grain.preset("Fine 35mm", W, H))
        )
        out = os.path.join(TMP, f"c av'1 & grain.{container}")  # spaces and shell metacharacters
        report = av1enc.encode(
            clip.frames(N, H, W), out, s, size=(W, H), audio=audio, log=lambda _m: None
        )
        assert report.audio and report.grain_frames == N, container
        with av.open(out) as f:
            assert f.streams.video[0].codec_context.name in ("av1", "libdav1d")
            assert len(f.streams.audio) == 1
            v = f.streams.video[0]
            assert abs(float(v.average_rate) - 24) < 0.01 if v.average_rate else True
            assert abs(f.duration / 1e6 - N / 24) < 0.1, (container, f.duration)


def test_frame_order_and_rate_are_preserved():
    path, report = encode(av1enc.Grain(), "order.mkv", fps=30000 / 1001)
    assert report.fps == 30000 / 1001
    decoded = av1enc.decode_rgb(path, apply_grain=False)
    detail = clip.regions(H, W)["detail"]
    for i in (0, 5, N - 1):
        ref = clip.frame(i, H, W)[detail]
        errors = [np.abs(decoded[j][detail] - ref).mean() for j in range(N)]
        assert int(np.argmin(errors)) == i


def test_odd_sizes_are_cropped_with_a_warning():
    _, report = encode(av1enc.Grain(), "odd.mkv", size=(W - 1, H - 1), frames=3)
    assert (report.width, report.height) == (W - 2, H - 2) and report.cropped_to_even
    assert any("even dimensions" in w for w in report.warnings)


def test_unsupported_combinations_fail_before_encoding():
    no_app = av1enc.Capabilities(True, False, True, None, None, False)
    try:
        av1enc.validate(av1enc.Settings(grain=table_grain(grain.preset("Fine 35mm", W, H))), no_app)
    except av1enc.EncodeError as exc:
        assert "SvtAv1EncApp" in str(exc) and "rejects fgs-table" in str(exc)
    else:
        raise AssertionError("a table with no SvtAv1EncApp must be refused")
    for bad in (
        av1enc.Settings(crf=70),
        av1enc.Settings(container="avi"),
        av1enc.Settings(encoder="libaom-av1"),
    ):
        try:
            av1enc.validate(bad, no_app)
        except av1enc.EncodeError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")


def test_ffmpeg_libsvtav1_really_drops_fgs_table():
    """The reason tables go through SvtAv1EncApp: FFmpeg's wrapper encodes without the grain."""
    table = os.path.join(TMP, "probe.tbl")
    with open(table, "w") as fh:
        fh.write(grain.dumps(grain.stylized(5.0, tone="flat")))
    out = os.path.join(TMP, "ffmpeg_fgs.ivf")
    with av.open(out, "w", format="ivf") as f:
        s = f.add_stream("libsvtav1", rate=24)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"preset": "10", "svtav1-params": f"fgs-table={table}"}
        for frame in av1enc.yuv_frames(clip.frames(3, H, W), "yuv420p"):
            f.mux(s.encode(frame))
        f.mux(s.encode(None))
    assert av1enc.inspect(out)[1] == 0, (
        "if this fails, FFmpeg now supports fgs-table: revisit the route"
    )


def test_temporary_files_are_removed():
    before = set(os.listdir(TMP))
    encode(table_grain(grain.preset("Fine 35mm", W, H)), "tmpcheck.mkv", frames=2)
    leftovers = {n for n in set(os.listdir(TMP)) - before if n.startswith("av1grain_")}
    assert not leftovers, leftovers


def test_libaom_route_honours_tables_seeds_and_awkward_paths():
    """Runs where PyAV has libaom-av1 (ComfyUI's wheel does not; a PyAV built against a system
    FFmpeg with libaom does)."""
    if not CAPS.pyav_aom:
        print("     (skipped: this PyAV has no libaom-av1)")
        return
    awkward = os.path.join(TMP, "dir with: colon & space")
    os.makedirs(awkward, exist_ok=True)
    for name, temp_dir in (("aom.mkv", None), ("aom_colon.mkv", awkward)):
        s = av1enc.Settings(
            encoder="libaom-av1",
            cpu_used=8,
            grain=table_grain(grain.stylized(4.0, tone="flat", seed=1234)),
        )
        out = os.path.join(TMP, name)
        report = av1enc.encode(
            clip.frames(N, H, W), out, s, size=(W, H), temp_dir=temp_dir, log=lambda _m: None
        )
        assert report.route == "pyav-libaom" and report.grain_frames == N
        assert report.grain_seeds[0] == 1234, "libaom starts from the table's seed"
        assert abs(flat_grain(out).std() - 4.0) < 0.6
    assert not os.listdir(awkward), "temporary table copies must be removed"


# ---------------------------------------------------------------- nodes


def _images(n=6):
    return torch.from_numpy(np.stack(list(clip.frames(n, H, W))))


def test_encode_node_end_to_end():
    import folder_paths

    folder_paths.set_output_directory(TMP)
    result = nodes.AV1FilmGrainEncode.execute(
        _images(),
        24.0,
        "grain test/out",
        {"encoder": "svt-av1", "crf": 30, "preset": 10},
        {
            "grain_mode": "preset",
            "preset": "Vintage Anime",
            "intensity": 1.5,
            "chroma_grain": False,
        },
        container="mkv",
    )
    video, report = result.result
    data = json.loads(report)
    assert data["grain_frames"] == data["frames"] == 6 and "filmgrn1" in data["table"]
    w, h = video.get_dimensions()
    assert (w, h) == (W, H)
    saved = result.ui.as_dict()
    entry = next(iter(saved.values()))[0]
    on_disk = os.path.join(TMP, entry["subfolder"], entry["filename"])
    assert os.path.isfile(on_disk) and entry["filename"].endswith(".mkv")
    with av.open(on_disk) as f:
        assert "av1_film_grain" in {k.lower() for k in f.metadata}  # Matroska upper-cases tag names


def test_table_file_mode_validates_the_file():
    import folder_paths

    models = os.path.join(TMP, "models with spaces")
    os.makedirs(models, exist_ok=True)
    folder_paths.folder_names_and_paths[nodes.TABLE_FOLDER] = ([models], nodes.TABLE_EXTENSIONS)
    folder_paths.filename_list_cache.clear() if hasattr(
        folder_paths, "filename_list_cache"
    ) else None
    with open(os.path.join(models, "my grain (v2).tbl"), "w") as fh:
        fh.write(grain.dumps(grain.photon_noise(3200, W, H, sensor=grain.PHONE_SENSOR)))
    with open(os.path.join(models, "broken.tbl"), "w") as fh:
        fh.write("filmgrn1\nE 0 10 1 7391 1\n\tp 9 6 0 8 0 1 0 0 0 0 0 0\n")
    settings = nodes.build_settings(
        {"encoder": "svt-av1", "crf": 30, "preset": 10},
        {"grain_mode": "table file", "table": "my grain (v2).tbl", "intensity": 1.0},
        W,
        H,
    )
    assert settings.grain.mode == "table"
    try:
        nodes.build_settings(
            {"encoder": "svt-av1", "crf": 30, "preset": 10},
            {"grain_mode": "table file", "table": "broken.tbl", "intensity": 1.0},
            W,
            H,
        )
    except ValueError as exc:
        assert "ar_coeff_lag 9" in str(exc)
    else:
        raise AssertionError("broken table accepted")
    try:
        nodes.build_settings(
            {"encoder": "svt-av1", "crf": 30, "preset": 10},
            {"grain_mode": "table file", "table": "../../etc/passwd", "intensity": 1.0},
            W,
            H,
        )
    except ValueError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("path outside models/film_grain accepted")


def test_preview_node_shows_the_decoders_grain():
    result = nodes.AV1FilmGrainPreview.execute(
        _images(8),
        2,
        3,
        24.0,
        {"encoder": "svt-av1", "crf": 30, "preset": 10},
        {
            "grain_mode": "photon noise",
            "iso": 3200,
            "sensor": grain.PHONE_SENSOR,
            "chroma_grain": False,
        },
    )
    with_grain, without, report = result.result
    assert with_grain.shape == without.shape == (3, H, W, 3)
    diff = (with_grain - without)[:, FLAT[0], FLAT[1]]
    assert diff.std() > 1 / 255, "the preview must show visible grain"
    assert json.loads(report)["grain_frames"] == 3


if __name__ == "__main__":
    try:
        code = _harness.run(dict(globals()))
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    raise SystemExit(code)
