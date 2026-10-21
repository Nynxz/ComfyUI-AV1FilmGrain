"""Film damage and film sound: each effect does what it says, stays put when off, and keeps sync.

COMFY_ROOT=/path/to/ComfyUI python tests/test_damage_sound.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _harness

_harness.load()

import importlib

import numpy as np
import torch
from scipy import signal

fd = importlib.import_module(f"{_harness.PACK_NAME}.filmdamage")
fs = importlib.import_module(f"{_harness.PACK_NAME}.filmsound")
nodes = importlib.import_module(f"{_harness.PACK_NAME}.nodes")

H, W, T = 216, 384, 120


def grey(n=1, value=0.45):
    return torch.full((n, H, W, 3), value)


def damaged(d, frames, total=T, value=0.45):
    sched = fd.schedules(d, total, 24, H, W, "cpu")
    return torch.stack(
        [
            fd.apply(grey(1, value), d, fps=24, frame_offset=f, total_frames=total, sched=sched)[0]
            for f in frames
        ]
    )


def changed(x, value=0.45):
    return (x - value).abs().amax(-1) > 0.02


# ---------------------------------------------------------------- damage


def test_damage_off_is_the_identity():
    x = grey(3)
    assert torch.equal(fd.apply(x, fd.Damage()), x)


def test_real_artefacts_are_bundled_and_used():
    atlas = fd.load_sprites()
    assert atlas is not None, "assets/damage_sprites.npz must ship with the pack"
    assert {k: len(v) > 100 for k, v in atlas.items()} == {
        "dust": True,
        "dirt": True,
        "short_hair": True,
        "long_hair": True,
    }
    specks = fd.dust_schedule(fd.Damage(dust=1.0, seed=1), 5, W / H)
    assert specks and all(s.sprite and s.sprite[0] in ("dust", "dirt") for s in specks)
    assert all(
        not s.sprite
        for s in fd.dust_schedule(fd.Damage(dust=1.0, seed=1, real_artefacts=False), 5, W / H)
    )


def test_the_same_frames_however_the_clip_is_chunked():
    d = fd.Damage(dust=0.8, hair=1.0, scratches=1.0, light_leaks=1.0, breath=1.0, gate=0.5, seed=4)
    x = grey(T)
    whole = fd.apply(x, d, fps=24)
    parts = torch.cat(
        [fd.apply(x[i : i + 7], d, fps=24, frame_offset=i, total_frames=T) for i in range(0, T, 7)]
    )
    assert torch.equal(whole, parts)
    assert not torch.equal(whole, fd.apply(x, fd.Damage(**{**d.__dict__, "seed": 5}), fps=24))


def test_dust_grows_with_amount_and_with_smaller_gauges():
    frames = range(0, 40, 2)

    def area(**kw):
        return float(changed(damaged(fd.Damage(seed=2, **kw), frames)).float().mean())

    assert area(dust=0.2, gauge="16mm") < area(dust=0.8, gauge="16mm")
    assert (
        area(dust=0.6, gauge="35mm")
        < area(dust=0.6, gauge="16mm")
        < area(dust=0.6, gauge="Super 8")
    )


def test_dust_changes_every_frame():
    a, b = damaged(fd.Damage(dust=0.8, gauge="Super 8", seed=3), [10, 11])
    ca, cb = changed(a), changed(b)
    overlap = float((ca & cb).sum()) / max(1.0, float(ca.sum()))
    assert ca.any() and overlap < 0.6, overlap


def test_a_hair_stays_for_its_lifetime_then_goes():
    d = fd.Damage(hair=1.0, gauge="Super 8", seed=1)
    hair = fd.hair_schedule(d, T)[0]
    inside = damaged(d, [hair.start, (hair.start + hair.end) // 2])
    outside = damaged(d, [max(0, hair.start - 1)]) if hair.start > 0 else None
    assert changed(inside[0]).any() and changed(inside[1]).any()
    overlap = float((changed(inside[0]) & changed(inside[1])).sum()) / float(
        changed(inside[0]).sum()
    )
    assert overlap > 0.3, "the same hair, trembling a little, must still be there"
    if outside is not None:
        assert not changed(outside[0]).any()


def test_scratches_are_thin_vertical_lines():
    d = fd.Damage(scratches=1.0, seed=6)
    sc = fd.scratch_schedule(d, T)[0]
    c = changed(damaged(d, [sc.start + 1])[0])
    cols = c.any(0).nonzero().flatten()
    assert c.sum() > H * 0.3, "a long line"
    assert cols.numel() <= 6, f"at most a few pixels wide, got {cols.numel()} columns"


def test_cue_marks_only_on_their_frames_top_right():
    d = fd.Damage(cue_marks=True)
    total, fps = 300, 24
    cues = sorted(fd.cue_frames(total, fps))
    assert len(cues) == 4 and cues[0] == total - 4 - 8 * fps
    sched = fd.schedules(d, total, fps, H, W, "cpu")
    on = fd.apply(grey(1), d, fps=fps, frame_offset=cues[0], total_frames=total, sched=sched)[0]
    off = fd.apply(grey(1), d, fps=fps, frame_offset=cues[0] - 1, total_frames=total, sched=sched)[
        0
    ]
    c = changed(on)
    ys, xs = c.nonzero(as_tuple=True)
    assert c.any() and ys.float().mean() < H * 0.3 and xs.float().mean() > W * 0.7
    assert not changed(off).any()


def test_gate_corners_sit_inside_the_letterbox():
    d = fd.Damage(gate=1.0, letterbox="2.39")
    out = damaged(d, [0])[0]
    bar = fd.letterbox_bar(d, H, W)
    assert bar > 0 and out[:bar].max() == 0 and out[H - bar :].max() == 0
    # The picture's own corners are rounded: dark just inside the bar at the side, bright in the middle.
    assert out[bar + 1, 1].max() < 0.1, (
        "the gate's corner must round off at the bar, not be cut by it"
    )
    # (the gate's soft edge shades a band ~1 % of the picture height along every edge, by design)
    assert (out[bar + 6, W // 2] - 0.45).abs().max() < 1e-6
    assert (out[H // 2, 1] - 0.45).abs().max() < 0.05, "the middle of the side edge stays bright"


def test_letterbox_gate_leaks_and_breath():
    lb = damaged(fd.Damage(letterbox="2.39"), [0])[0]
    bar = round((H - W / 2.39) / 2)
    assert (
        lb[:bar].max() == 0 and lb[H - bar :].max() == 0 and (lb[H // 2] - 0.45).abs().max() < 1e-6
    )
    gate = damaged(fd.Damage(gate=1.0), [0])[0]
    assert gate[0, 0].max() < 0.1 and (gate[H // 2, W // 2] - 0.45).abs().max() < 1e-6
    d = fd.Damage(light_leaks=1.0, seed=1)
    leak = fd.leak_schedule(d, T)[0]
    glow = damaged(d, [(leak.start + leak.end) // 2])[0] - 0.45
    assert glow[..., 0].max() > 0.1 and glow[..., 0].sum() > 2 * glow[..., 2].sum(), "a warm glow"
    breath = damaged(fd.Damage(breath=1.0, seed=1), range(0, 60, 10)).mean((1, 2))
    assert 0.001 < float(breath.std(0).mean()) < 0.02, "a slight drift, not a flicker"


# ---------------------------------------------------------------- sound

SR = 48000


def tone(seconds=3.0, hz=1000.0, channels=2):
    t = np.arange(int(SR * seconds)) / SR
    return np.stack([0.3 * np.sin(2 * np.pi * hz * t + c) for c in range(channels)])


def test_sound_off_is_the_identity_and_length_never_changes():
    x = tone()
    assert np.allclose(fs.apply(x, SR, fs.Sound()), x, atol=1e-6)
    full = fs.Sound(optical=1, mono=True, saturation=1, hiss=1, crackle=1, wow=1, flutter=1, seed=3)
    y = fs.apply(x, SR, full)
    assert y.shape == x.shape and np.abs(y).max() <= 1.0


def test_mono_and_optical_passband():
    x = np.random.default_rng(0).standard_normal((2, SR * 3)) * 0.1
    y = fs.apply(x, SR, fs.Sound(mono=True, optical=1.0))
    assert np.allclose(y[0], y[1])
    f, p0 = signal.welch(x.mean(0), SR, nperseg=4096)
    _, p1 = signal.welch(y[0], SR, nperseg=4096)

    def gain(hz):
        i = int(np.argmin(np.abs(f - hz)))
        return 10 * np.log10(p1[i] / p0[i])

    assert abs(gain(1000)) < 1.0
    assert gain(15000) < -30 and gain(30) < -25


def test_wow_and_flutter_keep_sync():
    """A bounded delay, not a drift: the end of the clip lines up as well as the start."""
    x = np.zeros((1, SR * 20))
    x[0, :: SR // 2] = 1.0  # a click every half second
    y = fs.apply(x, SR, fs.Sound(wow=1.0, flutter=1.0, seed=9))
    for at in (SR, SR * 19):
        seg_x, seg_y = x[0, at - 2000 : at + 2000], y[0, at - 2000 : at + 2000]
        lag = int(np.argmax(np.correlate(seg_y, seg_x, "full")) - (len(seg_x) - 1))
        assert abs(lag) <= SR * 0.0012, f"lag {lag} samples at {at / SR:.0f} s"


def test_wow_sways_the_pitch():
    y = fs.apply(tone(6.0, channels=1), SR, fs.Sound(wow=1.0, seed=1))[0]
    inst = np.diff(np.unwrap(np.angle(signal.hilbert(y)))) * SR / (2 * np.pi)
    inst = np.convolve(
        inst, np.ones(240) / 240, "same"
    )  # the per-sample estimate is noisy; 5 ms average
    sway = np.percentile(inst[SR:-SR], 95) - np.percentile(inst[SR:-SR], 5)
    assert 5.0 < sway < 10.0, sway  # +-0.4 % of 1 kHz at full: about 8 Hz peak to peak


def test_crackle_and_hiss_are_seeded_and_sized():
    silence = np.zeros((1, SR * 10))
    a = fs.apply(silence, SR, fs.Sound(crackle=0.5, seed=1))
    assert np.array_equal(a, fs.apply(silence, SR, fs.Sound(crackle=0.5, seed=1)))
    assert not np.array_equal(a, fs.apply(silence, SR, fs.Sound(crackle=0.5, seed=2)))
    clicks = int((np.diff((np.abs(a[0]) > 0.01).astype(int)) == 1).sum())
    assert 50 < clicks < 400, clicks  # ~10-25 per second at 0.5
    h = fs.apply(silence, SR, fs.Sound(hiss=0.5))
    level = 20 * np.log10(h.std())
    assert -52 < level < -42, level


# ---------------------------------------------------------------- nodes


def test_damage_and_sound_nodes():
    x = grey(10)
    out = nodes.FilmDamage.execute(
        24.0, "16mm", 0.6, 0.5, 0.5, 0.3, 0.3, 0.3, True, "none", 1, images=x
    ).result[0]
    assert out.shape == x.shape and not torch.equal(out, x)
    audio = {"waveform": torch.from_numpy(tone()[None].astype(np.float32)), "sample_rate": SR}
    res = nodes.FilmSound.execute(audio, 0.6, True, 0.3, 0.25, 0.2, 0.2).result[0]
    assert res["waveform"].shape == audio["waveform"].shape and res["sample_rate"] == SR
    assert torch.equal(res["waveform"][0, 0], res["waveform"][0, 1]), "mono"


if __name__ == "__main__":
    raise SystemExit(_harness.run(dict(globals())))
