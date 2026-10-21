"""The sound of a film print: optical soundtrack tone, hiss, crackle and pops, wow and flutter. No ComfyUI imports.

Until the late 1970s most prints carried their sound as an optical track, a strip of light and dark
beside the picture read by a photocell. What that does to sound:

    optical     band-limited (no deep bass, no air above ~7-12 kHz), usually mono, softly saturated
    hiss        the track's grain and the photocell, as broadband noise shaped like the passband
    crackle     dirt and scratches on the track: frequent small clicks and occasional pops
    wow/flutter the projector's speed varying: a slow pitch sway (wow, ~0.6 Hz) and a fast
                warble (flutter, ~6-12 Hz)

Wow and flutter are applied as a small, zero-mean *delay* (well under a millisecond), so the sound
never drifts out of sync with the picture; the filters are zero-phase for the same reason. Waveforms
are float (channels, samples) in -1..1, as in ComfyUI's AUDIO.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class Sound:
    """Every amount is 0..1 and 0 is off."""

    #: Optical-track band-limiting: 0 = full range, 1 = roughly 100 Hz .. 6 kHz.
    optical: float = 0.0
    #: Fold to mono, as an optical track is.
    mono: bool = False
    #: Soft saturation of the track.
    saturation: float = 0.0
    hiss: float = 0.0
    crackle: float = 0.0
    wow: float = 0.0
    flutter: float = 0.0
    seed: int = 0

    def is_off(self) -> bool:
        return not (
            self.optical
            or self.mono
            or self.saturation
            or self.hiss
            or self.crackle
            or self.wow
            or self.flutter
        )


def passband(optical: float) -> tuple[float, float]:
    """(low, high) cutoff in Hz for an optical amount."""
    return 30 + 70 * optical, 16000 - 10000 * optical


def _bandlimit(x: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    nyq = sr / 2
    if high >= nyq * 0.98:
        high = None
    if high is None:
        sos = signal.butter(2, low, "highpass", fs=sr, output="sos")
    else:
        sos = signal.butter(2, [low, high], "bandpass", fs=sr, output="sos")
    # Forward-backward: zero phase (no delay), and 4th-order slopes from a 2nd-order design.
    return signal.sosfiltfilt(sos, x, axis=-1)


def _band_noise(rng: np.random.Generator, n: int, sr: int, lo: float, hi: float) -> np.ndarray:
    """Zero-mean noise band-limited to lo..hi Hz, normalised to unit standard deviation."""
    z = rng.standard_normal(n)
    sos = signal.butter(2, [lo, min(hi, sr / 2 * 0.95)], "bandpass", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, z)
    return y / max(y.std(), 1e-12)


def speed_delay(n: int, sr: int, wow: float, flutter: float, seed: int) -> np.ndarray:
    """Delay in seconds per sample for wow and flutter -- bounded and zero-mean, so sync is kept.

    A speed deviation s(t) is the derivative of the delay. Wow: s = A sin(2 pi f t), so the delay is
    -A / (2 pi f) cos(...). Flutter: band-limited noise around 6-12 Hz used directly as the delay,
    scaled so the speed deviation it implies has the requested size.
    """
    t = np.arange(n) / sr
    rng = np.random.default_rng([seed & 0xFFFFFFFF, 11])
    delay = np.zeros(n)
    if wow > 0:
        f = rng.uniform(0.45, 0.8)
        depth = 0.004 * wow  # +-0.4 % speed at full: a clearly audible sway
        delay += (
            -depth / (2 * math.pi * f) * np.cos(2 * math.pi * f * t + rng.uniform(0, 2 * math.pi))
        )
    if flutter > 0 and n > sr // 4:
        depth = 0.0015 * flutter  # +-0.15 % speed
        delay += _band_noise(rng, n, sr, 6.0, 12.0) * depth / (2 * math.pi * 9.0)
    return delay


def _crackle(rng: np.random.Generator, n: int, sr: int, amount: float) -> np.ndarray:
    """Clicks (short, bright, frequent) and pops (louder, duller, rare)."""
    out = np.zeros(n)
    seconds = n / sr
    for rate, level, decay_ms, kind in (
        (30 * amount**1.3, 0.12, 0.25, "click"),
        (1.2 * amount, 0.35, 2.0, "pop"),
    ):
        count = rng.poisson(rate * seconds)
        if not count:
            continue
        pos = rng.integers(0, n, count)
        amp = level * amount**0.5 * rng.lognormal(0, 0.6, count) * rng.choice((-1.0, 1.0), count)
        impulses = np.zeros(n)
        np.add.at(impulses, pos, amp)
        length = max(4, int(sr * decay_ms / 1000 * 5))
        kernel = np.exp(-np.arange(length) / (sr * decay_ms / 1000))
        burst = signal.fftconvolve(impulses, kernel)[:n]
        if kind == "pop":
            burst = signal.sosfilt(signal.butter(2, 1800, "lowpass", fs=sr, output="sos"), burst)
        else:
            burst = signal.sosfilt(signal.butter(2, 1500, "highpass", fs=sr, output="sos"), burst)
        out += burst
    return out


def apply(waveform: np.ndarray, sr: int, s: Sound) -> np.ndarray:
    """The film-print treatment on (channels, samples) float audio. Same shape, same length."""
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (channels, samples), got shape {x.shape}")
    if s.is_off():
        return x.astype(np.float32)
    channels, n = x.shape
    if s.mono and channels > 1:
        x = np.repeat(x.mean(0, keepdims=True), channels, 0)

    if s.wow > 0 or s.flutter > 0:
        delay = speed_delay(n, sr, s.wow, s.flutter, s.seed)
        read = np.clip(np.arange(n) + delay * sr, 0, n - 1)
        x = np.stack([np.interp(read, np.arange(n), ch) for ch in x])

    low, high = passband(s.optical)
    if s.optical > 0:
        x = _bandlimit(x, sr, low, high)

    if s.saturation > 0:
        drive = 1 + 4 * s.saturation
        x = np.tanh(drive * x) / math.tanh(drive)

    rng = np.random.default_rng([s.seed & 0xFFFFFFFF, 12])
    noise = np.zeros(n)
    if s.hiss > 0:
        level = 10 ** (
            (-62 + 30 * s.hiss) / 20
        )  # -62 dBFS rms (barely there) .. -32 dBFS (prominent)
        noise += _band_noise(rng, n, sr, max(low, 200.0), min(high, 12000.0)) * level
    if s.crackle > 0:
        noise += _crackle(rng, n, sr, s.crackle)
    if noise.any():
        # The track's noise is in the track: mono, the same in every channel.
        x = x + noise[None, :]
    return np.clip(x, -1.0, 1.0).astype(np.float32)
