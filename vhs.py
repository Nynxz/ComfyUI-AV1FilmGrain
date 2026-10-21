"""NTSC composite video and VHS tape, simulated as signals. No ComfyUI imports.

Frames are resampled to the NTSC raster -- 480 active lines of 52.6 us, sampled at four times the
colour subcarrier (4 fsc = 14.318 MHz, ~754 samples) -- so every bandwidth below is the real one in
MHz, as in SMPTE 170M and the VHS format, not a blur radius tuned by eye. Then, per line:

    composite   Y + I cos(wt + 33 deg) + Q sin(wt + 33 deg), the subcarrier advancing half a cycle
                per line and per frame; decoded by a notch filter (or a comb filter). Luma detail
                near 3.58 MHz leaks into chroma (rainbows) and chroma into luma (dot crawl).
    VHS record  luma low-passed to ~3 MHz (SP; less for EP), chroma "colour-under" to ~0.5 MHz,
                chroma shifted right of luma, chroma averaged over two lines (the 1H delay line)
    playback    the VCR's sharpening (ringing halos), time-base error (per-line jitter, edge waving),
                head-switching skew at the bottom, mistracking bands, dropouts
    noise       luma FM noise and chroma noise -- returned as an AV1 film grain table, so the
                decoder synthesises it and the file stays small (or baked in, on request)

Each frame's randomness comes from (seed, frame index), so chunked runs match. Frames are IMAGE
tensors (N, H, W, 3) sRGB in 0..1; the output has the input's size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from . import grain as grain_tables

FSC = 315e6 / 88  # NTSC colour subcarrier, 3.579545 MHz
SAMPLE_RATE = 4 * FSC  # 14.318 MHz
ACTIVE_US = 52.6
LINES = 480
SAMPLES = round(ACTIVE_US * 1e-6 * SAMPLE_RATE)  # 753
PAD = 48  # black margin either side of a line, so horizontal shifts reveal black, as on a TV

# FCC YIQ, on gamma-encoded RGB
_RGB2YIQ = torch.tensor(
    [[0.299, 0.587, 0.114], [0.595716, -0.274453, -0.321263], [0.211456, -0.522591, 0.311135]]
)
_YIQ2RGB = torch.linalg.inv(_RGB2YIQ)


@dataclass(frozen=True)
class Tape:
    """One VHS/NTSC chain. Bandwidths in MHz, times in microseconds, amounts 0..1."""

    #: Pass through NTSC composite (dot crawl, rainbows). S-Video / component sources skip it.
    composite: bool = True
    #: "notch" (a cheap TV: more artefacts) or "comb" (a better TV: cleaner chroma).
    decoder: str = "notch"
    #: Record onto tape at all (False: broadcast NTSC only).
    tape: bool = True
    luma_mhz: float = 3.0
    chroma_mhz: float = 0.5
    #: Chroma recorded later than luma: colour shifted to the right.
    chroma_delay_us: float = 0.15
    #: Chroma averaged with the previous line (VHS's 1H delay line).
    chroma_vertical: float = 0.5
    sharpen: float = 0.5
    saturation: float = 0.9
    #: Per-line time-base error, standard deviation in us.
    jitter_us: float = 0.04
    #: Slow horizontal waving of the picture's edges, in us.
    edge_wave_us: float = 0.12
    head_switching: bool = True
    #: Mistracking: rolling bands of displaced, noisy lines.
    tracking: float = 0.0
    #: Magnetic dropouts per frame (average).
    dropouts: float = 0.5
    #: Noise standard deviations in 8-bit code values: luma FM noise, chroma noise.
    luma_noise: float = 2.5
    chroma_noise: float = 6.0
    seed: int = 0


PRESETS: dict[str, tuple[str, Tape]] = {
    "broadcast NTSC": (
        "Off-air television: composite artefacts and soft colour, no tape.",
        Tape(
            tape=False,
            luma_mhz=4.2,
            chroma_mhz=1.3,
            chroma_delay_us=0.0,
            chroma_vertical=0.0,
            sharpen=0.15,
            saturation=0.95,
            jitter_us=0.0,
            edge_wave_us=0.0,
            head_switching=False,
            dropouts=0.0,
            luma_noise=1.2,
            chroma_noise=2.0,
        ),
    ),
    "S-VHS": (
        "Better tape: sharper luma, Y/C kept separate (no dot crawl).",
        Tape(
            composite=False,
            luma_mhz=4.6,
            chroma_mhz=0.6,
            chroma_delay_us=0.1,
            sharpen=0.35,
            saturation=0.93,
            jitter_us=0.025,
            edge_wave_us=0.06,
            dropouts=0.2,
            luma_noise=1.5,
            chroma_noise=4.0,
        ),
    ),
    "VHS SP": ("A good VHS recording at standard play.", Tape()),
    "VHS EP (worn)": (
        "Extended play on an old tape: blurrier, noisier, unstable, with dropouts.",
        Tape(
            luma_mhz=2.4,
            chroma_mhz=0.4,
            chroma_delay_us=0.22,
            chroma_vertical=0.7,
            sharpen=0.7,
            saturation=0.8,
            jitter_us=0.08,
            edge_wave_us=0.3,
            tracking=0.25,
            dropouts=3.0,
            luma_noise=4.0,
            chroma_noise=9.0,
        ),
    ),
    "camcorder": (
        "A home-video camcorder: VHS-C at SP, with an oversharpened, slightly hot picture.",
        Tape(
            luma_mhz=2.8,
            chroma_mhz=0.45,
            sharpen=0.9,
            saturation=1.05,
            jitter_us=0.05,
            edge_wave_us=0.15,
            dropouts=0.8,
            luma_noise=3.0,
            chroma_noise=7.0,
        ),
    ),
}


def wear(t: Tape, amount: float) -> Tape:
    """``t`` aged by ``amount`` (0 = as is, 1 = badly worn): less bandwidth, more noise and instability."""
    a = max(0.0, amount)
    return replace(
        t,
        luma_mhz=t.luma_mhz * (1 - 0.3 * min(a, 1)),
        chroma_mhz=t.chroma_mhz * (1 - 0.3 * min(a, 1)),
        jitter_us=t.jitter_us * (1 + 2 * a),
        edge_wave_us=t.edge_wave_us * (1 + 2 * a),
        dropouts=t.dropouts * (1 + 5 * a),
        luma_noise=t.luma_noise * (1 + a),
        chroma_noise=t.chroma_noise * (1 + a),
        saturation=t.saturation * (1 - 0.2 * min(a, 1)),
    )


# ---------------------------------------------------------------- filters on lines (zero-phase, FFT)


def _freqs(n: int, device) -> torch.Tensor:
    """Frequency of each rfft bin in MHz, for lines of n samples at 4 fsc."""
    return torch.fft.rfftfreq(n, d=1 / SAMPLE_RATE, device=device) / 1e6


def _lowpass(f: torch.Tensor, mhz: float, order: int = 3) -> torch.Tensor:
    return 1 / torch.sqrt(1 + (f / max(mhz, 1e-3)) ** (2 * order))


def _filter(x: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    """Filter along the last axis (each line) by a real frequency response."""
    return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * response, n=x.shape[-1], dim=-1)


def _shift(x: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    """Shift every line right by ``samples`` (broadcast to x's lines), fractional, by a phase ramp."""
    n = x.shape[-1]
    k = torch.fft.rfftfreq(n, device=x.device)  # cycles per sample
    ramp = torch.exp(-2j * math.pi * k * samples.unsqueeze(-1))
    return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * ramp, n=n, dim=-1)


# ---------------------------------------------------------------- composite


def _carrier(lines: int, n: int, frame: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of the subcarrier phase at every sample: 227.5 cycles per line -> +pi per line and frame."""
    t = torch.arange(n, device=device, dtype=torch.float32) / SAMPLE_RATE
    line = torch.arange(lines, device=device, dtype=torch.float32).unsqueeze(-1)
    phase = 2 * math.pi * FSC * t + math.pi * line + math.pi * frame + math.radians(33)
    return torch.cos(phase), torch.sin(phase)


def composite(yiq: torch.Tensor, frame: int, decoder: str = "notch") -> torch.Tensor:
    """Encode (3, L, N) YIQ to one composite signal and decode it again, as a TV would."""
    _, lines, n = yiq.shape
    dev = yiq.device
    f = _freqs(n, dev)
    cos, sin = _carrier(lines, n, frame, dev)
    i = _filter(yiq[1], _lowpass(f, 1.3))  # the I and Q bandwidths of the broadcast standard
    q = _filter(yiq[2], _lowpass(f, 0.4))
    signal = yiq[0] + i * cos + q * sin

    if decoder == "comb":
        # The subcarrier inverts each line: the line difference is chroma, the sum is luma.
        prev = torch.cat([signal[:1], signal[:-1]])
        chroma = (signal - prev) / 2
        luma = (signal + prev) / 2
    else:
        # A band-pass around the subcarrier for chroma, the rest for luma: what a notch decoder does.
        # About +-0.6 MHz: chroma sidebands outside it stay in luma (dot crawl), luma inside it is
        # read as colour (rainbows on fine detail).
        band = torch.exp(-(((f - FSC / 1e6) / 0.6) ** 2))
        chroma = _filter(signal, band)
        luma = signal - chroma
    i_out = _filter(2 * chroma * cos, _lowpass(f, 1.3))
    q_out = _filter(2 * chroma * sin, _lowpass(f, 1.3))
    return torch.stack([luma, i_out, q_out])


# ---------------------------------------------------------------- one frame


def _rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng([seed & 0xFFFFFFFF, *keys])


def _line_noise(rng: np.random.Generator, lines: int, rho: float) -> np.ndarray:
    """Unit-variance AR(1) noise down the lines (neighbouring lines' timing errors are related)."""
    z = rng.standard_normal(lines)
    out = np.empty(lines)
    out[0] = z[0]
    for k in range(1, lines):
        out[k] = rho * out[k - 1] + math.sqrt(1 - rho * rho) * z[k]
    return out


def process_frame(
    yiq: torch.Tensor, t: Tape, frame: int, *, bake_noise: bool = False
) -> torch.Tensor:
    """The chain on one (3, LINES, SAMPLES + 2 PAD) YIQ frame; returns the same shape."""
    dev = yiq.device
    _, lines, n = yiq.shape
    f = _freqs(n, dev)
    us = SAMPLE_RATE * 1e-6  # samples per microsecond
    rng = _rng(t.seed, 21, frame)

    if t.composite:
        yiq = composite(yiq, frame, t.decoder)

    if t.tape:
        y = _filter(yiq[0], _lowpass(f, t.luma_mhz, 4))
        c = yiq[1:]
        c = _filter(c, _lowpass(f, t.chroma_mhz, 2))
        if t.chroma_delay_us:
            c = _shift(c, torch.full((lines,), t.chroma_delay_us * us, device=dev))
        if t.chroma_vertical:
            prev = torch.cat([c[:, :1], c[:, :-1]], 1)
            c = c * (1 - 0.5 * t.chroma_vertical) + prev * 0.5 * t.chroma_vertical
        if t.sharpen:
            # Playback peaking: boost the luma's upper band -> crisp-looking edges with halos.
            band = torch.exp(-(((f - 0.6 * t.luma_mhz) / (0.35 * t.luma_mhz)) ** 2))
            y = y + _filter(y, band) * 1.2 * t.sharpen
        yiq = torch.cat([y.unsqueeze(0), c])

    yiq[1:] = yiq[1:] * t.saturation

    # time-base error: each line starts a little early or late
    offset = np.zeros(lines)
    if t.jitter_us:
        offset += _line_noise(rng, lines, 0.6) * t.jitter_us * us
    if t.edge_wave_us:
        phase = _rng(t.seed, 22).uniform(0, 2 * math.pi)
        rows = np.arange(lines)
        offset += (
            np.sin(rows / lines * 2 * math.pi * 1.3 + phase + frame * 0.35) * t.edge_wave_us * us
        )
    if t.tape and t.head_switching:
        # The last ~6 lines are read by the other head before the switch settles: skewed right.
        k = 7
        offset[-k:] += np.linspace(0, 1, k) ** 1.5 * (3.0 + rng.uniform(0, 2)) * us
    band_rows = []
    if t.tracking > 0:
        period = 90 + 150 * (1 - t.tracking)  # frames for a band to roll through
        pos = ((frame / period) + _rng(t.seed, 23).uniform()) % 1.6 - 0.3
        height = int(8 + 40 * t.tracking)
        centre = int(pos * lines)
        band_rows = [r for r in range(centre - height // 2, centre + height // 2) if 0 <= r < lines]
        for r in band_rows:
            offset[r] += rng.normal(0, 4 + 12 * t.tracking) * us * 0.1 + 6 * t.tracking * us
    if offset.any():
        yiq = _shift(yiq, torch.tensor(offset, device=dev, dtype=torch.float32).expand(3, lines))

    # mistracking noise and dropouts are sparse and structural: drawn into the picture
    if band_rows:
        rows = torch.tensor(band_rows, device=dev)
        snow = torch.tensor(
            rng.standard_normal((len(band_rows), n)), device=dev, dtype=torch.float32
        )
        streak = torch.tensor(rng.random((len(band_rows), n)) < 0.02 * t.tracking, device=dev)
        yiq[0, rows] = yiq[0, rows] * 0.8 + snow * 0.12 * t.tracking + streak * 0.7
        yiq[1:, rows] *= 0.4
    for _ in range(rng.poisson(t.dropouts) if t.tape else 0):
        r = int(rng.integers(0, lines))
        x0 = int(rng.integers(PAD, n - PAD))
        length = int(rng.integers(8, 90))
        seg = slice(x0, min(n - PAD, x0 + length))
        if rng.random() < 0.5 and r > 0:
            yiq[:, r, seg] = yiq[
                :, r - 1, seg
            ]  # the VCR's dropout compensation repeats the line above
        else:
            yiq[0, r, seg] = 0.85 + 0.1 * torch.tensor(
                rng.random(seg.stop - seg.start), device=dev, dtype=torch.float32
            )
            yiq[1:, r, seg] = 0

    if bake_noise and (t.luma_noise or t.chroma_noise):
        ny = torch.tensor(rng.standard_normal((lines, n)), device=dev, dtype=torch.float32)
        nc = torch.tensor(rng.standard_normal((2, lines, n)), device=dev, dtype=torch.float32)
        yiq[0] += _filter(ny, _lowpass(f, t.luma_mhz)) * t.luma_noise / 255 * 1.6
        # Chroma noise is band-limited to the colour-under bandwidth: long horizontal smears.
        yiq[1:] += _filter(nc, _lowpass(f, t.chroma_mhz)) * t.chroma_noise / 255 * 3.5
    return yiq


# ---------------------------------------------------------------- frames in, frames out


def to_raster(rgb: torch.Tensor) -> torch.Tensor:
    """(N, H, W, 3) sRGB -> (N, 3, LINES, SAMPLES + 2 PAD) YIQ on the NTSC raster, black margins."""
    x = rgb[..., :3].permute(0, 3, 1, 2).float()
    x = F.interpolate(
        x, size=(LINES, SAMPLES), mode="bilinear", antialias=True, align_corners=False
    )
    yiq = torch.einsum("ij,njhw->nihw", _RGB2YIQ.to(x.device), x)
    return F.pad(yiq, (PAD, PAD))


def from_raster(yiq: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Back to (N, H, W, 3) sRGB at the original size (the raster's softness is the format's)."""
    yiq = yiq[..., PAD:-PAD]
    rgb = torch.einsum("ij,njhw->nihw", _YIQ2RGB.to(yiq.device), yiq)
    rgb = F.interpolate(rgb, size=size, mode="bicubic", align_corners=False)
    return rgb.clamp(0, 1).permute(0, 2, 3, 1)


def crt(frames: torch.Tensor, curvature: float, corner: float = 0.5) -> torch.Tensor:
    """A curved CRT screen on (N, H, W, 3) frames: barrel bulge, rounded tube corners, edge falloff.

    The picture is mapped as if seen on a convex face plate: each point samples the image at
    (1 + k r^2) times its position, so straight edges bow outwards into a rounded, bulging shape inside
    black. Static, so it costs the encoder almost nothing.
    """
    if curvature <= 0:
        return frames
    n, h, w, _ = frames.shape
    dev = frames.device
    k = 0.12 * curvature
    yy = torch.linspace(-1, 1, h, device=dev).view(h, 1).expand(h, w)
    xx = torch.linspace(-1, 1, w, device=dev).view(1, w).expand(h, w)
    r2 = xx * xx + yy * yy
    # Divided by 1 + k so the bulge just touches the frame at the middle of each edge.
    sx, sy = xx * (1 + k * r2) / (1 + k), yy * (1 + k * r2) / (1 + k)
    grid = torch.stack([sx, sy], -1).unsqueeze(0).expand(n, h, w, 2)
    out = F.grid_sample(
        frames.permute(0, 3, 1, 2).float(),
        grid,
        mode="bicubic",
        padding_mode="zeros",
        align_corners=True,
    )
    # The tube's rounded corners, in the picture's own (warped) coordinates, with a soft edge.
    radius = 0.06 + 0.2 * corner
    ax, ay = sx.abs(), sy.abs()
    soft = 3.0 / h
    dx = (ax - (1 - radius * h / w)).clamp_min(0) * (w / h)
    dy = (ay - (1 - radius)).clamp_min(0)
    inside = ((radius - torch.sqrt(dx * dx + dy * dy)) / soft).clamp(0, 1)
    inside = inside * ((1 - ax) / soft).clamp(0, 1) * ((1 - ay) / soft).clamp(0, 1)
    # A curved face plate is a little darker towards its edges.
    falloff = 1 - 0.25 * curvature * (sx * sx * 0.6 + sy * sy * 0.4)
    out = out * (inside * falloff.clamp(0, 1)).view(1, 1, h, w)
    return out.clamp(0, 1).permute(0, 2, 3, 1).to(frames.dtype)


def apply(
    frames: torch.Tensor,
    t: Tape,
    *,
    frame_offset: int = 0,
    bake_noise: bool = False,
    crt_curvature: float = 0.0,
    crt_corner: float = 0.5,
) -> torch.Tensor:
    """The tape on (N, H, W, 3) frames; same shape out, on the same device. Optionally on a curved CRT."""
    n, h, w, _ = frames.shape
    raster = to_raster(frames)
    out = torch.stack(
        [process_frame(raster[i], t, frame_offset + i, bake_noise=bake_noise) for i in range(n)]
    )
    return crt(from_raster(out, (h, w)), crt_curvature, crt_corner).to(frames.dtype)


def noise_table(t: Tape, width: int, height: int) -> grain_tables.Table | None:
    """The tape's noise as an AV1 film grain table: luma FM noise and separate, streakier chroma noise.

    AV1 synthesises it at the output resolution; its AR filter reaches only 3 samples, so the chroma
    streaks are shorter than a real tape's -- choose baked noise for the exact look (at a file-size cost).
    """
    if t.luma_noise <= 0 and t.chroma_noise <= 0:
        return None
    lag = 3
    taps = grain_tables._neighbourhood(lag)
    # Horizontal correlation only (the same line, samples to the left): streaks along the lines.
    luma = tuple(48 if (dr, dc) == (0, -1) else 24 if (dr, dc) == (0, -2) else 0 for dr, dc in taps)
    chroma = tuple(
        96
        if (dr, dc) == (0, -1)
        else 40
        if (dr, dc) == (0, -2)
        else 12
        if (dr, dc) == (0, -3)
        else 0
        for dr, dc in taps
    )
    std_y = t.luma_noise / grain_tables._ar_gain(luma, lag, 8)
    std_c = t.chroma_noise / grain_tables._ar_gain(chroma, lag, 8)
    # Luma and chroma share one scaling_shift: the finest that fits the stronger of the two.
    shift, _ = grain_tables._quantize([0, 1], [std_y, std_c])

    def flat(std: float) -> tuple[tuple[int, int], ...]:
        v = min(255, round(grain_tables.SCALING_PER_STD * std * (1 << (shift - 8))))
        return ((0, v), (255, v))

    # Chroma scaling is indexed by a mix of luma and chroma; flat points make it one constant level.
    y_points, c_points = flat(std_y), flat(std_c)
    p = grain_tables.GrainParams(
        random_seed=(t.seed % 65535) + 1,
        ar_coeff_lag=lag,
        ar_coeff_shift=8,
        scaling_shift=shift,
        cb_mult=128,
        cb_luma_mult=192,
        cb_offset=256,
        cr_mult=128,
        cr_luma_mult=192,
        cr_offset=256,
        scaling_points_y=y_points if t.luma_noise > 0 else (),
        scaling_points_cb=c_points if t.chroma_noise > 0 else (),
        scaling_points_cr=c_points if t.chroma_noise > 0 else (),
        ar_coeffs_y=luma,
        ar_coeffs_cb=(*chroma, 0),
        ar_coeffs_cr=(*chroma, 0),
    )
    return grain_tables.Table.single(p)
