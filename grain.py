"""AV1 film grain synthesis tables: model, text format, validation and generators. No ComfyUI imports.

An AV1 bitstream can carry film grain *parameters* instead of grain pixels: a piecewise-linear
scaling function (how strong the grain is at each brightness), an auto-regressive filter (how
the grain is textured) and a seed. The decoder synthesises the grain from them on playback, so a
clean encode stays as cheap as a clean encode.

aomenc, SvtAv1EncApp and rav1e read these parameters from the same text file, the "film grain
table" (header ``filmgrn1``) defined by libaom's ``grain_table.c``. This module writes, parses and
validates that format against the ranges the AV1 specification (section 5.9.30, film_grain_params)
allows, and generates tables:

    photon_noise   shot + read noise of a digital sensor at an ISO setting (the physical model
                   libaom's photon_noise_table tool and rav1e's --photon-noise use)
    stylized       an artistic look: strength, tonal response, grain size (AR filter), chroma
    PRESETS        named looks built from the two above

Grain strength throughout is the standard deviation of the synthesised grain in 8-bit code
values; AV1 defines the scaling function on an 8-bit scale whatever the coded bit depth.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace

import numpy as np

HEADER = "filmgrn1"
#: An entry valid for the whole video.
END_OF_TIME = 9223372036854775807

MAX_Y_POINTS = 14
MAX_UV_POINTS = 10
MAX_AR_LAG = 3

#: libaom's photon noise model maps a noise std (8-bit code values) to a scaling value with x 7.88
#: (at scaling_shift 8). Kept for the photon tables so they match aomenc / rav1e / av1an's.
PHOTON_SCALING_PER_STD = 7.88
#: The same mapping measured on dav1d's output (flat grey, 8- and 10-bit alike): a scaling value s
#: gives grain of std s / 6.8. Used where a strength is asked for directly (stylized looks).
SCALING_PER_STD = 6.8

#: Sensor sizes (width, height in mm) for the photon noise model. Smaller sensors have smaller
#: pixels at the same resolution, so fewer photons per pixel and more noise at the same ISO.
SENSORS: dict[str, tuple[float, float]] = {
    "full frame (36x24 mm)": (36.0, 24.0),
    "Super 35 (24.9x14 mm)": (24.89, 14.0),
    "APS-C (23.5x15.6 mm)": (23.5, 15.6),
    "Micro Four Thirds (17.3x13 mm)": (17.3, 13.0),
    "1-inch (13.2x8.8 mm)": (13.2, 8.8),
    "1/2.3-inch, phone-size (6.17x4.55 mm)": (6.17, 4.55),
}
DEFAULT_SENSOR = "full frame (36x24 mm)"


def ar_coefficient_count(lag: int) -> int:
    """Luma AR coefficients for a lag: the causal neighbourhood rows above plus the left half-row."""
    return 2 * lag * (lag + 1)


@dataclass(frozen=True)
class GrainParams:
    """One table entry's film_grain_params, named as in the AV1 specification."""

    random_seed: int = 7391
    apply_grain: bool = True
    update_parameters: bool = True
    ar_coeff_lag: int = 0
    ar_coeff_shift: int = 6
    grain_scale_shift: int = 0
    scaling_shift: int = 8
    chroma_scaling_from_luma: bool = False
    overlap_flag: bool = True
    cb_mult: int = 0
    cb_luma_mult: int = 0
    cb_offset: int = 0
    cr_mult: int = 0
    cr_luma_mult: int = 0
    cr_offset: int = 0
    scaling_points_y: tuple[tuple[int, int], ...] = ()
    scaling_points_cb: tuple[tuple[int, int], ...] = ()
    scaling_points_cr: tuple[tuple[int, int], ...] = ()
    ar_coeffs_y: tuple[int, ...] = ()
    ar_coeffs_cb: tuple[int, ...] = (0,)
    ar_coeffs_cr: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class Entry:
    start_time: int
    end_time: int
    params: GrainParams


@dataclass(frozen=True)
class Table:
    entries: tuple[Entry, ...] = field(default_factory=tuple)

    @classmethod
    def single(cls, params: GrainParams) -> Table:
        return cls((Entry(0, END_OF_TIME, params),))


class TableError(ValueError):
    """A film grain table that an AV1 encoder would reject or misread."""


# ---------------------------------------------------------------- text format


def dumps(table: Table) -> str:
    """The table as libaom's grain_table.c writes it (and aomenc / SvtAv1EncApp / rav1e read it)."""
    validate(table)
    lines = [HEADER]
    for e in table.entries:
        p = e.params
        lines.append(
            f"E {e.start_time} {e.end_time} {int(p.apply_grain)} {p.random_seed} {int(p.update_parameters)}"
        )
        if not p.update_parameters:
            continue
        lines.append(
            "\tp "
            + " ".join(
                str(int(v))
                for v in (
                    p.ar_coeff_lag,
                    p.ar_coeff_shift,
                    p.grain_scale_shift,
                    p.scaling_shift,
                    p.chroma_scaling_from_luma,
                    p.overlap_flag,
                    p.cb_mult,
                    p.cb_luma_mult,
                    p.cb_offset,
                    p.cr_mult,
                    p.cr_luma_mult,
                    p.cr_offset,
                )
            )
        )
        for tag, points in (
            ("sY", p.scaling_points_y),
            ("sCb", p.scaling_points_cb),
            ("sCr", p.scaling_points_cr),
        ):
            lines.append(f"\t{tag} {len(points)}" + "".join(f" {x} {y}" for x, y in points))
        lines.append("\tcY" + "".join(f" {c}" for c in p.ar_coeffs_y))
        # Always written: SvtAv1EncApp fails ("Unable to read Cb coeffs header") without them,
        # even for planes that have no grain.
        lines.append("\tcCb" + "".join(f" {c}" for c in p.ar_coeffs_cb))
        lines.append("\tcCr" + "".join(f" {c}" for c in p.ar_coeffs_cr))
    return "\n".join(lines) + "\n"


class _Tokens:
    def __init__(self, text: str):
        self.items = text.split()
        self.pos = 0

    def peek(self) -> str | None:
        return self.items[self.pos] if self.pos < len(self.items) else None

    def word(self, expected: str | None = None) -> str:
        if self.pos >= len(self.items):
            raise TableError(f"unexpected end of table, expected {expected or 'a value'}")
        value = self.items[self.pos]
        self.pos += 1
        if expected is not None and value != expected:
            raise TableError(f"expected {expected!r}, found {value!r}")
        return value

    def int(self, what: str) -> int:
        if self.pos >= len(self.items):
            raise TableError(f"unexpected end of table, expected {what}")
        value = self.word()
        try:
            return int(value)
        except ValueError:
            raise TableError(f"{what}: {value!r} is not an integer") from None

    def ints(self, n: int, what: str) -> tuple[int, ...]:
        return tuple(self.int(what) for _ in range(n))


def loads(text: str) -> Table:
    """Parse and validate a film grain table. Raises TableError with the offending field."""
    tokens = _Tokens(text)
    if tokens.peek() != HEADER:
        raise TableError(f"not a film grain table: it must start with {HEADER!r}")
    tokens.word()
    entries = []
    previous: GrainParams | None = None
    while tokens.peek() is not None:
        tokens.word("E")
        start, end = tokens.int("start_time"), tokens.int("end_time")
        apply, seed, update = (
            tokens.int("apply_grain"),
            tokens.int("random_seed"),
            tokens.int("update_parameters"),
        )
        n = len(entries) + 1
        if update not in (0, 1) or apply not in (0, 1):
            raise TableError(f"entry {n}: apply_grain and update_parameters must be 0 or 1")
        if not update:
            if previous is None:
                raise TableError("entry 1 must define its parameters (update_parameters = 1)")
            p = replace(
                previous, random_seed=seed, apply_grain=bool(apply), update_parameters=False
            )
        else:
            tokens.word("p")
            (lag, ar_shift, scale_shift, scaling_shift, csfl, overlap, *chroma) = tokens.ints(
                12, "p"
            )
            if not 0 <= lag <= MAX_AR_LAG:
                raise TableError(f"entry {n}: ar_coeff_lag {lag} is outside 0..{MAX_AR_LAG}")
            points = {}
            for tag, limit in (
                ("sY", MAX_Y_POINTS),
                ("sCb", MAX_UV_POINTS),
                ("sCr", MAX_UV_POINTS),
            ):
                tokens.word(tag)
                count = tokens.int(f"{tag} count")
                if not 0 <= count <= limit:
                    raise TableError(f"entry {n}: {tag} has {count} points, AV1 allows 0..{limit}")
                values = tokens.ints(2 * count, f"{tag} point")
                points[tag] = tuple(zip(values[::2], values[1::2], strict=True))
            tokens.word("cY")
            coeffs_y = tokens.ints(ar_coefficient_count(lag), "cY coefficient")
            chroma_n = ar_coefficient_count(lag) + 1
            coeffs = {}
            for tag, plane_points in (("cCb", points["sCb"]), ("cCr", points["sCr"])):
                # Required for a plane with grain; optional otherwise (libaom's writer skips them,
                # SvtAv1EncApp's reader needs them -- dumps always writes them).
                if plane_points or csfl or tokens.peek() == tag:
                    tokens.word(tag)
                    coeffs[tag] = tokens.ints(chroma_n, f"{tag} coefficient")
                else:
                    coeffs[tag] = (0,) * chroma_n
            p = GrainParams(
                random_seed=seed,
                apply_grain=bool(apply),
                update_parameters=True,
                ar_coeff_lag=lag,
                ar_coeff_shift=ar_shift,
                grain_scale_shift=scale_shift,
                scaling_shift=scaling_shift,
                chroma_scaling_from_luma=bool(csfl),
                overlap_flag=bool(overlap),
                cb_mult=chroma[0],
                cb_luma_mult=chroma[1],
                cb_offset=chroma[2],
                cr_mult=chroma[3],
                cr_luma_mult=chroma[4],
                cr_offset=chroma[5],
                scaling_points_y=points["sY"],
                scaling_points_cb=points["sCb"],
                scaling_points_cr=points["sCr"],
                ar_coeffs_y=coeffs_y,
                ar_coeffs_cb=coeffs["cCb"],
                ar_coeffs_cr=coeffs["cCr"],
            )
        entries.append(Entry(start, end, p))
        previous = p
    table = Table(tuple(entries))
    validate(table)
    return table


def _check(cond: bool, where: str, message: str) -> None:
    if not cond:
        raise TableError(f"{where}: {message}")


def _check_points(points, limit: int, where: str) -> None:
    _check(len(points) <= limit, where, f"{len(points)} scaling points, AV1 allows at most {limit}")
    xs = [x for x, _ in points]
    _check(
        all(0 <= x <= 255 and 0 <= y <= 255 for x, y in points), where, "points must be in 0..255"
    )
    _check(
        all(a < b for a, b in itertools.pairwise(xs)),
        where,
        "point x values must strictly increase",
    )


def validate(table: Table) -> None:
    """Every rule the AV1 specification puts on film_grain_params, plus the table's own structure."""
    _check(bool(table.entries), "table", "has no entries")
    last_end = None
    for n, e in enumerate(table.entries, 1):
        where = f"entry {n}"
        p = e.params
        _check(0 <= e.start_time < e.end_time, where, "start_time must be >= 0 and below end_time")
        _check(
            last_end is None or e.start_time >= last_end, where, "entries must not overlap in time"
        )
        last_end = e.end_time
        _check(0 <= p.random_seed <= 0xFFFF, where, "random_seed must fit in 16 bits")
        _check(0 <= p.ar_coeff_lag <= MAX_AR_LAG, where, f"ar_coeff_lag must be 0..{MAX_AR_LAG}")
        _check(6 <= p.ar_coeff_shift <= 9, where, "ar_coeff_shift must be 6..9")
        _check(0 <= p.grain_scale_shift <= 3, where, "grain_scale_shift must be 0..3")
        _check(8 <= p.scaling_shift <= 11, where, "scaling_shift must be 8..11")
        for name in ("cb_mult", "cb_luma_mult", "cr_mult", "cr_luma_mult"):
            _check(0 <= getattr(p, name) <= 255, where, f"{name} must be 0..255")
        for name in ("cb_offset", "cr_offset"):
            _check(0 <= getattr(p, name) <= 511, where, f"{name} must be 0..511")
        _check_points(p.scaling_points_y, MAX_Y_POINTS, f"{where} sY")
        _check_points(p.scaling_points_cb, MAX_UV_POINTS, f"{where} sCb")
        _check_points(p.scaling_points_cr, MAX_UV_POINTS, f"{where} sCr")
        if p.chroma_scaling_from_luma:
            _check(
                not p.scaling_points_cb and not p.scaling_points_cr,
                where,
                "chroma_scaling_from_luma takes no chroma scaling points",
            )
        else:
            # Conformance for 4:2:0: either both chroma planes have grain or neither does.
            _check(
                bool(p.scaling_points_cb) == bool(p.scaling_points_cr),
                where,
                "for 4:2:0 video sCb and sCr must both have points or both be empty",
            )
        n_y = ar_coefficient_count(p.ar_coeff_lag)
        _check(
            len(p.ar_coeffs_y) == n_y,
            where,
            f"cY needs {n_y} coefficients for lag {p.ar_coeff_lag}",
        )
        for tag, coeffs in (("cCb", p.ar_coeffs_cb), ("cCr", p.ar_coeffs_cr)):
            _check(
                len(coeffs) == n_y + 1,
                where,
                f"{tag} needs {n_y + 1} coefficients for lag {p.ar_coeff_lag}",
            )
        _check(
            all(-128 <= c <= 127 for c in (*p.ar_coeffs_y, *p.ar_coeffs_cb, *p.ar_coeffs_cr)),
            where,
            "AR coefficients must be -128..127",
        )


def has_grain(table: Table) -> bool:
    return any(
        e.params.apply_grain
        and (
            e.params.scaling_points_y
            or e.params.scaling_points_cb
            or e.params.chroma_scaling_from_luma
        )
        and any(
            y
            for _, y in e.params.scaling_points_y
            + e.params.scaling_points_cb
            + e.params.scaling_points_cr
        )
        for e in table.entries
    )


# ---------------------------------------------------------------- AR texture


def _neighbourhood(lag: int) -> list[tuple[int, int]]:
    """(row, col) offsets of the AR taps in coefficient order (AV1 spec 7.18.3.3)."""
    taps = []
    for dr in range(-lag, 1):
        for dc in range(-lag, lag + 1):
            if dr == 0 and dc == 0:
                return taps
            taps.append((dr, dc))
    return taps


#: The AR weights' sum at size 1. Swept on dav1d's output: from ~0.7 the grain correlates over
#: distances near AV1's 32x32 grain blocks, and their tiling shows as a repeating mosaic.
MAX_AR_SUM = 0.6
#: Spatial falloff of the AR taps, in samples.
AR_SIGMA = 0.7


def texture_coeffs(size: float, lag: int = MAX_AR_LAG, shift: int = 8) -> tuple[int, ...]:
    """AR coefficients that low-pass the grain: 0 = white (finest) grain, 1 = the coarsest clean grain.

    Each tap is weighted by a Gaussian of its distance and the weights sum to ``MAX_AR_SUM * size``,
    which keeps the recursion stable and the grain free of block mosaic. AV1's filter reaches at most
    3 samples and the grain is tiled from 32x32 blocks, so this is grain *texture* -- a few pixels --
    not large clumps: AV1 grain cannot make those without the tiling showing.
    """
    if size <= 0 or lag == 0:
        return (0,) * ar_coefficient_count(lag)
    taps = _neighbourhood(lag)
    weights = np.array([math.exp(-(dr * dr + dc * dc) / (2 * AR_SIGMA**2)) for dr, dc in taps])
    weights *= min(size, 1.0) * MAX_AR_SUM / weights.sum()
    scale = 1 << shift
    return tuple(int(np.clip(round(w * scale), -128, 127)) for w in weights)


def _ar_gain(coeffs: tuple[int, ...], lag: int, shift: int, trials: int = 4) -> float:
    """Std multiplier the AR filter applies to the grain template, by simulating it on white noise.

    The scaling function multiplies the *filtered* template, so a coarse look needs its scaling
    reduced by this factor to keep the requested strength.
    """
    if not any(coeffs):
        return 1.0
    taps = _neighbourhood(lag)
    rng = np.random.default_rng(0)
    h, w = 73, 82  # the luma grain template AV1 synthesises from
    ratios = []
    for _ in range(trials):
        g = rng.standard_normal((h, w))
        white = g.std()
        for y in range(lag, h):
            for x in range(lag, w - lag):
                g[y, x] += sum(
                    g[y + dr, x + dc] * c for (dr, dc), c in zip(taps, coeffs, strict=True)
                ) / (1 << shift)
        # The decoder reads 32x32 blocks from the template's interior.
        ratios.append(g[9:-9, 9:-9].std() / white)
    return float(np.mean(ratios))


# ---------------------------------------------------------------- generators


def _bt709_to_linear(v: float) -> float:
    return v / 4.5 if v < 0.081 else ((v + 0.099) / 1.099) ** (1 / 0.45)


def _bt709_from_linear(lin: float) -> float:
    return 4.5 * lin if lin < 0.018 else 1.099 * lin**0.45 - 0.099


def _quantize(
    xs: list[int], stds: list[float], per_std: float = SCALING_PER_STD
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """(scaling_shift, scaling points) for grain of these standard deviations at these levels.

    The grain is scaling(x) >> scaling_shift, so a larger shift gives finer steps for weak grain
    (up to 8x at shift 11) and a smaller one reaches stronger grain. Takes the finest shift that
    still fits the strongest point in 0..255.
    """
    for shift in (11, 10, 9, 8):
        values = [round(per_std * sd * (1 << (shift - 8))) for sd in stds]
        if max(values, default=0) <= 255 or shift == 8:
            return shift, tuple(
                (x, int(min(255, max(0, v)))) for x, v in zip(xs, values, strict=True)
            )
    raise AssertionError("unreachable")


def photon_noise(
    iso: float,
    width: int,
    height: int,
    *,
    sensor: str = DEFAULT_SENSOR,
    chroma: bool = False,
    seed: int = 7391,
) -> Table:
    """Sensor noise of a digital camera at ``iso``, at this video's resolution.

    The physical model libaom's ``photon_noise_table`` and rav1e's ``--photon-noise`` use: photon shot
    noise plus read noise and photo-response non-uniformity, for a daylight spectrum and a typical
    2010s sensor, mapped through the BT.709 transfer function. More pixels for the same sensor means
    smaller pixels, each collecting fewer photons, so the same ISO is grainier at higher resolution.

    It models a *digital sensor*, not a film stock: the grain is fine (white). ``sensor`` (a key of
    SENSORS; libaom assumes full frame) sets the pixel area. At video resolutions a full-frame pixel
    collects so many photons that even ISO 3200 is faint, and the photo-response non-uniformity term
    (0.5 % of the signal) dominates -- the model's physics, kept as libaom has it. A smaller sensor
    is how the same model gives visible grain.
    """
    if iso <= 0:
        raise ValueError("ISO must be positive")
    if sensor not in SENSORS:
        raise ValueError(f"sensor must be one of {list(SENSORS)}")
    sensor_w, sensor_h = SENSORS[sensor]
    photons_per_lx_s_um2 = 11260.0  # daylight-like spectrum
    quantum_efficiency = 0.20
    prnu = 0.005
    read_noise = 1.5  # electrons rms, input-referred
    mid_tone_exposure = 10.0 / iso  # lx·s for an 18% grey
    pixel_area = sensor_w * 1000 * sensor_h * 1000 / (width * height)  # µm² per pixel
    mid_electrons = quantum_efficiency * photons_per_lx_s_um2 * mid_tone_exposure * pixel_area
    max_electrons = mid_electrons / 0.18

    xs, stds = [], []
    for i in range(MAX_Y_POINTS):
        x = i / (MAX_Y_POINTS - 1)
        linear = _bt709_to_linear(x)
        electrons = max_electrons * linear
        noise = math.sqrt(read_noise**2 + electrons + (prnu * electrons) ** 2) / max_electrons
        lo, hi = max(0.0, linear - 2 * noise), min(1.0, linear + 2 * noise)
        slope = (_bt709_from_linear(hi) - _bt709_from_linear(lo)) / (hi - lo)
        xs.append(round(255 * x))
        stds.append(255 * noise * slope)
    shift, points = _quantize(xs, stds, PHOTON_SCALING_PER_STD)
    return Table.single(
        GrainParams(
            random_seed=seed,
            scaling_shift=shift,
            scaling_points_y=points,
            chroma_scaling_from_luma=chroma,
            ar_coeffs_cb=(0,),
            ar_coeffs_cr=(0,),
        )
    )


#: How grain strength varies with brightness, as a multiplier of the requested strength.
TONE_CURVES = {
    # The same strength everywhere -- suits flat-shaded animation, where every tone should show it.
    "flat": lambda x: 1.0,
    # Strongest in the mid-tones, fading towards black and white (a common look for film grain).
    "midtones": lambda x: 0.35 + 0.65 * math.sin(math.pi * x) ** 0.8,
    # Strongest in the shadows, as digital noise is.
    "shadows": lambda x: 0.4 + 0.6 * (1.0 - x) ** 1.5,
}


def stylized(
    strength: float,
    *,
    tone: str = "midtones",
    size: float = 0.0,
    chroma: bool = False,
    seed: int = 7391,
) -> Table:
    """An artistic grain look.

    ``strength``: grain standard deviation in 8-bit code values at its strongest tone (AV1 can
    reach about 32 with these settings). ``tone``: a key of TONE_CURVES. ``size``: 0 = fine white
    grain, 1 = the coarsest texture AV1's AR filter gives without block artefacts. ``chroma``: colour grain on the chroma
    planes as well (scaled from luma), otherwise monochrome.
    """
    if tone not in TONE_CURVES:
        raise ValueError(f"tone must be one of {sorted(TONE_CURVES)}")
    lag = MAX_AR_LAG if size > 0 else 0
    shift = 8
    coeffs = texture_coeffs(size, lag, shift)
    gain = _ar_gain(coeffs, lag, shift)
    curve = TONE_CURVES[tone]
    n = 8
    xs = [round(255 * i / (n - 1)) for i in range(n)]
    shift_s, points = _quantize(xs, [strength * curve(i / (n - 1)) / gain for i in range(n)])
    zero_chroma = (0,) * (ar_coefficient_count(lag) + 1)
    chroma_coeffs = (*coeffs, 0) if chroma else zero_chroma
    return Table.single(
        GrainParams(
            random_seed=seed,
            ar_coeff_lag=lag,
            ar_coeff_shift=shift if lag else 6,
            scaling_shift=shift_s,
            scaling_points_y=points,
            chroma_scaling_from_luma=chroma,
            ar_coeffs_y=coeffs,
            ar_coeffs_cb=chroma_coeffs,
            ar_coeffs_cr=chroma_coeffs,
        )
    )


def scale_strength(table: Table, factor: float) -> Table:
    """Every entry's grain made ``factor`` times stronger (the scaling function, clipped to 255)."""
    if factor < 0:
        raise ValueError("strength factor must be >= 0")

    entries = []
    for e in table.entries:
        p = e.params
        planes = (p.scaling_points_y, p.scaling_points_cb, p.scaling_points_cr)
        peak = max((y for pts in planes for _, y in pts), default=0) * factor
        # Re-pick the shift so stronger grain is not clipped and weaker grain keeps its precision.
        shift = p.scaling_shift
        while shift > 8 and peak > 255:
            peak /= 2
            shift -= 1
        while shift < 11 and peak * 2 <= 255 and peak > 0:
            peak *= 2
            shift += 1
        k = factor * (1 << shift) / (1 << p.scaling_shift)
        y, cb, cr = (tuple((x, min(255, round(v * k))) for x, v in pts) for pts in planes)
        entries.append(
            replace(
                e,
                params=replace(
                    p,
                    scaling_shift=shift,
                    scaling_points_y=y,
                    scaling_points_cb=cb,
                    scaling_points_cr=cr,
                ),
            )
        )
    return Table(tuple(entries))


@dataclass(frozen=True)
class Preset:
    description: str
    #: Builds the table for a video of this width and height.
    build: object


#: The ISO presets' sensor: small enough for the model's noise to be visible at video resolutions.
PHONE_SENSOR = "1/2.3-inch, phone-size (6.17x4.55 mm)"

PRESETS: dict[str, Preset] = {
    "Fine 35mm": Preset(
        "Subtle, fine grain strongest in the mid-tones. An artistic look, not a film-stock emulation.",
        lambda w, h, chroma, seed: stylized(
            3.0, tone="midtones", size=0.25, chroma=chroma, seed=seed
        ),
    ),
    "Coarse 16mm": Preset(
        "Pronounced, clumpy grain from AV1's coarsest AR texture, strongest in the mid-tones.",
        lambda w, h, chroma, seed: stylized(
            7.0, tone="midtones", size=1.0, chroma=chroma, seed=seed
        ),
    ),
    "Vintage Anime": Preset(
        "Strong grain, the same strength in every tone so flat cel colours show it; monochrome unless chroma is on.",
        lambda w, h, chroma, seed: stylized(9.0, tone="flat", size=0.6, chroma=chroma, seed=seed),
    ),
    "High ISO 800": Preset(
        "Photon noise of a phone-size (1/2.3-inch) sensor at ISO 800 for this resolution: noticeable, fine.",
        lambda w, h, chroma, seed: photon_noise(
            800, w, h, sensor=PHONE_SENSOR, chroma=chroma, seed=seed
        ),
    ),
    "High ISO 1600": Preset(
        "Photon noise of a phone-size (1/2.3-inch) sensor at ISO 1600 for this resolution: heavy, gritty.",
        lambda w, h, chroma, seed: photon_noise(
            1600, w, h, sensor=PHONE_SENSOR, chroma=chroma, seed=seed
        ),
    ),
    "High ISO 3200": Preset(
        "Photon noise of a phone-size (1/2.3-inch) sensor at ISO 3200 for this resolution: very strong.",
        lambda w, h, chroma, seed: photon_noise(
            3200, w, h, sensor=PHONE_SENSOR, chroma=chroma, seed=seed
        ),
    ),
}


def preset(name: str, width: int, height: int, *, chroma: bool = False, seed: int = 7391) -> Table:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose one of {list(PRESETS)}")
    return PRESETS[name].build(width, height, chroma, seed)


def peak_strength(table: Table) -> float:
    """The strongest luma grain any entry asks for, as a standard deviation in 8-bit code values."""
    return max(
        (
            y / SCALING_PER_STD * (256 / (1 << e.params.scaling_shift))
            for e in table.entries
            for _, y in e.params.scaling_points_y
        ),
        default=0.0,
    )
