"""A film look for clean frames: tone, colour, LUTs and optics, in torch. No ComfyUI imports.

Everything here is smooth, low-frequency change -- cheap for a video encoder -- so it is applied to
the pixels; grain, the one expensive part, stays in the AV1 bitstream (see grain.py). The stages run
roughly in the order light meets them:

    lens        chromatic aberration, softness, bloom, vignette      (linear light)
    film        halation, exposure, white balance                    (linear light)
    print       tone curve: contrast, toe, highlight roll-off        (display values)
    grade       saturation, split toning, dye crosstalk, bleach bypass, fade
    display     a .cube LUT
    projection  flicker, gate weave

Sizes (blur radii, weave) are fractions of the frame height, so a look holds at any resolution.
Frames are IMAGE tensors: (N, H, W, 3) float in 0..1, sRGB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- settings


@dataclass(frozen=True)
class Grade:
    """Tone and colour: what a built-in look sets. Every default leaves the image unchanged."""

    #: Slope of the tone curve at the pivot: 1 = unchanged, >1 = punchier.
    contrast: float = 1.0
    #: 0..1. Bends the curve smoothly into white: the soft shoulder of film.
    highlight_rolloff: float = 0.0
    #: 0..1. Rounds the curve into black: a soft toe.
    shadow_toe: float = 0.0
    #: Lifts black to this value (a faded print), 0..0.3.
    fade: float = 0.0
    #: Lowers white to 1 - this, 0..0.2.
    highlight_fade: float = 0.0
    saturation: float = 1.0
    #: Warm (+) / cool (-) white balance, in mireds/100-ish units: -1..1.
    temperature: float = 0.0
    #: Magenta (+) / green (-), -1..1.
    tint: float = 0.0
    #: RGB added in the shadows / highlights (split toning), each component about -0.1..0.1.
    shadow_tint: tuple[float, float, float] = (0.0, 0.0, 0.0)
    highlight_tint: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: 0..1. Film dyes absorb outside their band: each channel takes a little of the others.
    crosstalk: float = 0.0
    #: 0..1. Silver retained in the print: desaturated, contrasty, metallic.
    bleach_bypass: float = 0.0


LOOKS: dict[str, tuple[str, Grade]] = {
    "none": ("No grade: tone and colour unchanged.", Grade()),
    "neutral film": (
        "Film's tone response only: soft toe and shoulder, colour untouched.",
        Grade(contrast=1.08, highlight_rolloff=0.6, shadow_toe=0.35, fade=0.02, crosstalk=0.08),
    ),
    "warm print": (
        "A warm print: golden highlights, slightly cool shadows, gentle saturation.",
        Grade(
            contrast=1.1,
            highlight_rolloff=0.65,
            shadow_toe=0.4,
            fade=0.035,
            saturation=0.92,
            temperature=0.22,
            shadow_tint=(-0.01, 0.0, 0.015),
            highlight_tint=(0.025, 0.012, -0.02),
            crosstalk=0.15,
        ),
    ),
    "teal & orange": (
        "The blockbuster split: teal shadows, orange skin and highlights.",
        Grade(
            contrast=1.15,
            highlight_rolloff=0.55,
            shadow_toe=0.35,
            fade=0.02,
            saturation=1.05,
            temperature=0.05,
            shadow_tint=(-0.03, 0.01, 0.035),
            highlight_tint=(0.035, 0.012, -0.03),
            crosstalk=0.1,
        ),
    ),
    "faded 70s": (
        "A faded, warm, low-contrast print with lifted greenish blacks.",
        Grade(
            contrast=0.9,
            highlight_rolloff=0.7,
            shadow_toe=0.2,
            fade=0.09,
            highlight_fade=0.05,
            saturation=0.78,
            temperature=0.3,
            tint=-0.08,
            shadow_tint=(0.0, 0.015, 0.005),
            highlight_tint=(0.02, 0.01, -0.03),
            crosstalk=0.25,
        ),
    ),
    "bleach bypass": (
        "Silver left in the print: desaturated, hard contrast, metallic highlights.",
        Grade(
            contrast=1.2, highlight_rolloff=0.4, shadow_toe=0.3, saturation=0.85, bleach_bypass=0.6
        ),
    ),
    "cool night": (
        "Blue moonlight: cool, deep shadows, restrained colour.",
        Grade(
            contrast=1.12,
            highlight_rolloff=0.6,
            shadow_toe=0.45,
            fade=0.015,
            saturation=0.8,
            temperature=-0.35,
            shadow_tint=(-0.01, 0.0, 0.03),
            highlight_tint=(0.0, 0.005, 0.01),
            crosstalk=0.1,
        ),
    ),
    "vivid slide": (
        "Slide-film punch: deep saturated colour, dense blacks, bright clean whites.",
        Grade(
            contrast=1.22,
            highlight_rolloff=0.5,
            shadow_toe=0.55,
            saturation=1.25,
            temperature=0.05,
            crosstalk=0.05,
        ),
    ),
}


@dataclass(frozen=True)
class Optics:
    """Lens, film and projection effects. Every default is off."""

    #: Stops of exposure, applied in linear light.
    exposure: float = 0.0
    #: Red glow around bright areas (light scattered back through the film), 0..1.
    halation: float = 0.0
    #: Halation radius, as a fraction of the frame height.
    halation_size: float = 0.012
    #: Soft glow of highlights (lens / diffusion filter), 0..1.
    bloom: float = 0.0
    bloom_size: float = 0.04
    #: Loss of fine detail, 0..1 (1 = a blur of ~1.5 px at 1080p).
    softness: float = 0.0
    #: Darkening towards the corners, 0..1.
    vignette: float = 0.0
    #: Colour fringing towards the edges, 0..1.
    chromatic_aberration: float = 0.0
    #: Frame-to-frame drift of the image in the gate, 0..1.
    gate_weave: float = 0.0
    #: Frame-to-frame exposure flutter, 0..1.
    flicker: float = 0.0
    #: Deband the shadows, 0..1: smooth the blotches and steps a compressed source leaves in flat
    #: dark areas into gradients, leaving edges and texture alone.
    clean_shadows: float = 0.0
    seed: int = 0


# ---------------------------------------------------------------- colour helpers

_LUMA = (0.2126, 0.7152, 0.0722)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def luma(x: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(_LUMA, device=x.device, dtype=x.dtype)
    return (x * w).sum(-1, keepdim=True)


# ---------------------------------------------------------------- blur


def _gaussian_kernel(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    t = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of (N, C, H, W). Wide radii run on a downscaled copy (same look, a fraction of the work)."""
    if sigma < 0.3:
        return x
    h, w = x.shape[-2:]
    factor = 1
    while sigma / factor > 6 and min(h, w) // (factor * 2) >= 8:
        factor *= 2
    small = F.avg_pool2d(x, factor) if factor > 1 else x
    s = sigma / factor
    k = _gaussian_kernel(s, x.device, x.dtype)
    c = small.shape[1]
    pad = k.numel() // 2
    kh = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    kv = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    small = F.conv2d(F.pad(small, (pad, pad, 0, 0), mode="replicate"), kh, groups=c)
    small = F.conv2d(F.pad(small, (0, 0, pad, pad), mode="replicate"), kv, groups=c)
    if factor > 1:
        small = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    return small


# ---------------------------------------------------------------- tone + grade


def _soft_clip(y: torch.Tensor, start: float) -> torch.Tensor:
    """Identity below ``start``; above it, an exponential approach to 1 with matching slope."""
    if start >= 1:
        return y.clamp(max=1)
    span = 1 - start
    over = (y - start).clamp_min(0)
    return torch.where(y > start, start + span * (1 - torch.exp(-over / span)), y)


def _contrast(x: torch.Tensor, c: float, pivot: float) -> torch.Tensor:
    """Contrast as a power about the pivot: slope ``c`` there, 0 and 1 fixed.

    Below the pivot y = p (x/p)^c scales shadows by ratio -- a dark face stays a darker face -- where a
    straight line through the pivot would subtract the same amount from every shadow and crush a
    low-key picture into black. Above it the same, mirrored.
    """
    lo = pivot * (x / pivot).clamp_min(1e-8) ** c
    hi = 1 - (1 - pivot) * ((1 - x) / (1 - pivot)).clamp_min(1e-8) ** c
    return torch.where(x < pivot, lo, hi)


def _curve(x: torch.Tensor, g: Grade, pivot: float) -> torch.Tensor:
    y = _contrast(x, g.contrast, pivot)
    y = _soft_clip(y, 1 - 0.6 * g.highlight_rolloff)  # shoulder: the top ~60 % at most
    # Toe: only the bottom few percent, so shadow detail keeps its separation.
    return 1 - _soft_clip(1 - y, 1 - 0.12 * g.shadow_toe)


def tone_curve(x: torch.Tensor, g: Grade, pivot: float = 0.45) -> torch.Tensor:
    """Film's S-shaped response on display values.

    Contrast is a power curve about the pivot (shadows and highlights keep their ratios); the
    shoulder bends highlights smoothly into white and the toe softens the last few percent into
    black, instead of digital's hard clip. Rescaled so black stays black and white stays white --
    fade lifts them on purpose, later.
    """
    if g.contrast == 1 and not g.highlight_rolloff and not g.shadow_toe:
        return x.clamp(0, 1)
    ends = _curve(torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype), g, pivot)
    y = _curve(x.clamp(0, 1), g, pivot)
    return ((y - ends[0]) / (ends[1] - ends[0])).clamp(0, 1)


def white_balance(lin: torch.Tensor, temperature: float, tint: float) -> torch.Tensor:
    """Warm/cool and magenta/green as channel gains in linear light, keeping luminance."""
    if not temperature and not tint:
        return lin
    gains = torch.tensor(
        [
            math.exp(0.25 * temperature + 0.08 * tint),
            math.exp(-0.16 * tint),
            math.exp(-0.35 * temperature + 0.08 * tint),
        ],
        device=lin.device,
        dtype=lin.dtype,
    )
    gains = gains / (gains * torch.tensor(_LUMA, device=lin.device, dtype=lin.dtype)).sum()
    return lin * gains


def grade(d: torch.Tensor, g: Grade) -> torch.Tensor:
    """Display-referred colour: crosstalk, saturation, split toning, bleach bypass, fade."""
    if g.crosstalk:
        a = 0.12 * g.crosstalk
        m = torch.tensor(
            [[1 - 2 * a, a, a], [a, 1 - 2 * a, a], [a, a, 1 - 2 * a]],
            device=d.device,
            dtype=d.dtype,
        )
        d = d @ m.T
    y = luma(d)
    if g.saturation != 1.0:
        d = y + (d - y) * g.saturation
    if any(g.shadow_tint) or any(g.highlight_tint):
        yc = y.clamp(0, 1)
        sh = torch.tensor(g.shadow_tint, device=d.device, dtype=d.dtype)
        hl = torch.tensor(g.highlight_tint, device=d.device, dtype=d.dtype)
        d = d + sh * (1 - yc) ** 2 + hl * yc**2
    if g.bleach_bypass:
        # Overlay the image with its own luminance: silver adds density and takes colour.
        yc = luma(d).clamp(0, 1)
        overlay = torch.where(yc < 0.5, 2 * d * yc, 1 - 2 * (1 - d) * (1 - yc))
        mono = yc.expand_as(d)
        bb = overlay * 0.6 + mono * 0.4
        d = d + (bb - d) * g.bleach_bypass
    if g.fade or g.highlight_fade:
        d = g.fade + d.clamp(0, 1) * (1 - g.fade - g.highlight_fade)
    return d.clamp(0, 1)


def blend_grade(a: Grade, strength: float) -> Grade:
    """``a`` at ``strength`` (0 = no grade, 1 = as defined, >1 = exaggerated)."""
    neutral = Grade()
    out = {}
    for f in fields(Grade):
        x, n = getattr(a, f.name), getattr(neutral, f.name)
        if isinstance(x, tuple):
            out[f.name] = tuple(nv + (xv - nv) * strength for xv, nv in zip(x, n, strict=True))
        else:
            out[f.name] = n + (x - n) * strength
    return replace(neutral, **out)


# ---------------------------------------------------------------- LUTs


class LutError(ValueError):
    pass


@dataclass(frozen=True)
class Lut:
    #: (N, N, N, 3) indexed [b][g][r] for a 3D LUT, or (N, 3) for a 1D one.
    table: torch.Tensor
    domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    title: str = ""

    @property
    def is_3d(self) -> bool:
        return self.table.ndim == 4


def parse_cube(text: str) -> Lut:
    """An Adobe / Resolve ``.cube`` LUT (1D or 3D). Raises LutError naming the problem."""
    size3 = size1 = None
    dmin, dmax, title = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), ""
    rows = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head = line.split()[0].upper()
        try:
            if head == "TITLE":
                title = line[5:].strip().strip('"')
            elif head == "LUT_3D_SIZE":
                size3 = int(line.split()[1])
            elif head == "LUT_1D_SIZE":
                size1 = int(line.split()[1])
            elif head == "DOMAIN_MIN":
                dmin = tuple(float(v) for v in line.split()[1:4])
            elif head == "DOMAIN_MAX":
                dmax = tuple(float(v) for v in line.split()[1:4])
            elif head in ("LUT_1D_INPUT_RANGE", "LUT_3D_INPUT_RANGE"):
                lo, hi = (float(v) for v in line.split()[1:3])
                dmin, dmax = (lo,) * 3, (hi,) * 3
            elif head[0].isalpha():
                continue  # other keywords (e.g. Resolve's LUT_IN_VIDEO_RANGE) do not change the table
            else:
                values = [float(v) for v in line.split()]
                if len(values) != 3:
                    raise LutError(f"line {n}: expected 3 values, found {len(values)}")
                rows.append(values)
        except (ValueError, IndexError) as exc:
            if isinstance(exc, LutError):
                raise
            raise LutError(f"line {n}: cannot read {raw.strip()!r}") from None
    if size3 is None and size1 is None:
        raise LutError("no LUT_3D_SIZE or LUT_1D_SIZE: not a .cube LUT")
    if len(dmin) != 3 or len(dmax) != 3 or any(hi <= lo for lo, hi in zip(dmin, dmax, strict=True)):
        raise LutError("DOMAIN_MIN must be below DOMAIN_MAX in every channel")
    data = torch.tensor(rows, dtype=torch.float32)
    if size3 is not None:
        if not 2 <= size3 <= 256:
            raise LutError(f"LUT_3D_SIZE {size3} is outside 2..256")
        if len(rows) != size3**3:
            raise LutError(f"LUT_3D_SIZE {size3} needs {size3**3} entries, found {len(rows)}")
        table = data.view(size3, size3, size3, 3)  # red varies fastest -> [b][g][r]
    else:
        if not 2 <= size1 <= 65536:
            raise LutError(f"LUT_1D_SIZE {size1} is outside 2..65536")
        if len(rows) != size1:
            raise LutError(f"LUT_1D_SIZE {size1} needs {size1} entries, found {len(rows)}")
        table = data
    if not torch.isfinite(table).all():
        raise LutError("the LUT contains non-finite values")
    return Lut(table, dmin, dmax, title)


def parse_hald(pixels) -> Lut:
    """A Hald CLUT image (as used by RawTherapee, darktable, G'MIC): (S, S, 3) uint8/uint16 or float.

    A level-L Hald image is L^3 x L^3 pixels holding an L^2-point cube, read row by row with red
    varying fastest, then green, then blue -- the same order as a .cube file.
    """
    t = torch.tensor(pixels)  # a copy: image arrays are often read-only
    if t.ndim != 3 or t.shape[-1] < 3:
        raise LutError(f"a Hald CLUT is an RGB image; got shape {tuple(t.shape)}")
    h, w = t.shape[:2]
    level = round(w ** (1 / 3))
    if h != w or level**3 != w or not 2 <= level <= 16:
        raise LutError(
            f"a Hald CLUT image is square with a side of L^3 pixels (512, 1728, 4096); got {w}x{h}"
        )
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    elif t.dtype in (torch.int32, torch.uint16) or t.max() > 1.5:
        t = t.float() / 65535.0
    n = level * level
    return Lut(
        t[..., :3].float().reshape(n, n, n, 3).contiguous(), title=f"Hald CLUT level {level}"
    )


def apply_lut(
    d: torch.Tensor, lut: Lut, strength: float = 1.0, shadow_protect: float = 0.0
) -> torch.Tensor:
    """Apply to (..., 3) display values, trilinear for 3D LUTs and linear for 1D, blended by strength.

    ``shadow_protect`` (0..1) fades the LUT out in the darkest tones -- a luminance-keyed blend, as a
    colourist would use -- so a LUT made for normally exposed photos does not crush a low-key picture.
    """
    if strength <= 0:
        return d
    lo = torch.tensor(lut.domain_min, device=d.device, dtype=d.dtype)
    hi = torch.tensor(lut.domain_max, device=d.device, dtype=d.dtype)
    u = ((d - lo) / (hi - lo)).clamp(0, 1)
    shape = d.shape
    table = lut.table.to(device=d.device, dtype=d.dtype)
    if lut.is_3d:
        vol = table.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, B, G, R): grid x->R, y->G, z->B
        grid = u.reshape(1, 1, 1, -1, 3) * 2 - 1
        out = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)
        out = out.view(3, -1).T.reshape(shape)
    else:
        n = table.shape[0]
        pos = u * (n - 1)
        i0 = pos.floor().long().clamp(0, n - 2)
        frac = pos - i0
        chans = []
        for c in range(3):
            col = table[:, c]
            a, b = col[i0[..., c]], col[i0[..., c] + 1]
            chans.append(a + (b - a) * frac[..., c])
        out = torch.stack(chans, -1)
    weight = torch.full_like(d[..., :1], strength)
    if shadow_protect > 0:
        y = (
            (d * torch.tensor(_LUMA, device=d.device, dtype=d.dtype))
            .sum(-1, keepdim=True)
            .clamp(0, 1)
        )
        t = (y / 0.3).clamp(0, 1)
        key = t * t * (3 - 2 * t)  # smoothstep: 0 at black, 1 from 30 % up
        weight = weight * (1 - shadow_protect * (1 - key))
    return d + (out - d) * weight


def identity_cube(size: int = 17) -> str:
    """A 3D identity LUT as .cube text -- also the template for exporting a look."""
    lines = ['TITLE "identity"', f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(f"{r / (size - 1):.6f} {g / (size - 1):.6f} {b / (size - 1):.6f}")
    return "\n".join(lines) + "\n"


def grade_to_cube(g: Grade, size: int = 33, title: str = "film look") -> str:
    """The tone + colour part of a look as a 3D .cube LUT, for other apps (no optics: those are spatial)."""
    t = torch.linspace(0, 1, size)
    b, gg, r = torch.meshgrid(t, t, t, indexing="ij")
    rgb = torch.stack([r, gg, b], -1).reshape(-1, 3)
    lin = white_balance(srgb_to_linear(rgb), g.temperature, g.tint)
    out = grade(tone_curve(linear_to_srgb(lin), g), g)
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}"]
    lines += [f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in out.tolist()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- optics


def _radius_grid(h: int, w: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised coordinates (x, y), y in -1..1 and x scaled by the aspect ratio."""
    y = torch.linspace(-1, 1, h, device=device, dtype=dtype).view(h, 1).expand(h, w)
    x = torch.linspace(-1, 1, w, device=device, dtype=dtype).view(1, w).expand(h, w) * (w / h)
    return x, y


def clean_shadows(lin: torch.Tensor, amount: float, h: int) -> torch.Tensor:
    """Deband flat dark areas of (N, 3, H, W) linear light.

    A compressed source leaves its shadows as patches of near-constant value with steps between them;
    lifted by a look, they read as blotches. Where the picture is both dark and locally flat -- the
    difference from a wide blur is within a few code values -- it is blended towards that blur, which
    turns the patches into smooth gradients. Edges and real texture differ from the blur by more and
    are kept. Weighted to the shadows, so mid-tones and highlights are untouched.
    """
    d = linear_to_srgb(lin)
    # Wide enough to span a source macroblock (8-16 px) at any resolution.
    wide = blur(d, max(4.0, 0.006 * h * (0.5 + amount)))
    diff = (d - wide).abs().amax(1, keepdim=True)
    # What counts as "flat": within a few 8-bit code values of the blur. Full smoothing below half the
    # threshold, fading out to none at it -- source blotches step by 1-4 values, texture by more.
    thr = (2.0 + 4.0 * amount) / 255
    flat = ((thr - diff) / (0.5 * thr)).clamp(0, 1)
    y = (d * torch.tensor(_LUMA, device=d.device, dtype=d.dtype).view(1, 3, 1, 1)).sum(
        1, keepdim=True
    )
    dark = (1 - (y / 0.25)).clamp(0, 1)
    w = flat * dark * min(1.0, amount * 1.25)
    return srgb_to_linear(d + (wide - d) * w)


def vignette(lin: torch.Tensor, amount: float) -> torch.Tensor:
    """Natural falloff (cos^4-like) towards the corners, in linear light. lin: (N, 3, H, W)."""
    if amount <= 0:
        return lin
    h, w = lin.shape[-2:]
    x, y = _radius_grid(h, w, lin.device, lin.dtype)
    corner = math.hypot(w / h, 1)
    r = torch.sqrt(x * x + y * y) / corner
    falloff = 1 - amount * 0.75 * r**2.5
    return lin * falloff.clamp(0, 1)


def chromatic_aberration(lin: torch.Tensor, amount: float) -> torch.Tensor:
    """Red magnified, blue shrunk a little about the centre: fringes grow towards the edges."""
    if amount <= 0:
        return lin
    n, _, h, w = lin.shape
    yy = torch.linspace(-1, 1, h, device=lin.device, dtype=lin.dtype).view(h, 1).expand(h, w)
    xx = torch.linspace(-1, 1, w, device=lin.device, dtype=lin.dtype).view(1, w).expand(h, w)
    scale = 0.004 * amount  # 0.4% at full strength: ~4 px at the corner of a 1080p frame
    out = lin.clone()
    for c, s in ((0, 1 - scale), (2, 1 + scale)):
        grid = torch.stack([xx * s, yy * s], -1).unsqueeze(0).expand(n, h, w, 2)
        out[:, c : c + 1] = F.grid_sample(
            lin[:, c : c + 1], grid, mode="bilinear", padding_mode="border", align_corners=True
        )
    return out


def _smooth_noise(n: int, seed: int, dims: int, correlation: float = 0.7) -> torch.Tensor:
    """(n, dims) random walk that drifts frame to frame instead of jumping (an AR(1) process)."""
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, dims, generator=gen)
    out = torch.empty_like(z)
    state = z[0]
    for i in range(n):
        state = correlation * state + math.sqrt(1 - correlation**2) * z[i] if i else z[0]
        out[i] = state
    return out


def gate_weave(lin: torch.Tensor, amount: float, offsets: torch.Tensor) -> torch.Tensor:
    """Shift each frame by ``offsets`` (N, 2) * amount, zooming in 1% so no edge is exposed."""
    if amount <= 0:
        return lin
    n, _, h, w = lin.shape
    shift = offsets.to(lin.device, lin.dtype) * amount * 0.0025  # +-0.25% of the frame per sigma
    zoom = 1 - 0.01 * min(amount, 1)
    yy = torch.linspace(-1, 1, h, device=lin.device, dtype=lin.dtype).view(1, h, 1).expand(n, h, w)
    xx = torch.linspace(-1, 1, w, device=lin.device, dtype=lin.dtype).view(1, 1, w).expand(n, h, w)
    grid = torch.stack(
        [xx * zoom + shift[:, 0:1, None] * 2, yy * zoom + shift[:, 1:2, None] * 2 * (w / h) * 0.5],
        -1,
    )
    return F.grid_sample(
        lin, grid, mode="bicubic", padding_mode="border", align_corners=True
    ).clamp_min(0)


# ---------------------------------------------------------------- the pipeline


def apply(
    frames: torch.Tensor,
    g: Grade,
    o: Optics,
    *,
    lut: Lut | None = None,
    lut_strength: float = 1.0,
    lut_shadow_protect: float = 0.0,
    frame_offset: int = 0,
    total_frames: int | None = None,
) -> torch.Tensor:
    """The film look on (N, H, W, 3) frames in 0..1; returns the same shape on the same device.

    ``frame_offset`` / ``total_frames`` keep the per-frame randomness (flicker, weave) continuous when
    a long clip is processed in chunks.
    """
    n, h, _w, _ = frames.shape
    total = total_frames or (frame_offset + n)
    x = frames[..., :3].permute(0, 3, 1, 2).float()  # (N, 3, H, W)
    lin = srgb_to_linear(x) * (2.0**o.exposure)

    if o.clean_shadows > 0:
        lin = clean_shadows(lin, o.clean_shadows, h)

    # lens
    lin = chromatic_aberration(lin, o.chromatic_aberration)
    if o.softness > 0:
        lin = blur(lin, 1.5 * o.softness * h / 1080)
    if o.bloom > 0:
        lum = (lin * torch.tensor(_LUMA, device=lin.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
        bright = lin * ((lum - 0.5).clamp_min(0) / lum.clamp_min(1e-4))
        lin = lin + 0.6 * o.bloom * blur(bright, o.bloom_size * h)
    lin = vignette(lin, o.vignette)

    # film: halation is red-orange light bounced off the film base, behind the brightest areas
    if o.halation > 0:
        lum = (lin * torch.tensor(_LUMA, device=lin.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
        # A soft knee from ~0.15 linear (bright mid-tones) up: highlights glow, mid-tones barely.
        knee = ((lum - 0.15) / 0.35).clamp(0, 1)
        source = lum * knee * knee
        glow = blur(source, o.halation_size * h)
        tint = torch.tensor([1.0, 0.32, 0.08], device=lin.device).view(1, 3, 1, 1)
        lin = lin + 1.6 * o.halation * glow * tint
    lin = white_balance(lin.permute(0, 2, 3, 1), g.temperature, g.tint).permute(0, 3, 1, 2)

    # print + grade + display, on display values (N, H, W, 3)
    d = linear_to_srgb(lin).permute(0, 2, 3, 1)
    d = tone_curve(d, g)
    d = grade(d, g)
    if lut is not None:
        d = apply_lut(d, lut, lut_strength, lut_shadow_protect)

    # projection
    if o.flicker > 0 or o.gate_weave > 0:
        noise = _smooth_noise(total, o.seed, 3)[frame_offset : frame_offset + n]
        if o.flicker > 0:
            gain = (1 + 0.035 * o.flicker * noise[:, 2]).to(d.device, d.dtype).view(n, 1, 1, 1)
            d = (d * gain).clamp(0, 1)
        if o.gate_weave > 0:
            d = gate_weave(d.permute(0, 3, 1, 2), o.gate_weave, noise[:, :2]).permute(0, 2, 3, 1)
    return d.clamp(0, 1).to(frames.dtype)
