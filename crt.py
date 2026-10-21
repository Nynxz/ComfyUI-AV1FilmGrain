"""A CRT display: the picture as seen on a tube, not a filter. No ComfyUI imports.

The image is drawn the way a tube draws it, in linear light:

    raster      the picture is resampled to the tube's line count (240 for consoles and arcade, 480 for
                TV) and each line is drawn by a Gaussian electron beam per gun. A brighter beam is wider,
                so dark lines are thin with black gaps between them and bright lines bloom into each
                other -- the scanlines are visible in the shadows and fade in the highlights.
    convergence the red and blue guns land a fraction of a pixel off green.
    glow        light scattering in the phosphor and the face plate: a tight glow and a wide halation.
    persistence a phosphor fades over time; a bright moving thing leaves a short trail.
    hum bar     a soft dark band rolling slowly down the picture, as when a camera films a TV whose
                refresh doesn't match its frame rate (or mains hum leaks into the set).
    geometry    the curved face plate and rounded tube corners (vhs.crt).
    mask        the phosphor pattern: aperture grille stripes (Trinitron), slot mask (most TVs) or a
                shadow mask of dots (monitors). Applied last, in screen space, so it stays straight
                and fine instead of being resampled into moire.

The scanlines and the mask darken the picture, as on a real tube; it is brought back up by the
pattern's mean and highlights are soft-clipped. Frames are (N, H, W, 3) sRGB in 0..1; the output has
the input's size. Everything but persistence is per frame; persistence carries the previous output
frame, so chunks must run in order (the chain and the node do).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import vhs
from .filmlook import blur, linear_to_srgb, srgb_to_linear

MASKS = ("aperture grille", "slot mask", "shadow mask", "none")


@dataclass(frozen=True)
class Display:
    mask: str = "slot mask"
    mask_strength: float = 0.5  # 0 = no mask, 1 = fully dark between phosphors
    triads: float = 560.0  # phosphor triads across the picture width (sets the mask's size)
    lines: int = 480  # scanlines in the picture height; 0 = the source's own rows
    scanlines: float = 0.6  # 0 = no visible lines, 1 = the beam profile fully
    beam_min: float = 0.22  # beam width (sigma, in lines) of a black line ...
    beam_max: float = 0.45  # ... and of a white one
    convergence: float = 0.3  # red/blue misalignment in pixels at 1080 lines
    glow: float = 0.15
    halation: float = 0.05
    persistence: float = 0.0  # fraction of the last frame still glowing
    hum_bar: float = 0.0  # depth of the dark band rolling down the screen (0..1)
    hum_speed: float = 0.015  # how far it rolls each frame, in screen heights
    curvature: float = 0.0
    corner: float = 0.5


PRESETS: dict[str, tuple[str, Display]] = {
    "consumer TV": (
        "A 1990s living-room set: slot mask, 480 lines, soft beam, a little glow and a curved face.",
        Display(),
    ),
    "classic scanlines": (
        "Just the lines: 240 sharp horizontal scanlines with black gaps and no visible phosphor mask, "
        "the look most people mean by a CRT.",
        Display(
            mask="none",
            lines=240,
            scanlines=1.0,
            beam_min=0.14,
            beam_max=0.36,
            convergence=0.2,
            glow=0.12,
            halation=0.04,
        ),
    ),
    "Trinitron": (
        "Sony's aperture grille: fine vertical phosphor stripes, a flat-ish face, crisp beam.",
        Display(
            mask="aperture grille",
            triads=640,
            beam_min=0.2,
            beam_max=0.42,
            convergence=0.15,
            glow=0.1,
            halation=0.03,
        ),
    ),
    "arcade (240p)": (
        "A low-resolution arcade or console monitor: 240 lines with strong black gaps, coarse slot mask.",
        Display(
            lines=240,
            triads=420,
            scanlines=0.85,
            beam_min=0.18,
            beam_max=0.4,
            mask_strength=0.45,
            convergence=0.4,
            glow=0.2,
            halation=0.06,
        ),
    ),
    "PVM (broadcast)": (
        "A broadcast monitor: aperture grille, sharp 480 lines with clear gaps, little glow.",
        Display(
            mask="aperture grille",
            triads=800,
            scanlines=0.9,
            beam_min=0.16,
            beam_max=0.36,
            convergence=0.05,
            glow=0.06,
            halation=0.02,
            mask_strength=0.35,
        ),
    ),
    "PC monitor": (
        "A 90s computer monitor: fine shadow mask of dots, faint lines (high scan rate).",
        Display(
            mask="shadow mask",
            triads=900,
            lines=0,
            scanlines=0.0,
            convergence=0.2,
            glow=0.08,
            halation=0.03,
            mask_strength=0.4,
        ),
    ),
}


# ---------------------------------------------------------------- raster


def _beam_sigma(d: Display, level: torch.Tensor) -> torch.Tensor:
    # Beam current grows with level; the spot widens roughly with its square root.
    return d.beam_min + (d.beam_max - d.beam_min) * level.clamp(0, 1).sqrt()


@functools.lru_cache(maxsize=16)
def _gain(beam_min: float, beam_max: float) -> float:
    """1 / the light a mid-level (0.5 linear) line gives, averaged over the line pitch."""
    d = Display(beam_min=beam_min, beam_max=beam_max)
    s = float(_beam_sigma(d, torch.tensor(0.5)))
    v = torch.linspace(-0.5, 0.5, 201)
    total = sum(torch.exp(-0.5 * ((v - j) / s) ** 2) for j in range(-3, 4))
    return 1.0 / float(total.mean())


def line_count(d: Display, h: int) -> int:
    """The lines actually drawn: d.lines rounded so each is a whole number of output rows.

    480 lines on 1248 rows is 2.6 rows a line, and a sharp beam then beats against the pixel grid
    into uneven stripes; 416 lines of exactly 3 rows look like 480 and stay even.
    """
    if not d.lines or d.lines >= h:
        return h
    return max(1, round(h / max(1, round(h / d.lines))))


def scanlines(lin: torch.Tensor, d: Display) -> torch.Tensor:
    """(N, 3, H, W) linear light drawn as d.lines beam lines; same shape out."""
    h, w = lin.shape[-2:]
    lines = line_count(d, h)
    rows = h / lines
    # Below 2 rows per line there's no room for a gap; fade the effect out instead of aliasing.
    strength = d.scanlines * min(1.0, max(0.0, rows - 1.0))
    if strength <= 0:
        return lin
    src = F.interpolate(lin, size=(lines, w), mode="area") if lines != h else lin
    # Line centres sit on pixel rows: with an even number of rows a line (2 for 240 lines on 480),
    # centring it between rows would light both equally and draw no gap at all.
    shift = 0.5 / rows if round(rows) % 2 == 0 else 0.0
    v = (torch.arange(h, device=lin.device, dtype=lin.dtype) + 0.5) / rows - 0.5 + shift
    j0 = torch.floor(v).long()
    # Each output row averages the beam over its own height (a box filter), not a point sample: a
    # sharp beam narrower than a row would otherwise alias into uneven stripes.
    half = 0.5 * lines / h
    out = torch.zeros_like(lin)
    for tap in (-1, 0, 1, 2):
        j = (j0 + tap).clamp(0, lines - 1)
        dist = (v - (j0 + tap).to(v.dtype)).view(1, 1, h, 1)
        colour = src.index_select(2, j)
        sigma = _beam_sigma(d, colour) * math.sqrt(2)
        area = torch.erf((dist + half) / sigma) - torch.erf((dist - half) / sigma)
        out += colour * area * (sigma * math.sqrt(math.pi) / (4 * half))
    out = out * _gain(d.beam_min, d.beam_max)
    flat = F.interpolate(src, size=(h, w), mode="bilinear", align_corners=False)
    return flat + (out - flat) * strength


def _hshift(x: torch.Tensor, px: float) -> torch.Tensor:
    """Shift (N, H, W) right by a fractional number of pixels, edges replicated."""
    if abs(px) < 1e-3:
        return x
    i = math.floor(px)
    f = px - i
    pad = abs(i) + 2
    p = F.pad(x.unsqueeze(1), (pad, pad, 0, 0), mode="replicate").squeeze(1)
    w = x.shape[-1]
    a = p[..., pad - i : pad - i + w]
    b = p[..., pad - i - 1 : pad - i - 1 + w]
    return a * (1 - f) + b * f


# ---------------------------------------------------------------- mask


@functools.lru_cache(maxsize=8)
def phosphor_mask(kind: str, triads: float, h: int, w: int, device: str) -> torch.Tensor:
    """The phosphor pattern as (3, H, W) in 0..1, brightest at each phosphor's centre.

    Box-filtered over 4x2 subsamples per pixel, so a pitch of a few pixels stays a pattern instead of
    aliasing into moire.
    """
    pitch = max(3.0, w / max(1.0, triads))  # one R, G, B triad; under 3 px it can't be drawn
    acc = torch.zeros(3, h, w, device=device)
    centres = torch.tensor([1 / 6, 0.5, 5 / 6], device=device).view(3, 1, 1)
    sub_x, sub_y = (0.125, 0.375, 0.625, 0.875), (0.25, 0.75)
    for oy in sub_y:
        y = (torch.arange(h, device=device, dtype=torch.float32) + oy).view(1, h, 1)
        for ox in sub_x:
            x = (torch.arange(w, device=device, dtype=torch.float32) + ox).view(1, 1, w)
            if kind == "shadow mask":
                # Dots in a delta: rows of triads, every other row offset by half a triad.
                row_h = pitch * 0.866
                row = torch.floor(y / row_h)
                xs = x + (row % 2) * pitch / 2
                ry = (y / row_h) % 1 - 0.5
                vert = torch.exp(-0.5 * (ry / 0.3) ** 2)
            elif kind == "slot mask":
                # Stripes broken into slots, staggered by half a slot in alternate triads.
                slot_h = pitch * 1.6
                col = torch.floor(x / pitch)
                ry = ((y + (col % 2) * slot_h / 2) / slot_h) % 1
                vert = ((ry.sub(0.5).abs() * -1 + 0.5) / 0.08).clamp(
                    0, 1
                )  # dark band at the slot ends
                xs = x
            else:  # aperture grille: continuous vertical stripes
                vert = torch.ones_like(y)
                xs = x
            phase = (xs / pitch) % 1
            dist = (phase - centres).abs()
            dist = torch.minimum(dist, 1 - dist)
            acc += torch.exp(-0.5 * (dist / 0.12) ** 2) * vert
    return acc / (len(sub_x) * len(sub_y))


def _shoulder(x: torch.Tensor, knee: float = 0.8) -> torch.Tensor:
    """Soft-clip linear light above ``knee`` into 1."""
    over = (x - knee).clamp_min(0)
    return torch.where(x > knee, knee + (1 - knee) * torch.tanh(over / (1 - knee)), x)


# ---------------------------------------------------------------- apply


def hum_bar(d: Display, frame: int, h: int, device) -> torch.Tensor:
    """(H,) beam intensity for one frame: 1 everywhere but a soft dark band at the frame's phase."""
    y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    offset = (y - frame * d.hum_speed) % 1
    dist = torch.minimum(offset, 1 - offset)  # the band wraps from the bottom back to the top
    return 1 - 0.4 * min(1.0, d.hum_bar) * torch.exp(-0.5 * (dist / 0.1) ** 2)


def apply(
    frames: torch.Tensor,
    d: Display,
    prev: torch.Tensor | None = None,
    *,
    frame_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The tube on (N, H, W, 3) sRGB frames.

    Returns the frames and the last frame's linear light before the mask (pass it back as ``prev``
    for the next chunk, for persistence).
    """
    n, h, w, _ = frames.shape
    lin = srgb_to_linear(frames[..., :3].float()).permute(0, 3, 1, 2)
    if d.convergence > 0:
        px = d.convergence * h / 1080
        lin = torch.stack([_hshift(lin[:, 0], px), lin[:, 1], _hshift(lin[:, 2], -px)], 1)
    lin = scanlines(lin, d)
    if d.hum_bar > 0:
        bars = torch.stack([hum_bar(d, frame_offset + i, h, lin.device) for i in range(n)])
        lin = lin * bars.view(n, 1, h, 1).to(lin.dtype)
    # Glow and halation move light (scattered in the phosphor and the glass), they don't add it.
    if d.glow > 0:
        lin = (lin + d.glow * blur(lin, 0.004 * h)) / (1 + d.glow)
    if d.halation > 0:
        lin = (lin + d.halation * blur(lin, 0.03 * h)) / (1 + d.halation)
    if d.persistence > 0:
        # Each frame keeps what the last one left glowing, where that's brighter: trails, not ghosts.
        out = []
        for i in range(n):
            cur = lin[i]
            if prev is not None:
                cur = torch.maximum(cur, prev.to(cur) * d.persistence)
            out.append(cur)
            prev = cur
        lin = torch.stack(out)
    last = lin[-1].detach()
    lin = lin.permute(0, 2, 3, 1)
    if d.curvature > 0:
        lin = vhs.crt(lin, d.curvature, d.corner, clamp=False)
    if d.mask != "none" and d.mask_strength > 0:
        m = phosphor_mask(d.mask, d.triads, h, w, str(lin.device)).to(lin.dtype)
        m = (1 - d.mask_strength) + d.mask_strength * m
        m = m / m.mean((1, 2), keepdim=True)  # same average light: the tube is brighter to make up
        lin = lin * m.permute(1, 2, 0)
    out = linear_to_srgb(_shoulder(lin)).clamp(0, 1)
    return out.to(frames.dtype), last
