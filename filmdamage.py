"""Film damage for clean frames: dust, hair, scratches, light leaks, cue marks, breath, gate. No ComfyUI imports.

Every element comes from a schedule drawn once for the whole clip from a seed, so a frame looks the
same however the clip is chunked, and long-lived elements behave as they do on film:

    dust        specks and dirt on the print, white (on the negative) or black (on the print);
                each lasts one frame or two
    hair        a fibre: black (in the projector gate) or white (on the negative); stays a second
                or two, trembling, then gone
    scratches   thin vertical lines from the projector, drifting sideways, flickering, for seconds
    light leak  soft orange-red glow fading in and out at the frame's edge
    cue marks   the dot a projectionist watches for, top right, for 4 frames before the end
    breath      the print's density and colour drifting slightly from frame to frame
    gate        the rounded corners of the camera gate; letterbox crops to a wider aspect

Dust, dirt and hairs are real artefacts by default: alpha masks cut from the annotated scans of
FilmDamageSimulator (Ivanova et al., Eurographics 2023; MIT, see assets/), sized for the film
``gauge`` -- the same dust looks bigger on 16 mm and Super 8 because the frame is smaller.

What costs the encoder bits is detail that changes every frame. Dust does, so it is sparse; the rest
persists or is smooth. Frames are IMAGE tensors (N, H, W, 3) in 0..1. Sizes scale with the frame height.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass, replace

import numpy as np
import torch


@dataclass(frozen=True)
class Damage:
    """Every amount is 0..1 and 0 is off."""

    dust: float = 0.0
    #: Share of dust specks that are black (dirt on the print) rather than white (on the negative).
    dust_black: float = 0.3
    #: Size of dust and dirt relative to the scanned originals. The scans' annotators marked the
    #: specks they could see, which skews the set large; 0.5 brings it closer to real dust.
    dust_size: float = 0.5
    hair: float = 0.0
    scratches: float = 0.0
    light_leaks: float = 0.0
    cue_marks: bool = False
    breath: float = 0.0
    gate: float = 0.0
    #: "none", "1.85", "2.00", "2.39": black bars cropping to that aspect ratio.
    letterbox: str = "none"
    #: The film the damage is sized for (a key of GAUGES): dust is the same size whatever the film,
    #: so on a smaller frame it looks bigger.
    gauge: str = "35mm"
    #: Dust, dirt and hairs from real scanned artefacts (assets/damage_sprites.npz); else drawn shapes.
    real_artefacts: bool = True
    seed: int = 0


#: Frame heights in mm: Super 35 / 16 mm / Super 8 camera apertures.
GAUGES = {"35mm": 18.7, "16mm": 7.49, "Super 8": 4.01}

#: The scans the sprites come from: empty 35 mm still frames, 24 mm = 3408 px tall.
SPRITE_SCAN_MM, SPRITE_SCAN_PX = 24.0, 3408
SPRITES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "damage_sprites.npz"
)


@functools.cache
def load_sprites(path: str = SPRITES_PATH) -> dict[str, list[np.ndarray]] | None:
    """Real artefacts (uint8 alpha masks) by kind, from FilmDamageSimulator's annotated scans; None if absent."""
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        out = {}
        for kind in ("dust", "dirt", "short_hair", "long_hair"):
            shapes, data = z[f"{kind}_shapes"], z[f"{kind}_data"]
            sprites, o = [], 0
            for h, w in shapes:
                sprites.append(data[o : o + h * w].reshape(h, w))
                o += int(h) * int(w)
            out[kind] = sprites
    return out


def sprite_scale(d: Damage, frame_height: int) -> float:
    """Sprite pixels -> frame pixels, for this gauge and frame height."""
    return frame_height / SPRITE_SCAN_PX * SPRITE_SCAN_MM / GAUGES.get(d.gauge, GAUGES["35mm"])


LETTERBOX = {"none": None, "1.66": 1.66, "1.85": 1.85, "2.00": 2.0, "2.39": 2.39}


def _rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng([seed & 0xFFFFFFFF, *keys])


# ---------------------------------------------------------------- schedules (whole clip, CPU, small)


@dataclass(frozen=True)
class Speck:
    x: float  # 0..1 of the width
    y: float
    radius: float  # fraction of the height
    aspect: float
    angle: float
    black: bool
    opacity: float
    #: A real artefact: ("dust" | "dirt", index, orientation 0..7, size factor); empty for drawn ones.
    sprite: tuple = ()


@dataclass(frozen=True)
class Hair:
    start: int
    end: int
    points: np.ndarray  # (K, 2) in (x, y) fractions
    width: float  # fraction of the height
    opacity: float
    #: White: the hair was on the negative when the print was made. Black: in the projector gate.
    bright: bool = False
    #: A real hair: ("short_hair" | "long_hair", index, orientation, size factor); empty: drawn.
    sprite: tuple = ()


@dataclass(frozen=True)
class Scratch:
    start: int
    end: int
    x0: float
    drift: float  # width fractions per frame
    width: float  # fraction of the height
    bright: bool
    seed: int


@dataclass(frozen=True)
class Leak:
    start: int
    end: int
    x: float
    y: float
    radius: float
    colour: tuple[float, float, float]
    peak: float


def _pick_sprite(
    rng: np.random.Generator, d: Damage, kinds: tuple[tuple[str, float], ...]
) -> tuple:
    atlas = load_sprites() if d.real_artefacts else None
    if not atlas:
        return ()
    kind = kinds[0][0] if rng.random() < kinds[0][1] else kinds[1][0]
    return (
        kind,
        int(rng.integers(len(atlas[kind]))),
        int(rng.integers(8)),
        float(rng.lognormal(0, 0.2)),
    )


def _frame_specks(d: Damage, f: int, aspect: float) -> list[Speck]:
    rng = _rng(d.seed, 1, f)
    n = rng.poisson(14 * d.dust**1.5 * aspect / (16 / 9))
    return [
        Speck(
            x=rng.random(),
            y=rng.random(),
            radius=float(np.clip(rng.lognormal(math.log(0.0016), 0.55), 0.0006, 0.009)),
            aspect=float(rng.uniform(0.5, 1.0)),
            angle=float(rng.uniform(0, math.pi)),
            black=bool(rng.random() < d.dust_black),
            opacity=float(rng.uniform(0.4, 0.85)),
            sprite=_pick_sprite(rng, d, (("dust", 0.85), ("dirt", 0.15))),
        )
        for _ in range(n)
    ]


def dust_schedule(d: Damage, frame: int, aspect: float) -> list[Speck]:
    """This frame's specks, plus about a third of the previous frame's (dust that stuck a moment)."""
    if d.dust <= 0:
        return []
    specks = _frame_specks(d, frame, aspect)
    if frame > 0:
        stays = _rng(d.seed, 2, frame - 1)  # one draw per speck of the previous frame
        specks += [s for s in _frame_specks(d, frame - 1, aspect) if stays.random() < 0.33]
    return specks


def hair_schedule(d: Damage, total: int) -> list[Hair]:
    if d.hair <= 0:
        return []
    rng = _rng(d.seed, 3)
    hairs, f = [], int(rng.exponential(60 / d.hair))
    while f < total:
        life = int(rng.integers(18, 60))
        k = 14
        length = rng.uniform(0.07, 0.2)
        theta = rng.uniform(0, 2 * math.pi)
        start = np.array([rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)])
        bend = rng.normal(0, 0.45, k).cumsum()
        steps = np.stack([np.cos(theta + bend), np.sin(theta + bend)], 1) * (length / k)
        pts = start + np.vstack([[0, 0], steps.cumsum(0)])
        hairs.append(
            Hair(
                f,
                min(total, f + life),
                pts,
                float(rng.uniform(0.0016, 0.0026)),
                float(rng.uniform(0.6, 0.9)),
                bool(rng.random() < 0.4),
                _pick_sprite(rng, d, (("short_hair", 0.6), ("long_hair", 0.4))),
            )
        )
        f += life + int(rng.exponential(90 / d.hair))
    return hairs


def scratch_schedule(d: Damage, total: int) -> list[Scratch]:
    if d.scratches <= 0:
        return []
    rng = _rng(d.seed, 4)
    out, f = [], int(rng.exponential(40 / d.scratches))
    while f < total:
        life = int(rng.integers(30, 180))
        out.append(
            Scratch(
                f,
                min(total, f + life),
                float(rng.uniform(0.05, 0.95)),
                float(rng.normal(0, 0.0004)),
                float(rng.uniform(0.0016, 0.0035)),
                bool(rng.random() < 0.7),
                int(rng.integers(1 << 30)),
            )
        )
        f += int(rng.exponential(70 / d.scratches))
    return out


def leak_schedule(d: Damage, total: int) -> list[Leak]:
    if d.light_leaks <= 0:
        return []
    rng = _rng(d.seed, 5)
    out, f = [], int(rng.exponential(48 / d.light_leaks))
    while f < total:
        life = int(rng.integers(12, 48))
        edge = rng.integers(4)
        x, y = (
            (rng.uniform(-0.05, 0.15), rng.uniform(0, 1))
            if edge == 0
            else (rng.uniform(0.85, 1.05), rng.uniform(0, 1))
            if edge == 1
            else (rng.uniform(0, 1), rng.uniform(-0.05, 0.15))
            if edge == 2
            else (rng.uniform(0, 1), rng.uniform(0.85, 1.05))
        )
        warm = rng.uniform(0.0, 0.5)
        colour = (1.0, 0.45 + 0.3 * warm, 0.12 + 0.15 * warm)
        out.append(
            Leak(
                f,
                min(total, f + life),
                float(x),
                float(y),
                float(rng.uniform(0.25, 0.55)),
                colour,
                float(rng.uniform(0.5, 1.0)),
            )
        )
        f += life + int(rng.exponential(120 / d.light_leaks))
    return out


def cue_frames(total: int, fps: float) -> set[int]:
    """Changeover cue: 4 frames, about 8 seconds before the end -- or the last 4 frames of a short clip."""
    start = total - 4 - round(8 * fps)
    if start < 0:
        start = max(0, total - 4)
    return set(range(start, min(total, start + 4)))


# ---------------------------------------------------------------- rendering (per frame, torch)


def _alpha_ellipse(h: int, w: int, s: Speck, device) -> tuple[torch.Tensor, slice, slice] | None:
    r = max(0.6, s.radius * h)
    cx, cy = s.x * w, s.y * h
    pad = int(r + 2)
    x0, x1 = max(0, int(cx) - pad), min(w, int(cx) + pad + 1)
    y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = torch.meshgrid(
        torch.arange(y0, y1, device=device) + 0.5 - cy,
        torch.arange(x0, x1, device=device) + 0.5 - cx,
        indexing="ij",
    )
    c, sn = math.cos(s.angle), math.sin(s.angle)
    u = (xx * c + yy * sn) / r
    v = (-xx * sn + yy * c) / (r * s.aspect)
    dist = torch.sqrt(u * u + v * v)
    # A soft edge about a pixel wide, so tiny specks are anti-aliased rather than blocky.
    alpha = (1 - (dist - 1) * r).clamp(0, 1) * s.opacity
    return alpha, slice(y0, y1), slice(x0, x1)


def _alpha_polyline(
    h: int, w: int, pts: np.ndarray, width: float, device
) -> tuple[torch.Tensor, slice, slice] | None:
    p = torch.tensor(pts, device=device, dtype=torch.float32) * torch.tensor(
        [w, h], device=device, dtype=torch.float32
    )
    half = max(0.6, width * h / 2)
    lo, hi = p.min(0).values - half - 2, p.max(0).values + half + 2
    x0, x1 = max(0, int(lo[0])), min(w, int(hi[0]) + 1)
    y0, y1 = max(0, int(lo[1])), min(h, int(hi[1]) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = torch.meshgrid(
        torch.arange(y0, y1, device=device) + 0.5,
        torch.arange(x0, x1, device=device) + 0.5,
        indexing="ij",
    )
    q = torch.stack([xx, yy], -1).unsqueeze(-2)  # (h, w, 1, 2)
    a, b = p[:-1], p[1:]
    ab = b - a
    t = (((q - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-6)).clamp(0, 1)
    dist = (q - (a + t.unsqueeze(-1) * ab)).norm(dim=-1).min(-1).values
    return (half + 0.5 - dist).clamp(0, 1), slice(y0, y1), slice(x0, x1)


def _sprite_alpha(
    sprite: tuple, scale: float, device, cache: dict, max_extent: float | None = None
) -> torch.Tensor:
    """A real artefact's alpha, oriented and resampled to the frame (cached per run).

    ``max_extent`` caps its longest side in frame pixels (the scans include a few huge clumps).
    """
    key = (*sprite, round(scale, 4), max_extent)
    if key not in cache:
        kind, index, turn, size = sprite
        a = np.rot90(load_sprites()[kind][index], turn % 4)
        if turn >= 4:
            a = a[:, ::-1]
        t = torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=torch.float32) / 255
        k = scale * size
        if max_extent is not None:
            k = min(k, max_extent / max(t.shape))
        h, w = max(1, round(t.shape[0] * k)), max(1, round(t.shape[1] * k))
        # Antialiased: dust smaller than a pixel becomes a faint pixel, as it would on a scan.
        t = torch.nn.functional.interpolate(
            t[None, None], size=(h, w), mode="bilinear", antialias=True, align_corners=False
        )[0, 0]
        cache[key] = t.clamp(0, 1)
    return cache[key]


def _place(
    img: torch.Tensor, alpha: torch.Tensor, cx: float, cy: float, value: float, opacity: float
) -> None:
    """Blend ``alpha`` (centred at cx, cy in pixels) towards ``value``, clipped to the frame."""
    h, w = img.shape[:2]
    ah, aw = alpha.shape
    x0, y0 = round(cx - aw / 2), round(cy - ah / 2)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(aw, w - x0), min(ah, h - y0)
    if sx0 >= sx1 or sy0 >= sy1:
        return
    _lerp_to(
        img,
        alpha[sy0:sy1, sx0:sx1] * opacity,
        value,
        slice(y0 + sy0, y0 + sy1),
        slice(x0 + sx0, x0 + sx1),
    )


def _lerp_to(img: torch.Tensor, alpha: torch.Tensor, value: float, ys: slice, xs: slice) -> None:
    region = img[ys, xs]
    img[ys, xs] = region + (value - region) * alpha.unsqueeze(-1)


def render_frame(
    img: torch.Tensor, d: Damage, index: int, total: int, fps: float, schedules: dict
) -> torch.Tensor:
    """Damage on one frame (H, W, 3), in place on a copy; returns it."""
    h, w, _ = img.shape
    dev = img.device
    img = img.clone()

    # breath: the print's density and colour drift, smoothly, frame to frame
    if d.breath > 0:
        drift = schedules["breath"][index]
        img = img * (1 + 0.03 * d.breath * torch.tensor(drift, device=dev, dtype=img.dtype))

    # light leaks: screen-blended soft colour, fading in and out
    for leak in schedules["leaks"]:
        if leak.start <= index < leak.end:
            phase = (index - leak.start) / max(1, leak.end - leak.start - 1)
            env = math.sin(math.pi * phase) ** 1.5 * leak.peak * d.light_leaks
            yy = torch.linspace(0, 1, h, device=dev).view(h, 1)
            xx = torch.linspace(0, 1, w, device=dev).view(1, w) * (w / h)
            r2 = (xx - leak.x * w / h) ** 2 + (yy - leak.y) ** 2
            glow = torch.exp(-r2 / (2 * leak.radius**2)) * env
            col = torch.tensor(leak.colour, device=dev, dtype=img.dtype)
            img = 1 - (1 - img) * (1 - glow.unsqueeze(-1) * col * 0.8)

    # scratches: thin vertical lines with breaks, flickering
    for sc in schedules["scratches"]:
        if sc.start <= index < sc.end:
            rng = _rng(sc.seed, index)
            x = (sc.x0 + sc.drift * (index - sc.start)) * w + rng.normal(0, 0.4)
            half = max(0.7, sc.width * h / 2)
            cols = torch.arange(w, device=dev, dtype=img.dtype) + 0.5
            across = (half + 0.5 - (cols - x).abs()).clamp(0, 1)
            # Breaks along the line: a few dropouts per frame, re-drawn every frame.
            along = torch.ones(h, device=dev, dtype=img.dtype)
            for _ in range(int(rng.integers(0, 4))):
                a = int(rng.integers(0, h))
                along[a : a + int(rng.integers(h // 20, h // 4))] = 0
            strength = float(rng.uniform(0.5, 0.9)) * min(1.0, 0.4 + d.scratches)
            alpha = along.view(h, 1) * across.view(1, w) * strength
            nz = across.nonzero()
            if nz.numel():
                xs = slice(int(nz.min()), int(nz.max()) + 1)
                _lerp_to(img, alpha[:, xs], 0.92 if sc.bright else 0.06, slice(0, h), xs)

    scale = sprite_scale(d, h)
    cache = schedules.setdefault("sprite_cache", {})

    # hair in the gate: trembling slightly while it stays
    for hair in schedules["hairs"]:
        if hair.start <= index < hair.end:
            jitter = _rng(d.seed, 6, index).normal(0, 0.0008, 2)
            if hair.sprite:
                cx, cy = hair.points[0] + jitter
                _place(
                    img,
                    _sprite_alpha(hair.sprite, scale, dev, cache),
                    cx * w,
                    cy * h,
                    0.9 if hair.bright else 0.04,
                    hair.opacity,
                )
                continue
            got = _alpha_polyline(h, w, hair.points + jitter, hair.width, dev)
            if got is not None:
                alpha, ys, xs = got
                _lerp_to(img, alpha * hair.opacity, 0.9 if hair.bright else 0.04, ys, xs)

    # dust
    for s in dust_schedule(d, index, w / h):
        if s.sprite:
            _place(
                img,
                _sprite_alpha(s.sprite, scale * d.dust_size, dev, cache, max_extent=0.025 * h),
                s.x * w,
                s.y * h,
                0.04 if s.black else 0.92,
                s.opacity,
            )
            continue
        got = _alpha_ellipse(h, w, replace(s, radius=s.radius * d.dust_size * 2), dev)
        if got is not None:
            alpha, ys, xs = got
            _lerp_to(img, alpha, 0.04 if s.black else 0.95, ys, xs)

    # cue mark: a soft-edged dot, top right
    if d.cue_marks and index in schedules["cue"]:
        r = 0.028 * h
        cy, cx = 0.12 * h, w - 0.12 * h * (w / h) ** 0.5
        yy = torch.arange(h, device=dev).view(h, 1) + 0.5
        xx = torch.arange(w, device=dev).view(1, w) + 0.5
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ring = (r + 0.5 - dist).clamp(0, 1) * 0.85
        img = img + (0.95 - img) * ring.unsqueeze(-1)

    # gate: rounded corners of the camera aperture
    if d.gate > 0:
        img = img * schedules["gate"].unsqueeze(-1)

    bar = letterbox_bar(d, h, w)
    if bar:
        img[:bar] = 0
        img[h - bar :] = 0
    return img.clamp(0, 1)


def letterbox_bar(d: Damage, h: int, w: int) -> int:
    """Height in pixels of each black bar for the letterbox, 0 if none (or the frame is already wider)."""
    ratio = LETTERBOX.get(d.letterbox)
    if not ratio or w / h >= ratio:
        return 0
    return max(0, round((h - w / ratio) / 2))


def _gate_mask(h: int, w: int, amount: float, device, bar: int = 0) -> torch.Tensor:
    """1 inside the aperture, soft rounded corners whose radius grows with the amount.

    With a letterbox the aperture is the picture between the bars -- a widescreen gate -- so its
    corners round off at the bars, not at the frame's top and bottom where the bars would hide them.
    """
    if bar:
        inner = _gate_mask(h - 2 * bar, w, amount, device)
        mask = torch.zeros(h, w, device=device)
        mask[bar : h - bar] = inner
        return mask
    r = 0.12 * h * amount
    soft = max(1.0, 0.01 * h)
    yy = torch.arange(h, device=device, dtype=torch.float32).view(h, 1) + 0.5
    xx = torch.arange(w, device=device, dtype=torch.float32).view(1, w) + 0.5
    dx = (torch.minimum(xx, w - xx) - r).clamp(max=0)
    dy = (torch.minimum(yy, h - yy) - r).clamp(max=0)
    outside = torch.sqrt(dx * dx + dy * dy) - r
    return (-outside / soft).clamp(0, 1) if r > 0 else torch.ones(h, w, device=device)


def _breath(total: int, seed: int) -> np.ndarray:
    """(total, 3) slow AR(1) drift per channel, mostly shared (density) plus a little per channel (colour)."""
    rng = _rng(seed, 7)
    z = rng.standard_normal((total, 4))
    out = np.zeros((total, 4))
    for i in range(total):
        out[i] = 0.85 * out[i - 1] + math.sqrt(1 - 0.85**2) * z[i] if i else z[0]
    return out[:, :1] * 0.8 + out[:, 1:] * 0.35


def schedules(d: Damage, total: int, fps: float, h: int, w: int, device) -> dict:
    return {
        "hairs": hair_schedule(d, total),
        "scratches": scratch_schedule(d, total),
        "leaks": leak_schedule(d, total),
        "cue": cue_frames(total, fps) if d.cue_marks else set(),
        "breath": _breath(total, d.seed) if d.breath > 0 else None,
        "gate": _gate_mask(h, w, d.gate, device, letterbox_bar(d, h, w)) if d.gate > 0 else None,
    }


def apply(
    frames: torch.Tensor,
    d: Damage,
    *,
    fps: float = 24.0,
    frame_offset: int = 0,
    total_frames: int | None = None,
    sched: dict | None = None,
) -> torch.Tensor:
    """Damage on (N, H, W, 3) frames; returns the same shape on the same device."""
    n, h, w, _ = frames.shape
    total = total_frames or (frame_offset + n)
    sched = sched or schedules(d, total, fps, h, w, frames.device)
    out = torch.empty_like(frames[..., :3], dtype=torch.float32)
    for i in range(n):
        out[i] = render_frame(frames[i, ..., :3].float(), d, frame_offset + i, total, fps, sched)
    return out.to(frames.dtype)
