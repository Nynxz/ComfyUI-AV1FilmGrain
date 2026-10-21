"""Effects applied while encoding, a few frames at a time, and film frame rates. No ComfyUI imports.

ComfyUI passes video between nodes as whole float32 IMAGE batches: 25 MB per 1680x1248 frame, so a
383-frame clip is 9.6 GB, and every node in a chain keeps another copy. A Chain is instead a list of
effect *settings*: the encoder pulls frames from the one source batch, runs the chain on a small chunk,
encodes it, and moves on -- peak memory is one copy of the clip plus a chunk.

Every effect is indexed by frame number and seeded, so a chain gives exactly the frames the batch
nodes give.

Frame rate: ``retime`` resamples to a film rate with a shutter. A film camera exposes each frame for
(shutter angle / 360) / fps seconds -- 1/48 s at 24 fps and 180 degrees -- so each output frame
averages the source frames inside that window: real motion blur from a high-frame-rate source.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from . import filmdamage, filmlook, vhs


@dataclass(frozen=True)
class Step:
    kind: str  # "look", "damage" or "vhs"
    settings: tuple


@dataclass(frozen=True)
class Chain:
    steps: tuple[Step, ...] = ()

    def then(self, step: Step) -> Chain:
        return Chain((*self.steps, step))

    def describe(self) -> list[str]:
        return [s.kind for s in self.steps]


def look_step(
    grade: filmlook.Grade,
    optics: filmlook.Optics,
    lut=None,
    lut_strength: float = 1.0,
    lut_shadow_protect: float = 0.0,
) -> Step:
    return Step("look", (grade, optics, lut, lut_strength, lut_shadow_protect))


def damage_step(damage: filmdamage.Damage) -> Step:
    return Step("damage", (damage,))


def vhs_step(
    tape: vhs.Tape, bake_noise: bool, crt_curvature: float = 0.0, crt_corner: float = 0.5
) -> Step:
    return Step("vhs", (tape, bake_noise, crt_curvature, crt_corner))


@dataclass
class Runner:
    """A chain prepared for one clip: total frames, fps and size fixed, per-step state built once."""

    chain: Chain
    total: int
    fps: float
    height: int
    width: int
    device: torch.device
    state: dict = field(default_factory=dict)

    def __post_init__(self):
        for i, step in enumerate(self.chain.steps):
            if step.kind == "damage":
                self.state[i] = filmdamage.schedules(
                    step.settings[0], self.total, self.fps, self.height, self.width, self.device
                )

    def apply(self, frames: torch.Tensor, frame_offset: int) -> torch.Tensor:
        """Run the chain on (N, H, W, 3) frames that start at ``frame_offset`` in the clip."""
        x = frames
        for i, step in enumerate(self.chain.steps):
            if step.kind == "look":
                grade, optics, lut, strength, protect = step.settings
                x = filmlook.apply(
                    x,
                    grade,
                    optics,
                    lut=lut,
                    lut_strength=strength,
                    lut_shadow_protect=protect,
                    frame_offset=frame_offset,
                    total_frames=self.total,
                )
            elif step.kind == "damage":
                x = filmdamage.apply(
                    x,
                    step.settings[0],
                    fps=self.fps,
                    frame_offset=frame_offset,
                    total_frames=self.total,
                    sched=self.state[i],
                )
            elif step.kind == "vhs":
                tape, bake, curvature, corner = step.settings
                x = vhs.apply(
                    x,
                    tape,
                    frame_offset=frame_offset,
                    bake_noise=bake,
                    crt_curvature=curvature,
                    crt_corner=corner,
                )
            else:
                raise ValueError(f"unknown effect {step.kind!r}")
        return x


# ---------------------------------------------------------------- frame rate


def retime_plan(
    n_source: int, source_fps: float, target_fps: float, shutter_angle: float
) -> list[tuple[int, ...]]:
    """For each output frame, the source frames its exposure averages.

    Output frame k opens at k / target_fps for (angle / 360) / target_fps seconds; it averages every
    source frame whose time falls in that window (at least the nearest one). 48 -> 24 fps at 180
    degrees averages pairs; 0 degrees just picks frames.
    """
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("frame rates must be positive")
    duration = n_source / source_fps
    count = max(1, math.floor(duration * target_fps + 1e-6))
    window = max(0.0, min(360.0, shutter_angle)) / 360.0 / target_fps
    plan = []
    for k in range(count):
        t0 = k / target_fps
        first = min(n_source - 1, round(t0 * source_fps))
        last = min(n_source - 1, math.floor((t0 + window) * source_fps + 1e-6))
        plan.append(tuple(range(first, max(first, last) + 1)))
    return plan


def frames(
    images: torch.Tensor,
    plan: list[tuple[int, ...]] | None,
    start: int = 0,
    stop: int | None = None,
) -> Iterator[torch.Tensor]:
    """Output frames ``start``..``stop`` (H, W, 3) float32 from the source batch, retimed by ``plan``.

    Averaging happens in linear light, as light accumulating on film does.
    """
    n = len(plan) if plan is not None else images.shape[0]
    for k in range(start, n if stop is None else min(stop, n)):
        if plan is None or len(plan[k]) == 1:
            yield images[plan[k][0] if plan else k, ..., :3].float()
            continue
        lin = torch.stack(
            [filmlook.srgb_to_linear(images[i, ..., :3].float()) for i in plan[k]]
        ).mean(0)
        yield filmlook.linear_to_srgb(lin).clamp(0, 1)


def chunks(
    source: Iterator[torch.Tensor], runner: Runner | None, size: int, device, start: int = 0
) -> Iterator[torch.Tensor]:
    """Batch ``source`` into chunks, run the chain on ``device``, and yield frames back on the CPU one by one."""
    buf, index = [], start
    for f in source:
        buf.append(f)
        if len(buf) == size:
            yield from _run(buf, runner, device, index)
            index += len(buf)
            buf = []
    if buf:
        yield from _run(buf, runner, device, index)


def _run(buf, runner, device, index):
    x = torch.stack(buf)
    if runner is not None:
        x = runner.apply(x.to(device), index).float().cpu()
    yield from x
