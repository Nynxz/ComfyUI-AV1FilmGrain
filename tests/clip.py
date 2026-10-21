"""A clean, noise-free test clip: flat colour, a smooth gradient, fine moving detail."""

from __future__ import annotations

import numpy as np


#: Regions of the clip, as (row slice, column slice) of a frame of size (H, W).
def regions(h: int, w: int) -> dict[str, tuple[slice, slice]]:
    third = w // 3
    return {
        "flat": (slice(16, h - 16), slice(16, third - 16)),
        "gradient": (slice(16, h - 16), slice(third + 16, 2 * third - 16)),
        "detail": (slice(16, h - 16), slice(2 * third + 16, w - 16)),
    }


def frame(t: int, h: int = 360, w: int = 640) -> np.ndarray:
    """Frame ``t`` as float32 RGB in 0..1."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    third = w // 3
    out = np.empty((h, w, 3), np.float32)
    out[:, :third] = (0.30, 0.42, 0.55)  # flat mid-tone
    g = (x[:, third : 2 * third] - third) / third
    out[:, third : 2 * third] = np.stack([g, g * 0.85, g * 0.7], -1)  # smooth black-to-white ramp
    d = (
        np.sin((x[:, 2 * third :] + 3 * t) / 2.5) * np.cos(y[:, 2 * third :] / 3.5) > 0
    ) * 0.7 + 0.15
    out[:, 2 * third :] = d[..., None]  # fine, moving detail
    return out


def frames(n: int = 48, h: int = 360, w: int = 640):
    for t in range(n):
        yield frame(t, h, w)
