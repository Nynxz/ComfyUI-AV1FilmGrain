"""AV1 encoding with bitstream film grain synthesis, through PyAV and (optionally) SvtAv1EncApp.

No ComfyUI imports. What each route can do was checked against the actual encoders (SVT-AV1 4.2,
libaom 3.15, FFmpeg 9 via PyAV 19), not assumed:

    route                     estimation          film grain table
    PyAV libsvtav1            film-grain=N        no -- FFmpeg's svtav1-params rejects fgs-table
                                                  ("Error parsing option") and encodes WITHOUT grain
    SvtAv1EncApp (binary)     --film-grain N      --fgs-table (the app reads the table itself)
    PyAV libaom-av1           denoise-noise-level film-grain-table via aom-params

Neither SVT-AV1 nor libaom has a "photon noise" option: photon noise is a *table* (see grain.py),
so it goes through whichever route takes a table.

Every encode is verified afterwards by decoding it with dav1d and reading the AV1 film grain
parameters it exports per frame, so a route that silently drops the grain fails loudly instead.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from fractions import Fraction

import av
import numpy as np

SVT_APP_ENV = "SVTAV1ENCAPP"
CONTAINERS = {"mkv": "matroska", "mp4": "mp4", "webm": "webm"}
PIXEL_FORMATS = ("yuv420p10le", "yuv420p")
FILM_GRAIN_SIDE_DATA = "FILM_GRAIN_PARAMS"


class EncodeError(RuntimeError):
    """An encode that could not run or did not produce what was asked for."""


# ---------------------------------------------------------------- capabilities


def _has_codec(name: str, mode: str) -> bool:
    try:
        av.codec.Codec(name, mode)
    except Exception:  # noqa: BLE001 -- PyAV raises several types for "not built in"
        return False
    return True


@dataclass(frozen=True)
class Capabilities:
    pyav_svt: bool
    pyav_aom: bool
    dav1d: bool
    svt_app: str | None
    svt_app_version: str | None
    svt_app_fgs_table: bool

    def encoders(self) -> list[str]:
        out = []
        if self.pyav_svt or self.svt_app:
            out.append("svt-av1")
        if self.pyav_aom:
            out.append("libaom-av1")
        return out

    def summary(self) -> str:
        parts = [
            f"PyAV libsvtav1: {'yes' if self.pyav_svt else 'no'}",
            f"PyAV libaom-av1: {'yes' if self.pyav_aom else 'no'}",
            f"libdav1d decoder: {'yes' if self.dav1d else 'no'}",
            "SvtAv1EncApp: "
            + (
                f"{self.svt_app} ({self.svt_app_version or 'unknown version'}"
                + (", --fgs-table" if self.svt_app_fgs_table else ", no --fgs-table")
                + ")"
                if self.svt_app
                else f"not found (put it on PATH or set {SVT_APP_ENV})"
            ),
        ]
        return "; ".join(parts)


def find_svt_app() -> str | None:
    explicit = os.environ.get(SVT_APP_ENV)
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    return shutil.which("SvtAv1EncApp")


@functools.cache
def capabilities() -> Capabilities:
    app = find_svt_app()
    version, fgs = None, False
    if app:
        try:
            out = subprocess.run([app, "--version"], capture_output=True, text=True, timeout=20)
            version = (
                (out.stdout or out.stderr).strip().splitlines()[0]
                if (out.stdout or out.stderr)
                else None
            )
            helptext = subprocess.run([app, "--help"], capture_output=True, text=True, timeout=20)
            fgs = "--fgs-table" in (helptext.stdout + helptext.stderr)
        except (OSError, subprocess.SubprocessError):
            app = None
    return Capabilities(
        pyav_svt=_has_codec("libsvtav1", "w"),
        pyav_aom=_has_codec("libaom-av1", "w"),
        dav1d=_has_codec("libdav1d", "r"),
        svt_app=app,
        svt_app_version=version,
        svt_app_fgs_table=fgs,
    )


# ---------------------------------------------------------------- settings


@dataclass(frozen=True)
class Grain:
    """What grain to put in the bitstream. ``mode``: disabled, estimate, or table."""

    mode: str = "disabled"
    #: estimate: the encoder's denoising level used to measure the source's own grain (1..50).
    estimation_level: int = 0
    #: estimate, SVT-AV1 only: code the denoised frames (True) or the original frames (False).
    denoise: bool = False
    #: estimate, SVT-AV1 only: adapt the grain block size to the resolution.
    adaptive: bool = True
    #: table: libaom film grain table text (grain.dumps).
    table: str | None = None


@dataclass(frozen=True)
class Settings:
    encoder: str = "svt-av1"  # or libaom-av1
    crf: int = 28
    #: SVT-AV1 --preset (-1..13; the app rejects -2). Only used by svt-av1.
    preset: int = 4
    #: libaom --cpu-used (0..8). Only used by libaom-av1.
    cpu_used: int = 4
    container: str = "mkv"
    pixel_format: str = "yuv420p10le"
    fps: float = 24.0
    grain: Grain = field(default_factory=Grain)
    #: Extra SVT-AV1 options (name without dashes -> value), e.g. {"enable-variance-boost": "1"}.
    svt_options: tuple[tuple[str, str], ...] = ()


@dataclass
class Report:
    route: str
    command: list[str] | None
    width: int
    height: int
    frames: int
    fps: float
    file_size: int
    bitrate_kbps: float
    grain_mode: str
    #: Frames (of those decoded for verification) that carry AV1 film grain parameters.
    grain_frames: int
    verified_frames: int
    grain_seeds: list[int]
    warnings: list[str]
    table: str | None = None
    audio: bool = False
    cropped_to_even: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def fps_fraction(fps: float) -> Fraction:
    """Frame rate as a fraction, keeping NTSC rates exact (29.97 -> 30000/1001)."""
    for base in (24, 30, 60, 120):
        if abs(fps - base * 1000 / 1001) < 0.005:
            return Fraction(base * 1000, 1001)
    return Fraction(fps).limit_denominator(1001)


def validate(settings: Settings, caps: Capabilities | None = None) -> None:
    """Refuse a combination no installed encoder can honour -- before encoding anything."""
    caps = caps or capabilities()
    s, g = settings, settings.grain
    if s.container not in CONTAINERS:
        raise EncodeError(f"container must be one of {sorted(CONTAINERS)}")
    if s.pixel_format not in PIXEL_FORMATS:
        raise EncodeError(f"pixel_format must be one of {PIXEL_FORMATS}")
    if not 0 <= s.crf <= 63:
        raise EncodeError("crf must be 0..63")
    if s.fps <= 0:
        raise EncodeError("fps must be positive")
    if g.mode not in ("disabled", "estimate", "table"):
        raise EncodeError(f"unknown grain mode {g.mode!r}")
    if g.mode == "estimate" and not 1 <= g.estimation_level <= 50:
        raise EncodeError("estimation_level must be 1..50")
    if g.mode == "table" and not g.table:
        raise EncodeError("table mode needs a film grain table")
    if not caps.dav1d:
        raise EncodeError("PyAV has no libdav1d decoder, so the grain cannot be verified")
    if s.encoder == "svt-av1":
        if not -1 <= s.preset <= 13:
            raise EncodeError("SVT-AV1 preset must be -1..13")
        if g.mode == "table":
            if not caps.svt_app:
                raise EncodeError(
                    "SVT-AV1 can only take a film grain table through SvtAv1EncApp: FFmpeg's libsvtav1 "
                    "wrapper rejects fgs-table and would encode without grain. Install SvtAv1EncApp "
                    f"(on PATH, or set {SVT_APP_ENV} to its path), or choose the libaom-av1 encoder, "
                    "which takes tables in-process."
                )
            if not caps.svt_app_fgs_table:
                raise EncodeError(f"{caps.svt_app} has no --fgs-table option; update SVT-AV1")
        elif not caps.pyav_svt and not caps.svt_app:
            raise EncodeError(
                "no SVT-AV1 encoder: PyAV lacks libsvtav1 and SvtAv1EncApp was not found"
            )
    elif s.encoder == "libaom-av1":
        if not caps.pyav_aom:
            raise EncodeError("PyAV has no libaom-av1 encoder")
        if not 0 <= s.cpu_used <= 8:
            raise EncodeError("libaom cpu_used must be 0..8")
    else:
        raise EncodeError(f"unknown encoder {s.encoder!r}")


def _route(settings: Settings, caps: Capabilities) -> str:
    if settings.encoder == "libaom-av1":
        return "pyav-libaom"
    if settings.grain.mode == "table" or not caps.pyav_svt:
        return "svt-app"
    return "pyav-svt"


# ---------------------------------------------------------------- frames


def even_size(width: int, height: int) -> tuple[int, int]:
    """4:2:0 needs even dimensions: an odd last row or column is dropped."""
    return width - width % 2, height - height % 2


def yuv_frames(rgb_frames: Iterable[np.ndarray], pixel_format: str) -> Iterator[av.VideoFrame]:
    """RGB (H, W, 3) float 0..1 frames -> BT.709 limited-range YUV, one frame at a time."""
    ten_bit = pixel_format == "yuv420p10le"
    for rgb in rgb_frames:
        h, w = even_size(rgb.shape[0], rgb.shape[1])
        rgb = rgb[:h, :w, :3]
        if ten_bit:
            # 16-bit RGB keeps float precision through the conversion to 10-bit YUV (no banding).
            frame = av.VideoFrame.from_ndarray(
                np.clip(rgb * 65535.0 + 0.5, 0, 65535).astype(np.uint16), format="rgb48le"
            )
        else:
            frame = av.VideoFrame.from_ndarray(
                np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8), format="rgb24"
            )
        yield frame.reformat(format=pixel_format, dst_colorspace=1, dst_color_range=1)


def _tag_srgb(target) -> None:
    """Colour tags for ComfyUI's sRGB frames: BT.709 primaries and matrix, limited range, sRGB transfer.

    The pixels are sRGB-encoded, so the transfer is tagged sRGB (IEC 61966-2-1, AV1 code 13), as
    ComfyUI's own Save Video does. Tagged BT.709 instead, players that honour it decode with BT.1886
    (~gamma 2.4) and show the shadows darker than they were made.
    """
    from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc

    target.color_primaries = ColorPrimaries.BT709
    target.color_trc = ColorTrc.IEC61966_2_1
    target.colorspace = 1  # BT.709 matrix
    target.color_range = ColorRange.MPEG


def _y4m_frame(frame: av.VideoFrame) -> bytes:
    """One frame's planes, tightly packed, as y4m and SvtAv1EncApp expect them."""
    out = bytearray(b"FRAME\n")
    sample = 2 if frame.format.name.endswith("10le") else 1
    for i, plane in enumerate(frame.planes):
        width = frame.width if i == 0 else frame.width // 2
        height = frame.height if i == 0 else frame.height // 2
        rows = np.frombuffer(plane, dtype=np.uint8).reshape(-1, plane.line_size)[
            :height, : width * sample
        ]
        out += rows.tobytes()
    return bytes(out)


# ---------------------------------------------------------------- encode routes


def _svtav1_params(grain: Grain) -> str:
    if grain.mode != "estimate":
        return ""
    return (
        f"film-grain={grain.estimation_level}:film-grain-denoise={int(grain.denoise)}"
        f":adaptive-film-grain={int(grain.adaptive)}"
    )


def _dict_safe(path: str) -> str | None:
    """``path`` as FFmpeg's ``key=value:key=value`` option parser reads it intact, or None.

    The parser has no escaping, so a ':' anywhere -- a Windows drive letter, or in a folder name --
    cuts the value. A path relative to the working directory has no drive letter.
    """
    candidates = [path]
    with contextlib.suppress(ValueError):  # on another drive than the working directory
        candidates.append(os.path.relpath(path))
    return next((c for c in candidates if ":" not in c), None)


@contextlib.contextmanager
def _option_safe_table(table_path: str):
    """A copy of the table at a path libaom can be given through FFmpeg, removed afterwards."""
    if safe := _dict_safe(table_path):
        yield safe
        return
    for base in (tempfile.gettempdir(), os.getcwd()):
        folder = tempfile.mkdtemp(prefix="av1grain_", dir=base)
        try:
            copy = os.path.join(folder, "grain.tbl")
            if safe := _dict_safe(copy):
                shutil.copyfile(table_path, copy)
                yield safe
                return
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    raise EncodeError(
        f"no path for the film grain table that FFmpeg's libaom options can carry (every candidate, "
        f"starting with {table_path!r}, contains ':'). Use the svt-av1 encoder with SvtAv1EncApp, "
        "which takes the table as a plain argument."
    )


def _encode_pyav(frames, out_path, settings, size, rate, table_path, log) -> tuple[list[str], int]:
    with contextlib.ExitStack() as stack:
        if settings.encoder == "libaom-av1" and table_path is not None:
            table_path = stack.enter_context(_option_safe_table(table_path))
        return _encode_pyav_stream(frames, out_path, settings, size, rate, table_path, log)


def _encode_pyav_stream(
    frames, out_path, settings, size, rate, table_path, log
) -> tuple[list[str], int]:
    s, g = settings, settings.grain
    with av.open(out_path, "w", format="ivf") as out:
        if s.encoder == "libaom-av1":
            stream = out.add_stream("libaom-av1", rate=rate)
            options = {"crf": str(s.crf), "b": "0", "cpu-used": str(s.cpu_used), "row-mt": "1"}
            aom = []
            if g.mode == "estimate":
                aom.append(f"denoise-noise-level={g.estimation_level}")
            elif g.mode == "table":
                aom.append(f"film-grain-table={table_path}")
            if aom:
                options["aom-params"] = ":".join(aom)
        else:
            stream = out.add_stream("libsvtav1", rate=rate)
            options = {"crf": str(s.crf), "preset": str(s.preset)}
            params = [p for p in (_svtav1_params(g), *(f"{k}={v}" for k, v in s.svt_options)) if p]
            if params:
                options["svtav1-params"] = ":".join(params)
        stream.width, stream.height = size
        stream.pix_fmt = s.pixel_format
        stream.options = options
        _tag_srgb(stream.codec_context)
        log(f"encoding with PyAV {stream.codec_context.name} {options}")
        count = 0
        for frame in frames:
            _tag_srgb(frame)
            out.mux(stream.encode(frame))
            count += 1
        out.mux(stream.encode(None))
    command = [stream.codec_context.name] + [f"-{k} {v}" for k, v in options.items()]
    return command, count


def _encode_svt_app(frames, out_path, settings, size, rate, table_path, log, workdir):
    s, g = settings, settings.grain
    caps = capabilities()
    ten_bit = s.pixel_format == "yuv420p10le"
    cmd = [
        caps.svt_app,
        "-i", "stdin",
        "--preset", str(s.preset),
        "--crf", str(s.crf),
        "--input-depth", "10" if ten_bit else "8",
        "--color-primaries", "1",  # BT.709
        "--transfer-characteristics", "13",  # sRGB
        "--matrix-coefficients", "1",
        "--color-range", "0",  # studio (limited)
        "--progress", "0",
        "-b", out_path,
    ]  # fmt: skip
    if g.mode == "estimate":
        cmd += [
            "--film-grain", str(g.estimation_level),
            "--film-grain-denoise", str(int(g.denoise)),
            "--adaptive-film-grain", str(int(g.adaptive)),
        ]  # fmt: skip
    elif g.mode == "table":
        cmd += ["--fgs-table", table_path]
    for key, value in s.svt_options:
        cmd += [f"--{key}", value]
    log("running " + subprocess.list2cmdline(cmd))
    w, h = size
    header = f"YUV4MPEG2 W{w} H{h} F{rate.numerator}:{rate.denominator} Ip A1:1 C420{'p10' if ten_bit else ''}\n"
    stderr_path = os.path.join(workdir, "svtav1.log")
    count = 0
    with open(stderr_path, "wb") as stderr:
        # An argument list, never a shell: paths with spaces or quotes pass through untouched.
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr
        )
        try:
            proc.stdin.write(header.encode("ascii"))
            for frame in frames:
                proc.stdin.write(_y4m_frame(frame))
                count += 1
            proc.stdin.close()
            code = proc.wait()
        except (BrokenPipeError, OSError):
            code = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
    if code != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) <= 32:
        with open(stderr_path, encoding="utf-8", errors="replace") as fh:
            errors = [
                line.strip()
                for line in fh
                if line.strip() and not line.startswith(("Svt[info]", "SvtMalloc"))
            ]
        raise EncodeError(
            f"SvtAv1EncApp failed (exit {code}): " + ("; ".join(errors[-5:]) or "no output")
        )
    return cmd, count


# ---------------------------------------------------------------- mux + verify


def _audio_codec(container: str) -> str:
    if container == "webm":
        for name in ("libopus", "opus"):
            if _has_codec(name, "w"):
                return name
        raise EncodeError("WebM needs an Opus audio encoder, and PyAV has none")
    return "aac"


def _mux(
    video_path: str,
    out_path: str,
    container: str,
    audio: dict | None,
    frames: int,
    rate: Fraction,
    metadata: dict[str, str] | None = None,
) -> bool:
    """Copy the AV1 stream (no re-encode, so the grain parameters survive) and add the audio."""
    options = {"movflags": "use_metadata_tags+faststart"} if container == "mp4" else {}
    with (
        av.open(video_path) as src,
        av.open(out_path, "w", format=CONTAINERS[container], options=options) as out,
    ):
        for key, value in (metadata or {}).items():
            out.metadata[key] = value
        vin = src.streams.video[0]
        # opaque: copy the stream as-is; without it PyAV looks for an encoder named after the decoder.
        vout = out.add_stream_from_template(template=vin, opaque=True)
        vout.time_base = vin.time_base
        aout = waveform = None
        if audio is not None:
            sample_rate = int(audio["sample_rate"])
            waveform = audio["waveform"]
            # Trim to the video, as ComfyUI's own video saving does.
            waveform = waveform[0, :, : math.ceil(sample_rate / float(rate) * frames)]
            layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(waveform.shape[0], "stereo")
            codec = _audio_codec(container)
            out_rate = 48000 if codec in ("libopus", "opus") else sample_rate
            aout = out.add_stream(codec, rate=out_rate, layout=layout)
        for packet in src.demux(vin):
            if packet.dts is None:
                continue
            packet.stream = vout
            out.mux(packet)
        if aout is not None:
            samples = (
                waveform.float().cpu().contiguous().numpy()
                if hasattr(waveform, "cpu")
                else np.ascontiguousarray(waveform, dtype=np.float32)
            )
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=aout.layout.name)
            frame.sample_rate = sample_rate
            frame.pts = 0
            resampler = None
            if aout.rate != sample_rate:
                resampler = av.audio.resampler.AudioResampler(
                    format="fltp", layout=aout.layout.name, rate=aout.rate
                )
            for f in [frame] if resampler is None else resampler.resample(frame):
                out.mux(aout.encode(f))
            if resampler is not None:
                for f in resampler.resample(None):
                    out.mux(aout.encode(f))
            out.mux(aout.encode(None))
    return aout is not None


def _side_data_seed(side) -> int | None:
    """``seed`` of an AVFilmGrainParams (offset 8, stable across FFmpeg versions), if readable."""
    try:
        raw = bytes(memoryview(side))
        if len(raw) >= 16 and int.from_bytes(raw[0:4], "little") == 1:  # AV_FILM_GRAIN_PARAMS_AV1
            return int.from_bytes(raw[8:16], "little")
    except (TypeError, ValueError):
        pass
    return None


def decode(path: str, *, apply_grain: bool = True, limit: int | None = None, export: bool = False):
    """Decode with dav1d, applying the film grain or not. Yields (frame, film grain side data or None)."""
    with av.open(path) as src:
        stream = src.streams.video[0]
        ctx = av.codec.CodecContext.create("libdav1d", "r")
        options = {}
        if export or not apply_grain:
            # Export the parameters instead of applying them: dav1d then returns the clean frame.
            options["export_side_data"] = "film_grain"
        ctx.options = options
        if stream.codec_context.extradata:
            ctx.extradata = stream.codec_context.extradata
        n = 0
        for packet in src.demux(stream):
            for frame in ctx.decode(packet):
                side = None
                for key in frame.side_data:
                    if getattr(key, "name", str(key)) == FILM_GRAIN_SIDE_DATA:
                        side = frame.side_data[key]
                yield frame, side
                n += 1
                if limit is not None and n >= limit:
                    return


def inspect(path: str, limit: int | None = None) -> tuple[int, int, list[int]]:
    """(frames decoded, frames carrying film grain parameters, their seeds) -- from the bitstream."""
    total = with_grain = 0
    seeds = []
    for _frame, side in decode(path, export=True, limit=limit):
        total += 1
        if side is not None:
            with_grain += 1
            if (seed := _side_data_seed(side)) is not None:
                seeds.append(seed)
    return total, with_grain, seeds


# ---------------------------------------------------------------- the whole thing


_TEMP_LOCK = threading.Lock()


def encode(
    rgb_frames: Iterable[np.ndarray],
    out_path: str,
    settings: Settings,
    *,
    size: tuple[int, int],
    audio: dict | None = None,
    temp_dir: str | None = None,
    log=print,
    verify_frames: int | None = 48,
    metadata: dict[str, str] | None = None,
) -> Report:
    """Encode RGB frames to ``out_path`` with the requested grain, then verify it is in the bitstream.

    ``rgb_frames`` is consumed lazily, one frame at a time. ``size`` is the frames' (width, height).
    """
    caps = capabilities()
    validate(settings, caps)
    width, height = even_size(*size)
    warnings = []
    if (width, height) != tuple(size):
        warnings.append(
            f"cropped {size[0]}x{size[1]} to {width}x{height}: 4:2:0 AV1 needs even dimensions"
        )
    rate = fps_fraction(settings.fps)
    route = _route(settings, caps)
    with _TEMP_LOCK:
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)  # ComfyUI empties its temp folder on restart
        workdir = tempfile.mkdtemp(prefix="av1grain_", dir=temp_dir)
    try:
        table_path = None
        if settings.grain.mode == "table":
            table_path = os.path.join(workdir, "grain.tbl")
            with open(table_path, "w", encoding="ascii", newline="\n") as fh:
                fh.write(settings.grain.table)
        video_path = os.path.join(workdir, "video.ivf")
        frames = yuv_frames(rgb_frames, settings.pixel_format)
        if route == "svt-app":
            command, count = _encode_svt_app(
                frames, video_path, settings, (width, height), rate, table_path, log, workdir
            )
        else:
            command, count = _encode_pyav(
                frames, video_path, settings, (width, height), rate, table_path, log
            )
        if count == 0:
            raise EncodeError("no frames to encode")
        tags = dict(metadata or {})
        tags["av1_film_grain"] = json.dumps(
            {"mode": settings.grain.mode, "encoder": settings.encoder, "route": route}
        )
        has_audio = _mux(video_path, out_path, settings.container, audio, count, rate, tags)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    verified, with_grain, seeds = inspect(out_path, limit=verify_frames)
    mode = settings.grain.mode
    if mode == "table" and with_grain == 0:
        raise EncodeError(
            f"the encoder ({route}) wrote no film grain parameters although a table was given; "
            "the output has no grain. Check the encoder's log above."
        )
    if mode == "estimate" and with_grain == 0:
        warnings.append(
            "the encoder estimated no grain in this footage, so none is signalled. Estimation only "
            "measures grain the source already has; for clean footage use a preset, photon noise or a table."
        )
    if mode == "disabled" and with_grain:
        raise EncodeError("grain parameters found although grain was disabled")
    size_bytes = os.path.getsize(out_path)
    duration = count / float(rate)
    for w in warnings:
        log("warning: " + w)
    return Report(
        route=route,
        command=command,
        width=width,
        height=height,
        frames=count,
        fps=float(rate),
        file_size=size_bytes,
        bitrate_kbps=round(size_bytes * 8 / duration / 1000, 1),
        grain_mode=mode,
        grain_frames=with_grain,
        verified_frames=verified,
        grain_seeds=sorted(set(seeds))[:8],
        warnings=warnings,
        table=settings.grain.table,
        audio=has_audio,
        cropped_to_even=(width, height) != tuple(size),
    )


def decode_rgb(path: str, *, apply_grain: bool = True, limit: int | None = None) -> np.ndarray:
    """Decode to float32 RGB (N, H, W, 3) in 0..1, with or without dav1d's film grain synthesis."""
    out = []
    for frame, _ in decode(path, apply_grain=apply_grain, limit=limit):
        rgb = frame.reformat(format="rgb48le", src_colorspace=1, src_color_range=1).to_ndarray()
        out.append(rgb.astype(np.float32) / 65535.0)
    if not out:
        raise EncodeError(f"{path} decoded to no frames")
    return np.stack(out)


def grain_difference(path: str, frames: int = 8) -> np.ndarray:
    """Luma of the decoder's grain: (decoded with grain) - (decoded without), float, 8-bit scale."""
    with_grain = [f.reformat(format="gray16le").to_ndarray() for f, _ in decode(path, limit=frames)]
    without = [
        f.reformat(format="gray16le").to_ndarray()
        for f, _ in decode(path, apply_grain=False, limit=frames)
    ]
    return (np.stack(with_grain).astype(np.float32) - np.stack(without).astype(np.float32)) / 257.0


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_filename(name: str) -> str:
    return _SAFE_NAME.sub("_", name).strip() or "grain"


# ---------------------------------------------------------------- share copy (H.264 for chat apps)

#: name -> size limit in MB (None: quality only).
SHARE_TARGETS = {"Discord (10 MB)": 10.0, "Discord Nitro (50 MB)": 50.0, "H.264 high quality": None}


@dataclass
class ShareReport:
    path: str
    file_size: int
    width: int
    height: int
    mode: str  # "crf 18" or "<kbps> kb/s"
    attempts: int


def _h264_pass(
    src: str, out: str, *, crf: int | None, kbps: int | None, size: tuple[int, int] | None
) -> None:
    """One H.264 encode of ``src`` with dav1d's film grain drawn in. Streams frame by frame."""
    with (
        av.open(src) as inp,
        av.open(out, "w", format="mp4", options={"movflags": "+faststart"}) as outp,
    ):
        vin = inp.streams.video[0]
        ain = inp.streams.audio[0] if inp.streams.audio else None
        dec = av.codec.CodecContext.create("libdav1d", "r")
        rate = vin.average_rate or vin.guessed_rate or Fraction(24)
        w, h = size or (vin.codec_context.width, vin.codec_context.height)
        vout = outp.add_stream("libx264", rate=rate)
        vout.width, vout.height, vout.pix_fmt = w, h, "yuv420p"
        vout.codec_context.time_base = Fraction(1, 1) / rate
        opts = {"preset": "slow", "tune": "grain", "profile": "high"}
        if kbps:
            opts.update({"b": f"{kbps}k", "maxrate": f"{kbps}k", "bufsize": f"{2 * kbps}k"})
        else:
            opts["crf"] = str(crf)
        vout.options = opts
        _tag_srgb(vout.codec_context)
        aout = resampler = None
        if ain is not None:
            aout = outp.add_stream("aac", rate=48000, layout="stereo")
            aout.bit_rate = 128000
            resampler = av.audio.resampler.AudioResampler(
                format="fltp", layout="stereo", rate=48000
            )
            adec = av.codec.CodecContext.create(ain.codec_context.name, "r")
            adec.extradata = ain.codec_context.extradata
        n_video = n_audio = 0
        for packet in inp.demux(*(s for s in (vin, ain) if s is not None)):
            if packet.stream.type == "video":
                for frame in dec.decode(packet):  # libdav1d applies the film grain
                    f = frame.reformat(width=w, height=h, format="yuv420p", interpolation="LANCZOS")
                    _tag_srgb(f)
                    f.pts, f.time_base = n_video, vout.codec_context.time_base
                    n_video += 1
                    outp.mux(vout.encode(f))
            else:
                for frame in adec.decode(packet):
                    frame.pts = None
                    for r in resampler.resample(frame):
                        r.pts, r.time_base = n_audio, Fraction(1, 48000)
                        n_audio += r.samples
                        outp.mux(aout.encode(r))
        outp.mux(vout.encode(None))
        if aout is not None:
            for r in resampler.resample(None):
                outp.mux(aout.encode(r))
            outp.mux(aout.encode(None))


def share_copy(src: str, out: str, target: str, log=print) -> ShareReport:
    """An H.264 MP4 of an AV1 encode that chat apps preview everywhere, with the grain baked in.

    Quality first (crf 18, x264's grain tuning). If a size limit is set and that is too big, the
    bitrate is set from the duration (minus 128 kb/s of audio), with 5 % headroom; and if that leaves
    too few bits per pixel for grain to survive, the picture is scaled to 720p first.
    """
    if target not in SHARE_TARGETS:
        raise EncodeError(f"unknown share target {target!r}")
    limit = SHARE_TARGETS[target]
    with av.open(src) as probe:
        v = probe.streams.video[0]
        w, h = v.codec_context.width, v.codec_context.height
        duration = (
            float(probe.duration / 1e6)
            if probe.duration
            else (v.frames or 1) / float(v.average_rate or 24)
        )
        fps = float(v.average_rate or 24)
    size = None
    attempts = 1
    _h264_pass(src, out, crf=18, kbps=None, size=None)
    mode = "crf 18"
    budget = None if limit is None else limit * 1024 * 1024
    while budget is not None and os.path.getsize(out) > budget * 0.98 and attempts < 4:
        kbps = int((budget * 8 / duration / 1000 - 128) * 0.95 * (0.9 ** (attempts - 1)))
        if kbps < 300:
            raise EncodeError(
                f"{duration:.0f} s cannot fit in {limit:g} MB with watchable quality; shorten the clip"
            )
        cw, ch = size or (w, h)
        if kbps * 1000 / (cw * ch * fps) < 0.05 and ch > 720:  # too few bits per pixel for grain
            scale = 720 / ch
            size = (round(cw * scale / 2) * 2, 720)
        log(
            f"share copy over {limit:g} MB at {mode}; re-encoding at {kbps} kb/s"
            + (f", {size[0]}x{size[1]}" if size else "")
        )
        _h264_pass(src, out, crf=None, kbps=kbps, size=size)
        mode = f"{kbps} kb/s"
        attempts += 1
    if budget is not None and os.path.getsize(out) > budget:
        raise EncodeError(
            f"the share copy is {os.path.getsize(out) / 2**20:.1f} MB, over {limit:g} MB; shorten the clip"
        )
    sw, sh = size or (w, h)
    return ShareReport(out, os.path.getsize(out), sw, sh, mode, attempts)
