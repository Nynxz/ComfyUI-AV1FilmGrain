"""The pack's nodes: encode AV1 with bitstream film grain, and preview the decoder's grain.

Both nodes share the grain inputs. The grain is never added to the pixels: the encoder signals
film grain parameters in the AV1 bitstream and the decoder synthesises the grain on playback.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from dataclasses import replace

import comfy.model_management
import comfy.utils
import folder_paths
import numpy as np
import torch
from comfy.cli_args import args
from comfy_api.latest import InputImpl, io, ui
from PIL import Image

from . import av1enc, crt, filmdamage, filmlook, filmsound, fxchain, grain, vhs
from .pack import PackNode

log = logging.getLogger(__name__)

#: models/film_grain holds reusable film grain tables (aomenc / SvtAv1EncApp format).
TABLE_FOLDER = "film_grain"
TABLE_EXTENSIONS = {".tbl", ".txt"}
_NO_TABLES = "(no tables in models/film_grain)"

if TABLE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[TABLE_FOLDER] = (
        [os.path.join(folder_paths.models_dir, TABLE_FOLDER)],
        TABLE_EXTENSIONS,
    )


def _chroma_input():
    return io.Boolean.Input(
        "chroma_grain",
        default=False,
        tooltip="Grain on the colour planes too (AV1 chroma_scaling_from_luma). Off: monochrome grain.",
    )


def _intensity_input():
    return io.Float.Input(
        "intensity",
        default=1.0,
        min=0.0,
        max=4.0,
        step=0.05,
        tooltip="Multiplies the table's scaling function: 2.0 is grain twice as strong. Capped by AV1's range.",
    )


def _table_files() -> list[str]:
    return folder_paths.get_filename_list(TABLE_FOLDER) or [_NO_TABLES]


#: An AV1 film grain table passed between nodes (e.g. a VHS tape's noise).
AV1Grain = io.Custom("AV1_GRAIN")
#: A chain of effect settings, applied by the encoder a few frames at a time (see fxchain.py).
FilmFX = io.Custom("FILM_FX")

_STREAM_TIP = (
    "Leave 'images' unconnected and chain 'fx' into AV1 Film Grain Encode's 'effects' to apply this "
    "while encoding, a few frames at a time: the clip is then held in memory once, not once per node."
)


def _fx_input():
    return FilmFX.Input("fx", optional=True, tooltip="Effects before this one in a streamed chain.")


def _fx_output():
    return FilmFX.Output(
        display_name="fx", tooltip="This effect appended to the chain. " + _STREAM_TIP
    )


def _chain(fx) -> fxchain.Chain:
    return fx if isinstance(fx, fxchain.Chain) else fxchain.Chain()


def grain_mode_input():
    return io.DynamicCombo.Input(
        "grain_mode",
        options=[
            io.DynamicCombo.Option(
                "preset",
                [
                    io.Combo.Input(
                        "preset",
                        options=list(grain.PRESETS),
                        default="Fine 35mm",
                        tooltip="\n".join(
                            f"{k}: {v.description}" for k, v in grain.PRESETS.items()
                        ),
                    ),
                    _intensity_input(),
                    _chroma_input(),
                ],
            ),
            io.DynamicCombo.Option(
                "photon noise",
                [
                    io.Int.Input(
                        "iso",
                        default=1600,
                        min=50,
                        max=102400,
                        step=50,
                        tooltip="Camera ISO. Noise grows with ISO and with resolution (smaller pixels).",
                    ),
                    io.Combo.Input(
                        "sensor",
                        options=list(grain.SENSORS),
                        default=grain.DEFAULT_SENSOR,
                        tooltip="Sensor size. Smaller sensors have smaller pixels, so more noise at the same ISO. "
                        "Full frame is libaom's / rav1e's assumption, and is faint at video resolutions.",
                    ),
                    _chroma_input(),
                ],
            ),
            io.DynamicCombo.Option(
                "table file",
                [
                    io.Combo.Input(
                        "table",
                        options=_table_files(),
                        tooltip="A film grain table (aomenc / SvtAv1EncApp 'filmgrn1' format) in models/film_grain. "
                        "Validated against the AV1 specification before encoding.",
                    ),
                    _intensity_input(),
                ],
            ),
            io.DynamicCombo.Option(
                "automatic estimation",
                [
                    io.Int.Input(
                        "estimation_level",
                        default=12,
                        min=1,
                        max=50,
                        tooltip="Denoising strength the encoder uses to MEASURE the grain already in the source "
                        "(SVT-AV1 --film-grain / libaom denoise-noise-level). It does not add grain: clean "
                        "footage gets little or none.",
                    ),
                    io.Boolean.Input(
                        "denoise",
                        default=False,
                        tooltip="Code the denoised frames (the grain is then only the synthesised one). Off: code "
                        "the original frames and signal the estimated grain as well.",
                    ),
                    io.Boolean.Input(
                        "adaptive_block_size",
                        default=True,
                        tooltip="SVT-AV1 only: adapt the grain estimation block size to the resolution.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                "from VHS node",
                [
                    AV1Grain.Input(
                        "grain",
                        tooltip="The 'grain' output of VHS / NTSC: the tape's noise, synthesised by the decoder.",
                    ),
                ],
            ),
            io.DynamicCombo.Option("disabled", []),
        ],
        tooltip="preset / photon noise / table file: grain synthesised from explicit parameters, for clean "
        "footage. automatic estimation: the encoder measures grain the source already has. disabled: no grain.",
    )


def encoder_input():
    caps = av1enc.capabilities()
    crf = io.Int.Input(
        "crf", default=28, min=0, max=63, tooltip="AV1 quality: lower is better and bigger."
    )
    options = [
        io.DynamicCombo.Option(
            "svt-av1",
            [
                crf,
                io.Int.Input(
                    "preset",
                    default=4,
                    min=-1,
                    max=13,
                    tooltip="SVT-AV1 preset: lower is slower and better. Above 6 SVT-AV1 warns that film grain "
                    "costs a lot of compute.",
                ),
                io.Boolean.Input(
                    "shadow_detail",
                    default=False,
                    tooltip="SVT-AV1 variance boost: more bits for flat, dark areas, against blocky shadows. "
                    "Measured on a low-key clip: +0.8 dB in the shadows for ~40% more file size. Try a lower "
                    "crf first.",
                ),
            ],
        ),
    ]
    if caps.pyav_aom:
        options.append(
            io.DynamicCombo.Option(
                "libaom-av1",
                [
                    io.Int.Input(
                        "crf",
                        default=28,
                        min=0,
                        max=63,
                        tooltip="AV1 quality: lower is better and bigger.",
                    ),
                    io.Int.Input(
                        "cpu_used",
                        default=4,
                        min=0,
                        max=8,
                        tooltip="libaom speed: higher is faster.",
                    ),
                    io.Int.Input(
                        "grain_seed",
                        default=7391,
                        min=1,
                        max=65535,
                        tooltip="Starting seed of the grain pattern. libaom honours it; SVT-AV1 picks its own seeds.",
                    ),
                ],
            )
        )
    return io.DynamicCombo.Input(
        "encoder",
        options=options,
        tooltip="Detected: "
        + caps.summary()
        + ". svt-av1 takes tables through SvtAv1EncApp only (FFmpeg's libsvtav1 cannot).",
    )


def build_settings(
    encoder: dict, grain_mode: dict, width: int, height: int, **common
) -> av1enc.Settings:
    """Node inputs -> av1enc.Settings, generating the film grain table where the mode needs one."""
    name = encoder["encoder"]
    seed = int(encoder.get("grain_seed", 7391))
    mode = grain_mode["grain_mode"]
    if mode == "disabled":
        g = av1enc.Grain()
    elif mode == "automatic estimation":
        g = av1enc.Grain(
            "estimate",
            estimation_level=int(grain_mode["estimation_level"]),
            denoise=bool(grain_mode["denoise"]),
            adaptive=bool(grain_mode["adaptive_block_size"]),
        )
    else:
        if mode == "preset":
            table = grain.preset(
                grain_mode["preset"],
                width,
                height,
                chroma=bool(grain_mode["chroma_grain"]),
                seed=seed,
            )
            table = grain.scale_strength(table, float(grain_mode["intensity"]))
        elif mode == "photon noise":
            table = grain.photon_noise(
                int(grain_mode["iso"]), width, height,
                sensor=grain_mode["sensor"], chroma=bool(grain_mode["chroma_grain"]), seed=seed,
            )  # fmt: skip
        elif mode == "table file":
            table = load_table_file(grain_mode["table"])
            table = grain.scale_strength(table, float(grain_mode["intensity"]))
        elif mode == "from VHS node":
            table = grain_mode.get("grain")
            if table is None:
                raise ValueError(
                    "the VHS node gave no grain: its noise is off or baked into the pixels. Set its noise to "
                    "'AV1 grain', or choose another grain mode here."
                )
        else:
            raise ValueError(f"unknown grain mode {mode!r}")
        if not grain.has_grain(table):
            raise ValueError(
                "this grain configuration has no grain (all scaling values are 0); raise the intensity"
            )
        g = av1enc.Grain("table", table=grain.dumps(table))
    return av1enc.Settings(
        encoder=name,
        crf=int(encoder["crf"]),
        svt_options=(("enable-variance-boost", "1"),)
        if name == "svt-av1" and encoder.get("shadow_detail")
        else (),
        preset=int(encoder.get("preset", 4)),
        cpu_used=int(encoder.get("cpu_used", 4)),
        grain=g,
        **common,
    )


def load_table_file(name: str) -> grain.Table:
    if name == _NO_TABLES:
        raise ValueError("put a film grain table (.tbl) in models/film_grain, then refresh")
    # get_full_path only resolves names inside the registered folders, so no path escapes them.
    path = folder_paths.get_full_path(TABLE_FOLDER, name)
    if path is None:
        raise ValueError(f"film grain table {name!r} not found in models/film_grain")
    if os.path.getsize(path) > 4 * 1024 * 1024:
        raise ValueError(f"{name} is over 4 MB, too large to be a film grain table")
    with open(path, encoding="utf-8", errors="strict") as fh:
        text = fh.read()
    try:
        return grain.loads(text)
    except grain.TableError as exc:
        raise ValueError(f"{name} is not a valid AV1 film grain table: {exc}") from exc


FRAME_RATES = {
    "same as source": None,
    "24 (film)": 24.0,
    "23.976": 24000 / 1001,
    "25": 25.0,
    "30": 30.0,
}


def _frame_rate_inputs():
    return [
        io.Combo.Input(
            "frame_rate", options=list(FRAME_RATES), default="same as source",
            tooltip="Output frame rate. Film runs at 24 fps; footage at 48 or 60 fps looks like video.",
        ),
        io.Float.Input(
            "shutter_angle", default=180.0, min=0.0, max=360.0, step=1.0,
            tooltip="When changing the frame rate: each frame averages the source frames during its exposure, "
            "(angle / 360) / fps seconds. 180 is film's standard (1/48 s at 24 fps); 0 just picks frames.",
        ),
    ]  # fmt: skip


def _stream(images, fps, frame_rate, shutter_angle, effects, *, start=0, count=None, progress=None):
    """(frame generator of float32 numpy, output fps, output frame count) for an encode.

    Frames come from the one source batch, retimed if asked, through the effects chain in small
    chunks -- the clip is never copied whole.
    """
    target = FRAME_RATES.get(frame_rate)
    plan = None
    out_fps = float(fps)
    if target is not None and abs(target - fps) > 1e-3:
        plan = fxchain.retime_plan(images.shape[0], float(fps), target, float(shutter_angle))
        out_fps = target
    total = len(plan) if plan is not None else images.shape[0]
    stop = total if count is None else min(total, start + count)
    chain = _chain(effects)
    device = comfy.model_management.get_torch_device()
    runner = None
    if chain.steps:
        runner = fxchain.Runner(
            chain, total, out_fps, int(images.shape[1]), int(images.shape[2]), device
        )

    def gen():
        for frame in fxchain.chunks(
            fxchain.frames(images, plan, start, stop), runner, 8, device, start
        ):
            comfy.model_management.throw_exception_if_processing_interrupted()
            yield frame.numpy()
            if progress is not None:
                progress.update(1)

    return gen(), out_fps, stop - start


def _report_text(report: av1enc.Report, caps: av1enc.Capabilities) -> str:
    data = report.as_dict()
    data["capabilities"] = caps.summary()
    return json.dumps(data, indent=1)


class AV1FilmGrainEncode(PackNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="AV1FilmGrainEncode",
            display_name="AV1 Film Grain Encode",
            description=(
                "Encodes frames to AV1 with film grain carried in the bitstream: the encoder signals grain "
                "parameters, and AV1 players synthesise the grain while decoding. The frames stay clean, so "
                "the file stays about as small as a clean encode. Each encode is decoded afterwards to "
                "check the grain parameters are really in the bitstream.\n\nThe VIDEO output is the encoded "
                "file; Save Video with codec 'auto' copies it without re-encoding (re-encoding drops the "
                "grain)."
            ),
            inputs=[
                io.Image.Input(
                    "images", tooltip="Clean frames. Odd sizes lose their last row/column (4:2:0)."
                ),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.001),
                io.String.Input("filename_prefix", default="video/AV1Grain"),
                encoder_input(),
                grain_mode_input(),
                io.Combo.Input(
                    "container",
                    options=["webm", "mp4", "mkv"],
                    default="webm",
                    tooltip="All three keep the grain. webm (Opus audio) and mp4 play in browsers, so in "
                    "ComfyUI's preview; mkv plays in desktop players (mpv, VLC) but browsers mostly show "
                    "it black.",
                ),
                io.Combo.Input(
                    "pixel_format",
                    options=list(av1enc.PIXEL_FORMATS),
                    default="yuv420p10le",
                    advanced=True,
                    tooltip="10-bit avoids banding in gradients.",
                ),
                io.Audio.Input("audio", optional=True),
                FilmFX.Input(
                    "effects",
                    optional=True,
                    tooltip="A chain from Film Look / Film Damage / VHS 'fx' outputs, applied while encoding. Connect "
                    "the original frames to 'images' then, not ones those nodes already processed.",
                ),
                *_frame_rate_inputs(),
                io.Combo.Input(
                    "share_copy",
                    options=["off", *av1enc.SHARE_TARGETS],
                    default="off",
                    tooltip="Also save an H.264 MP4 that Discord and other chat apps preview everywhere (desktop, "
                    "browser, phone). H.264 cannot synthesise grain, so it is drawn into that copy, which is "
                    "therefore bigger; the Discord targets keep it under the upload limit.",
                ),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
                io.String.Output(
                    display_name="report",
                    tooltip="Encoder, command, sizes, grain verification and the table used, as JSON.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images,
        fps,
        filename_prefix,
        encoder,
        grain_mode,
        container="webm",
        pixel_format="yuv420p10le",
        audio=None,
        effects=None,
        frame_rate="same as source",
        shutter_angle=180.0,
        share_copy="off",
    ) -> io.NodeOutput:
        height, width = int(images.shape[1]), int(images.shape[2])
        progress = comfy.utils.ProgressBar(1)
        frames, out_fps, total = _stream(
            images, fps, frame_rate, shutter_angle, effects, progress=progress
        )
        progress.total = total
        settings = build_settings(
            encoder,
            grain_mode,
            width,
            height,
            container=container,
            pixel_format=pixel_format,
            fps=out_fps,
        )
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height
        )
        file = f"{filename}_{counter:05}_.{container}"
        path = os.path.join(full_folder, file)
        # The workflow, as Save Video embeds it (hidden is only set when ComfyUI runs the node).
        metadata = {}
        hidden = getattr(cls, "hidden", None)
        if hidden is not None and not args.disable_metadata:
            if hidden.prompt is not None:
                metadata["prompt"] = json.dumps(hidden.prompt)
            for key, value in (hidden.extra_pnginfo or {}).items():
                metadata[key] = json.dumps(value)
        report = av1enc.encode(
            frames,
            path,
            settings,
            size=(width, height),
            audio=audio,
            temp_dir=folder_paths.get_temp_directory(),
            log=lambda m: log.info("AV1 Film Grain: %s", m),
            metadata=metadata,
        )
        log.info(
            "AV1 Film Grain: %s, %d frames, %.1f kB, grain parameters on %d/%d checked frames",
            file,
            report.frames,
            report.file_size / 1000,
            report.grain_frames,
            report.verified_frames,
        )
        saved = [ui.SavedResult(file, subfolder, io.FolderType.output)]
        data = json.loads(_report_text(report, av1enc.capabilities()))
        if share_copy != "off":
            share_file = f"{os.path.splitext(file)[0]}share.mp4"
            share = av1enc.share_copy(
                path,
                os.path.join(full_folder, share_file),
                share_copy,
                log=lambda m: log.info("AV1 Film Grain: %s", m),
            )
            log.info(
                "AV1 Film Grain: share copy %s, %.1f MB (%s)",
                share_file,
                share.file_size / 2**20,
                share.mode,
            )
            data["share_copy"] = {**share.__dict__, "target": share_copy}
            saved.append(ui.SavedResult(share_file, subfolder, io.FolderType.output))
        return io.NodeOutput(
            InputImpl.VideoFromFile(path),
            json.dumps(data, indent=1),
            ui=ui.PreviewVideo(saved),
        )


class AV1FilmGrainPreview(PackNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="AV1FilmGrainPreview",
            display_name="AV1 Film Grain Preview",
            description=(
                "Encodes a few frames with the chosen grain and decodes them with dav1d, so the preview shows "
                "the grain an AV1 player really synthesises -- not an imitation. Outputs the decoded frames "
                "with grain and the same frames decoded without it."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("start_frame", default=0, min=0, max=100000),
                io.Int.Input(
                    "frames",
                    default=4,
                    min=1,
                    max=64,
                    tooltip="How many frames to encode. Grain changes every frame; a few show that.",
                ),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.001),
                encoder_input(),
                grain_mode_input(),
                io.Combo.Input(
                    "pixel_format",
                    options=list(av1enc.PIXEL_FORMATS),
                    default="yuv420p10le",
                    advanced=True,
                ),
                FilmFX.Input(
                    "effects", optional=True, tooltip="The same effects chain the encode uses."
                ),
                *_frame_rate_inputs(),
            ],
            outputs=[
                io.Image.Output(
                    display_name="with_grain", tooltip="Decoded by dav1d with film grain synthesis."
                ),
                io.Image.Output(
                    display_name="without_grain",
                    tooltip="The same bitstream decoded without the grain.",
                ),
                io.String.Output(display_name="report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls, images, start_frame, frames, fps, encoder, grain_mode, pixel_format="yuv420p10le", effects=None,
        frame_rate="same as source", shutter_angle=180.0,
    ) -> io.NodeOutput:  # fmt: skip
        height, width = int(images.shape[1]), int(images.shape[2])
        source, out_fps, total = _stream(
            images, fps, frame_rate, shutter_angle, effects, start=start_frame, count=frames
        )
        if total <= 0:
            raise ValueError(f"start_frame {start_frame} is past the last frame")
        settings = build_settings(
            encoder,
            grain_mode,
            width,
            height,
            container="mkv",
            pixel_format=pixel_format,
            fps=out_fps,
        )
        temp = folder_paths.get_temp_directory()
        os.makedirs(temp, exist_ok=True)
        path = os.path.join(temp, f"av1grain_preview_{os.getpid()}_{id(images):x}.mkv")
        try:
            report = av1enc.encode(
                source,
                path,
                settings,
                size=(width, height),
                temp_dir=temp,
                log=lambda m: log.info("AV1 Film Grain preview: %s", m),
            )
            with_grain = torch.from_numpy(av1enc.decode_rgb(path, apply_grain=True))
            without = torch.from_numpy(av1enc.decode_rgb(path, apply_grain=False))
        finally:
            if os.path.exists(path):
                os.remove(path)
        return io.NodeOutput(
            with_grain,
            without,
            _report_text(report, av1enc.capabilities()),
            ui=ui.PreviewImage(with_grain, cls=cls),
        )


# ---------------------------------------------------------------- film look

LUT_FOLDER = "luts"
LUT_EXTENSIONS = {".cube", ".png", ".tif", ".tiff"}  # .cube, or Hald CLUT images
_NO_LUT = "none"

if LUT_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[LUT_FOLDER] = (
        [os.path.join(folder_paths.models_dir, LUT_FOLDER)],
        LUT_EXTENSIONS,
    )


def _slider(name, default, lo, hi, step, tooltip, advanced=False):
    return io.Float.Input(
        name, default=default, min=lo, max=hi, step=step, tooltip=tooltip, advanced=advanced
    )


def look_input():
    strength = _slider(
        "strength", 1.0, 0.0, 2.0, 0.05, "0 = no grade, 1 = the look as designed, 2 = exaggerated."
    )
    options = [
        io.DynamicCombo.Option(name, [strength] if name != "none" else [])
        for name in filmlook.LOOKS
    ]
    options.append(io.DynamicCombo.Option("custom", [
        _slider("contrast", 1.1, 0.5, 2.0, 0.01, "Slope of the tone curve at mid-grey."),
        _slider("highlight_rolloff", 0.6, 0.0, 1.0, 0.01, "Bends highlights smoothly into white (film's shoulder)."),
        _slider("shadow_toe", 0.4, 0.0, 1.0, 0.01, "Bends shadows smoothly into black (film's toe)."),
        _slider("fade", 0.03, 0.0, 0.3, 0.005, "Lifts black: a faded print."),
        _slider("saturation", 0.95, 0.0, 2.0, 0.01, "Colour intensity."),
        _slider("temperature", 0.1, -1.0, 1.0, 0.01, "Warm (+) / cool (-)."),
        _slider("tint", 0.0, -1.0, 1.0, 0.01, "Magenta (+) / green (-)."),
        _slider("crosstalk", 0.15, 0.0, 1.0, 0.01, "Film dyes bleeding into each other: softer, denser colour."),
        _slider("bleach_bypass", 0.0, 0.0, 1.0, 0.01, "Silver kept in the print: desaturated, contrasty."),
    ]))  # fmt: skip
    return io.DynamicCombo.Input(
        "look",
        options=options,
        tooltip="\n".join(f"{k}: {v[0]}" for k, v in filmlook.LOOKS.items())
        + "\ncustom: set the tone and colour yourself.",
    )


def _grade_from(look: dict) -> filmlook.Grade:
    name = look["look"]
    if name == "custom":
        keys = (
            "contrast",
            "highlight_rolloff",
            "shadow_toe",
            "fade",
            "saturation",
            "temperature",
            "tint",
            "crosstalk",
            "bleach_bypass",
        )
        return filmlook.Grade(**{k: float(look[k]) for k in keys})
    if name not in filmlook.LOOKS:
        raise ValueError(f"unknown look {name!r}")
    return filmlook.blend_grade(filmlook.LOOKS[name][1], float(look.get("strength", 1.0)))


@functools.lru_cache(maxsize=8)
def _read_lut(path: str, mtime: float) -> filmlook.Lut:
    """Parsed once per file version: a level-12 Hald image is 3 million pixels."""
    if path.lower().endswith(".cube"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            return filmlook.parse_cube(fh.read())
    with Image.open(path) as im:
        if im.size[0] * im.size[1] > 4096 * 4096:
            raise filmlook.LutError(
                f"{im.size[0]}x{im.size[1]} is larger than any Hald CLUT (level 16 is 4096x4096)"
            )
        pixels = np.asarray(im.convert("RGB") if im.mode not in ("RGB", "I;16") else im)
    return filmlook.parse_hald(pixels)


def load_lut(name: str) -> filmlook.Lut | None:
    if not name or name == _NO_LUT:
        return None
    path = folder_paths.get_full_path(LUT_FOLDER, name)
    if path is None:
        raise ValueError(f"LUT {name!r} not found in models/luts")
    if os.path.getsize(path) > 64 * 1024 * 1024:
        raise ValueError(f"{name} is over 64 MB, too large to be a LUT")
    try:
        return _read_lut(path, os.path.getmtime(path))
    except filmlook.LutError as exc:
        raise ValueError(f"{name} is not a valid LUT: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"{name} could not be read as an image: {exc}") from exc


def _split(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Left half before, right half after, with a thin divider: (1, H, W, 3)."""
    w = before.shape[1]
    out = before.clone()
    out[:, w // 2 :] = after[:, w // 2 :]
    line = max(1, w // 400)
    out[:, w // 2 - line : w // 2 + line] = 1.0
    return out.unsqueeze(0)


class FilmLook(PackNode):
    CHUNK = 8

    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="FilmLook",
            display_name="Film Look",
            description=(
                "Tone, colour and optics of film for clean frames: a print tone curve (soft toe and "
                "shoulder), a colour look or your own .cube LUT, halation, bloom, softness, vignette, "
                "chromatic aberration, gate weave and flicker. All of it is smooth, so it costs the video "
                "encoder almost nothing -- put it before AV1 Film Grain Encode and leave the grain to the "
                "AV1 bitstream. 'comparison' shows one frame half before / half after.\n\n"
                + _STREAM_TIP
            ),
            inputs=[
                io.Image.Input("images", optional=True),
                look_input(),
                io.Combo.Input(
                    "lut",
                    options=[_NO_LUT, *folder_paths.get_filename_list(LUT_FOLDER)],
                    tooltip="A LUT from models/luts -- a .cube file or a Hald CLUT image (.png / .tif, e.g. the "
                    "RawTherapee film simulations) -- applied after the look. Expects display / Rec.709 values: "
                    "a creative LUT, not a log-to-Rec.709 conversion LUT.",
                ),
                _slider("lut_strength", 1.0, 0.0, 1.0, 0.01, "Blend of the LUT."),
                _slider(
                    "lut_shadow_protect",
                    0.5,
                    0.0,
                    1.0,
                    0.01,
                    "Fades the LUT out in the darkest tones. Film-simulation LUTs are made for normally exposed "
                    "photos and often crush a dark scene's shadows; this keeps their colour without that.",
                ),
                _slider(
                    "halation",
                    0.35,
                    0.0,
                    1.0,
                    0.01,
                    "Red-orange glow around bright areas, as light scatters back through film.",
                ),
                _slider(
                    "bloom",
                    0.15,
                    0.0,
                    1.0,
                    0.01,
                    "Soft glow of the highlights (lens, diffusion filter).",
                ),
                _slider("softness", 0.3, 0.0, 1.0, 0.01, "Takes the digital edge off fine detail."),
                _slider("vignette", 0.25, 0.0, 1.0, 0.01, "Darker corners."),
                _slider(
                    "clean_shadows",
                    0.0,
                    0.0,
                    1.0,
                    0.01,
                    "Deband the shadows: smooth the blotches a compressed source leaves in flat dark areas into "
                    "gradients (edges and texture are kept). Pair it with some grain, which hides what is left.",
                ),
                _slider(
                    "exposure", 0.0, -3.0, 3.0, 0.05, "Stops, before the tone curve.", advanced=True
                ),
                _slider(
                    "chromatic_aberration",
                    0.0,
                    0.0,
                    1.0,
                    0.01,
                    "Colour fringes towards the edges.",
                    advanced=True,
                ),
                _slider(
                    "gate_weave",
                    0.0,
                    0.0,
                    1.0,
                    0.01,
                    "The frame drifting slightly in the projector gate. Keep it low.",
                    advanced=True,
                ),
                _slider(
                    "flicker",
                    0.0,
                    0.0,
                    1.0,
                    0.01,
                    "Frame-to-frame exposure flutter.",
                    advanced=True,
                ),
                _slider(
                    "halation_size",
                    0.012,
                    0.002,
                    0.05,
                    0.001,
                    "Halation radius, as a fraction of the frame height.",
                    advanced=True,
                ),
                _slider(
                    "bloom_size",
                    0.04,
                    0.005,
                    0.15,
                    0.005,
                    "Bloom radius, as a fraction of the frame height.",
                    advanced=True,
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    advanced=True,
                    tooltip="For gate weave and flicker.",
                ),
                io.Int.Input(
                    "compare_frame",
                    default=0,
                    min=0,
                    max=100000,
                    tooltip="The frame shown in 'comparison'.",
                ),
                _fx_input(),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Image.Output(
                    display_name="comparison",
                    tooltip="One frame: left half original, right half with the look.",
                ),
                _fx_output(),
            ],
        )

    @classmethod
    def execute(
        cls, look, lut, lut_strength, halation, bloom, softness, vignette, compare_frame=0, exposure=0.0,
        chromatic_aberration=0.0, gate_weave=0.0, flicker=0.0, halation_size=0.012, bloom_size=0.04, seed=0,
        images=None, fx=None, lut_shadow_protect=0.5, clean_shadows=0.0,
    ) -> io.NodeOutput:  # fmt: skip
        g = _grade_from(look)
        optics = filmlook.Optics(
            exposure=exposure, halation=halation, halation_size=halation_size, bloom=bloom, bloom_size=bloom_size,
            softness=softness, vignette=vignette, chromatic_aberration=chromatic_aberration, gate_weave=gate_weave,
            flicker=flicker, seed=seed, clean_shadows=clean_shadows,
        )  # fmt: skip
        table = load_lut(lut)
        chain = _chain(fx).then(
            fxchain.look_step(g, optics, table, lut_strength, lut_shadow_protect)
        )
        if images is None:
            return io.NodeOutput(None, None, chain)
        device = comfy.model_management.get_torch_device()
        n = images.shape[0]
        out = torch.empty(
            (n, *images.shape[1:3], 3),
            dtype=torch.float32,
            device=comfy.model_management.intermediate_device(),
        )
        progress = comfy.utils.ProgressBar(n)
        for start in range(0, n, cls.CHUNK):
            comfy.model_management.throw_exception_if_processing_interrupted()
            chunk = images[start : start + cls.CHUNK, ..., :3].to(device)
            result = filmlook.apply(
                chunk,
                g,
                optics,
                lut=table,
                lut_strength=lut_strength,
                lut_shadow_protect=lut_shadow_protect,
                frame_offset=start,
                total_frames=n,
            )
            out[start : start + chunk.shape[0]] = result.to(out.device, torch.float32)
            progress.update(chunk.shape[0])
        i = min(compare_frame, n - 1)
        return io.NodeOutput(
            out, _split(images[i, ..., :3].to(out.device, torch.float32), out[i]), chain
        )


class SaveFilmLookLUT(PackNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="SaveFilmLookLUT",
            display_name="Save Film Look as LUT",
            description=(
                "Writes a look's tone and colour as a 33-point .cube LUT to models/luts, for other apps or to "
                "start your own. Optics (halation, bloom, ...) are spatial, so a LUT cannot hold them."
            ),
            inputs=[look_input(), io.String.Input("filename", default="my film look")],
            outputs=[io.String.Output(display_name="path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, look, filename) -> io.NodeOutput:
        folder = folder_paths.get_folder_paths(LUT_FOLDER)[0]
        os.makedirs(folder, exist_ok=True)
        name = av1enc.safe_filename(filename).removesuffix(".cube") + ".cube"
        path = os.path.join(folder, name)
        if os.path.exists(path):
            raise ValueError(f"{name} already exists in models/luts; choose another filename")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(filmlook.grade_to_cube(_grade_from(look), title=os.path.splitext(name)[0]))
        return io.NodeOutput(path, ui=ui.PreviewText(f"saved {path}"))


# ---------------------------------------------------------------- film damage + sound


class FilmDamage(PackNode):
    CHUNK = 16

    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="FilmDamage",
            display_name="Film Damage",
            description=(
                "Dust, dirt, hairs, projector scratches, light leaks, cue marks, density breath and the camera "
                "gate's rounded corners. Dust, dirt and hairs are real artefacts cut from scans of 35 mm film "
                "(FilmDamageSimulator, Ivanova et al. 2023, MIT), sized for the chosen film gauge. Everything is "
                "drawn from a schedule for the whole clip, so it is the same however it is run. Use it after Film "
                "Look and before AV1 Film Grain Encode.\n\n" + _STREAM_TIP
            ),
            inputs=[
                io.Image.Input("images", optional=True),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.001,
                    tooltip="For the timing of cue marks. In a streamed chain the encoder's output rate is used.",
                ),
                io.Combo.Input(
                    "gauge",
                    options=list(filmdamage.GAUGES),
                    default="16mm",
                    tooltip="Dust is the same size on any film, so it looks bigger on a smaller frame: barely visible "
                    "on 35 mm at video resolutions, clear on 16 mm, big on Super 8.",
                ),
                _slider(
                    "dust",
                    0.4,
                    0.0,
                    1.0,
                    0.01,
                    "Specks and dirt, new every frame. The one effect that costs the encoder noticeably.",
                ),
                _slider("hair", 0.3, 0.0, 1.0, 0.01, "Hairs and fibres that stay a second or two."),
                _slider(
                    "scratches",
                    0.3,
                    0.0,
                    1.0,
                    0.01,
                    "Thin vertical projector scratches, drifting and breaking up.",
                ),
                _slider(
                    "light_leaks",
                    0.0,
                    0.0,
                    1.0,
                    0.01,
                    "Warm glows fading in and out at the frame edge.",
                ),
                _slider(
                    "breath",
                    0.3,
                    0.0,
                    1.0,
                    0.01,
                    "The print's density and colour drifting slightly frame to frame.",
                ),
                _slider("gate", 0.3, 0.0, 1.0, 0.01, "Rounded corners of the camera gate."),
                _slider(
                    "dust_black",
                    0.3,
                    0.0,
                    1.0,
                    0.01,
                    "Share of dark dust (on the print) vs white (on the negative).",
                    advanced=True,
                ),
                _slider(
                    "dust_size",
                    0.5,
                    0.1,
                    2.0,
                    0.05,
                    "Size of dust and dirt. 1 = the scanned specks as annotated, which skew large; 0.5 is "
                    "closer to real dust. The gauge scales it further.",
                ),
                io.Boolean.Input(
                    "cue_marks",
                    default=False,
                    tooltip="The changeover dot, top right, for 4 frames ~8 s before the end.",
                ),
                io.Combo.Input(
                    "letterbox",
                    options=list(filmdamage.LETTERBOX),
                    default="none",
                    tooltip="Black bars cropping to a wider aspect ratio.",
                ),
                io.Boolean.Input(
                    "real_artefacts",
                    default=True,
                    advanced=True,
                    tooltip="Real scanned dust, dirt and hairs. Off: simple drawn shapes.",
                ),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF),
                _fx_input(),
            ],
            outputs=[io.Image.Output(display_name="images"), _fx_output()],
        )

    @classmethod
    def execute(
        cls, fps, gauge, dust, hair, scratches, light_leaks, breath, gate, cue_marks, letterbox, seed,
        dust_black=0.3, real_artefacts=True, images=None, fx=None, dust_size=0.5,
    ) -> io.NodeOutput:  # fmt: skip
        d = filmdamage.Damage(
            dust=dust, dust_black=dust_black, hair=hair, scratches=scratches, light_leaks=light_leaks,
            cue_marks=cue_marks, breath=breath, gate=gate, letterbox=letterbox, gauge=gauge, dust_size=dust_size,
            real_artefacts=real_artefacts, seed=seed,
        )  # fmt: skip
        if real_artefacts and filmdamage.load_sprites() is None:
            log.warning("Film Damage: assets/damage_sprites.npz is missing; using drawn shapes")
        chain = _chain(fx).then(fxchain.damage_step(d))
        if images is None:
            return io.NodeOutput(None, chain)
        device = comfy.model_management.get_torch_device()
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        sched = filmdamage.schedules(d, n, fps, h, w, device)
        out = torch.empty(
            (n, h, w, 3), dtype=torch.float32, device=comfy.model_management.intermediate_device()
        )
        progress = comfy.utils.ProgressBar(n)
        for start in range(0, n, cls.CHUNK):
            comfy.model_management.throw_exception_if_processing_interrupted()
            chunk = images[start : start + cls.CHUNK, ..., :3].to(device)
            result = filmdamage.apply(
                chunk, d, fps=fps, frame_offset=start, total_frames=n, sched=sched
            )
            out[start : start + chunk.shape[0]] = result.to(out.device, torch.float32)
            progress.update(chunk.shape[0])
        return io.NodeOutput(out, chain)


class FilmSound(PackNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="FilmSound",
            display_name="Film Sound",
            description=(
                "The sound of a film print's optical soundtrack: band-limited and usually mono, with hiss, "
                "crackle and pops, and the projector's wow and flutter. Wow and flutter are a sub-millisecond "
                "delay, so the sound stays in sync with the picture; the length never changes."
            ),
            inputs=[
                io.Audio.Input("audio"),
                _slider(
                    "optical",
                    0.6,
                    0.0,
                    1.0,
                    0.01,
                    "Band-limiting: 0 = full range, 1 = about 100 Hz to 6 kHz.",
                ),
                io.Boolean.Input(
                    "mono", default=True, tooltip="Optical tracks were mono until the late 1970s."
                ),
                _slider(
                    "crackle",
                    0.3,
                    0.0,
                    1.0,
                    0.01,
                    "Clicks and pops from dirt and scratches on the track.",
                ),
                _slider("hiss", 0.25, 0.0, 1.0, 0.01, "Background noise of the track."),
                _slider("wow", 0.2, 0.0, 1.0, 0.01, "Slow pitch sway of the projector's speed."),
                _slider("flutter", 0.2, 0.0, 1.0, 0.01, "Fast warble of the projector's speed."),
                _slider(
                    "saturation",
                    0.2,
                    0.0,
                    1.0,
                    0.01,
                    "Soft saturation of the track.",
                    advanced=True,
                ),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF, advanced=True),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(
        cls, audio, optical, mono, crackle, hiss, wow, flutter, saturation=0.2, seed=0
    ) -> io.NodeOutput:
        s = filmsound.Sound(
            optical=optical,
            mono=mono,
            saturation=saturation,
            hiss=hiss,
            crackle=crackle,
            wow=wow,
            flutter=flutter,
            seed=seed,
        )
        wave = audio["waveform"]
        sr = int(audio["sample_rate"])
        items = [
            filmsound.apply(wave[b].float().cpu().numpy(), sr, s) for b in range(wave.shape[0])
        ]
        out = torch.from_numpy(np.stack(items)).to(wave.dtype)
        return io.NodeOutput({**audio, "waveform": out, "sample_rate": sr})


# ---------------------------------------------------------------- VHS / NTSC

_VHS_NOISE = ["AV1 grain (small file)", "baked into pixels (exact, larger file)", "off"]


class VHSTape(PackNode):
    CHUNK = 8

    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="VHSTape",
            display_name="VHS / NTSC",
            description=(
                "NTSC composite video and VHS tape, simulated as signals on the real NTSC raster (480 lines, "
                "4x subcarrier sampling) with the formats' own bandwidths: dot crawl and rainbows, colour-under "
                "chroma smear and delay, VCR sharpening halos, time-base jitter, head-switching, mistracking "
                "and dropouts. The tape's noise goes out as an AV1 film grain table: feed 'grain' to AV1 Film "
                "Grain Encode (grain mode 'from VHS node') and the player synthesises it, keeping the file small."
                "\n\n" + _STREAM_TIP
            ),
            inputs=[
                io.Image.Input("images", optional=True),
                io.Combo.Input(
                    "preset", options=list(vhs.PRESETS), default="VHS SP",
                    tooltip="\n".join(f"{k}: {v[0]}" for k, v in vhs.PRESETS.items()),
                ),
                _slider("wear", 0.0, 0.0, 1.0, 0.01, "Ages the tape: less sharpness, more noise, jitter and dropouts."),
                _slider("tracking", 0.0, 0.0, 1.0, 0.01, "Mistracking: rolling bands of torn, noisy lines (added to the preset's)."),
                io.Combo.Input(
                    "noise", options=_VHS_NOISE, default=_VHS_NOISE[0],
                    tooltip="AV1 grain: the decoder makes the noise (small file; its streaks are a little shorter than "
                    "a real tape's). Baked: exact tape noise in the pixels, which every encoder must then store.",
                ),
                io.Boolean.Input("head_switching", default=True, tooltip="The skewed band at the bottom of a VHS picture."),
                io.Combo.Input("decoder", options=["notch", "comb"], default="notch", advanced=True,
                               tooltip="How the TV separates colour: notch (cheap set, more dot crawl) or comb (cleaner)."),
                _slider(
                    "crt_curvature",
                    0.0,
                    0.0,
                    1.0,
                    0.01,
                    "Show it on a curved CRT: the picture bulges outwards inside rounded tube corners, a little "
                    "darker towards the edges. 0 = flat (as recorded).",
                ),
                _slider("crt_corner", 0.5, 0.0, 1.0, 0.01, "How rounded the tube's corners are.", advanced=True),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF),
                _fx_input(),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                AV1Grain.Output(display_name="grain", tooltip="The tape noise as an AV1 film grain table (None if baked or off)."),
                _fx_output(),
            ],
        )  # fmt: skip

    @classmethod
    def execute(
        cls,
        preset,
        wear,
        tracking,
        noise,
        head_switching,
        seed,
        decoder="notch",
        images=None,
        fx=None,
        crt_curvature=0.0,
        crt_corner=0.5,
    ) -> io.NodeOutput:
        t = vhs.wear(vhs.PRESETS[preset][1], wear)
        t = replace(t, tracking=min(1.0, t.tracking + tracking), head_switching=head_switching and t.head_switching,
                    decoder=decoder, seed=seed)  # fmt: skip
        if noise == _VHS_NOISE[2]:
            t = replace(t, luma_noise=0.0, chroma_noise=0.0)
        bake = noise == _VHS_NOISE[1]
        chain = _chain(fx).then(fxchain.vhs_step(t, bake, crt_curvature, crt_corner))
        table = None if bake else vhs.noise_table(t, 0, 0)
        if images is None:
            return io.NodeOutput(None, table, chain)
        device = comfy.model_management.get_torch_device()
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        out = torch.empty(
            (n, h, w, 3), dtype=torch.float32, device=comfy.model_management.intermediate_device()
        )
        progress = comfy.utils.ProgressBar(n)
        for start in range(0, n, cls.CHUNK):
            comfy.model_management.throw_exception_if_processing_interrupted()
            chunk = images[start : start + cls.CHUNK, ..., :3].to(device)
            out[start : start + chunk.shape[0]] = vhs.apply(
                chunk,
                t,
                frame_offset=start,
                bake_noise=bake,
                crt_curvature=crt_curvature,
                crt_corner=crt_corner,
            ).to(out.device, torch.float32)
            progress.update(chunk.shape[0])
        return io.NodeOutput(out, table, chain)


_FROM_PRESET = "from preset"


class CRTDisplay(PackNode):
    CHUNK = 8

    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="CRTDisplay",
            display_name="CRT Display",
            description=(
                "The picture as seen on a CRT: each line drawn by an electron beam that widens with "
                "brightness (scanlines in the shadows, blooming highlights), a phosphor mask (aperture "
                "grille, slot mask or shadow mask), red/blue convergence error, glow and glass halation, "
                "phosphor persistence and a curved tube. Put it after VHS / NTSC to watch the tape on a TV."
                "\n\nThe mask and scanlines are fine, high-contrast patterns, which encoders find expensive "
                "and 4:2:0 colour halves: use a low crf, and judge the mask at 1:1, not in a scaled preview."
                "\n\n" + _STREAM_TIP
            ),
            inputs=[
                io.Image.Input("images", optional=True),
                io.Combo.Input(
                    "preset", options=list(crt.PRESETS), default="consumer TV",
                    tooltip="\n".join(f"{k}: {v[0]}" for k, v in crt.PRESETS.items()),
                ),
                _slider("mask", 1.0, 0.0, 2.0, 0.05, "Phosphor mask strength, times the preset's. 0 = no mask."),
                _slider("scanlines", 1.0, 0.0, 1.5, 0.05, "How visible the beam lines are, times the preset's."),
                _slider("glow", 1.0, 0.0, 4.0, 0.05, "Phosphor glow and glass halation, times the preset's."),
                _slider("curvature", 0.3, 0.0, 1.0, 0.01, "Curved face plate and rounded tube corners. 0 = flat."),
                _slider(
                    "persistence", 0.0, 0.0, 0.95, 0.01,
                    "Phosphor afterglow: bright moving things leave a short fading trail. 0 = none.",
                ),
                _slider(
                    "hum_bar", 0.0, 0.0, 1.0, 0.01,
                    "A soft dark band rolling slowly down the screen, as when a camera films a TV. 0 = none.",
                ),
                io.Combo.Input("mask_type", options=[_FROM_PRESET, *crt.MASKS], default=_FROM_PRESET, advanced=True,
                               tooltip="aperture grille: vertical stripes (Trinitron). slot mask: stripes broken "
                               "into staggered slots (most TVs). shadow mask: dots (monitors)."),
                io.Int.Input("lines", default=0, min=0, max=2160, advanced=True,
                             tooltip="Scanlines in the picture height; 0 = the preset's (240 or 480). Rounded so "
                             "each line is a whole number of pixels."),
                _slider("mask_size", 1.0, 0.25, 4.0, 0.05,
                        "Phosphor triad size, times the preset's. Triads are at least 3 pixels wide.", advanced=True),
                _slider("convergence", 1.0, 0.0, 5.0, 0.05,
                        "Red/blue misalignment, times the preset's.", advanced=True),
                _slider("corner", 0.5, 0.0, 1.0, 0.01, "How rounded the tube's corners are.", advanced=True),
                _fx_input(),
            ],
            outputs=[io.Image.Output(display_name="images"), _fx_output()],
        )  # fmt: skip

    @classmethod
    def execute(
        cls,
        preset,
        mask,
        scanlines,
        glow,
        curvature,
        persistence,
        hum_bar=0.0,
        mask_type=_FROM_PRESET,
        lines=0,
        mask_size=1.0,
        convergence=1.0,
        corner=0.5,
        images=None,
        fx=None,
    ) -> io.NodeOutput:
        base = crt.PRESETS[preset][1]
        d = replace(
            base,
            mask=base.mask if mask_type == _FROM_PRESET else mask_type,
            mask_strength=min(1.0, base.mask_strength * mask),
            triads=base.triads / mask_size,
            lines=lines or base.lines,
            scanlines=min(1.0, base.scanlines * scanlines),
            glow=base.glow * glow,
            halation=base.halation * glow,
            convergence=base.convergence * convergence,
            persistence=persistence,
            hum_bar=hum_bar,
            curvature=curvature,
            corner=corner,
        )
        chain = _chain(fx).then(fxchain.crt_step(d))
        if images is None:
            return io.NodeOutput(None, chain)
        device = comfy.model_management.get_torch_device()
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        out = torch.empty(
            (n, h, w, 3), dtype=torch.float32, device=comfy.model_management.intermediate_device()
        )
        progress = comfy.utils.ProgressBar(n)
        prev = None
        for start in range(0, n, cls.CHUNK):
            comfy.model_management.throw_exception_if_processing_interrupted()
            chunk = images[start : start + cls.CHUNK, ..., :3].to(device)
            frames, prev = crt.apply(chunk, d, prev, frame_offset=start)
            out[start : start + chunk.shape[0]] = frames.to(out.device, torch.float32)
            progress.update(chunk.shape[0])
        return io.NodeOutput(out, chain)
