# ComfyUI-AV1FilmGrain

AV1 video with **film grain synthesised by the decoder**, not baked into the frames.

AV1 can carry film grain as a few dozen *parameters* in the bitstream: how strong the grain is at
each brightness, how it is textured, and a seed. Every AV1 player regenerates the grain from them
while decoding. The frames you encode stay clean, so the file stays about as small as a clean
encode, while noise baked into the pixels is the most expensive thing a video encoder can be asked
to keep.

```
clean frames ──► Film Look ──► Film Damage ──► VHS / NTSC ──► AV1 Film Grain Encode ──► .webm / .mp4 / .mkv
                 tone, colour,  dust, hair,     composite,     grain preset / photon     grain = parameters,
                 LUT, halation  scratches, gate tape, CRT      noise / table / tape      the player adds it
audio ─────────► Film Sound ──────────────────────────────────┘                          (+ H.264 share copy)
                 optical track, crackle, wow
```

Grain alone doesn't make footage look like film. Most of "film" is the tone curve, the colour, and how
light behaves in the lens and the emulsion. That part is all smooth change, which is cheap to encode,
so **Film Look** applies it to the pixels. Grain is the one expensive part, so it stays in the AV1 bitstream.

## Film Look

| stage | controls |
|---|---|
| **look** | `neutral film`, `warm print`, `teal & orange`, `faded 70s`, `bleach bypass`, `cool night`, `vivid slide` (each with a `strength`), or `custom` for contrast, highlight roll-off, shadow toe, fade, saturation, temperature, tint, dye crosstalk and bleach bypass |
| **lut** | a `.cube` LUT (1D or 3D) or a **Hald CLUT** image (`.png` / `.tif`) from `models/luts/`, after the look, with `lut_strength`. It expects display / Rec.709 values: a creative LUT, not a log conversion LUT |
| **halation** | the red-orange glow film gets around bright areas |
| **bloom** | the soft glow of highlights (lens, diffusion filter) |
| **softness** | takes the digital edge off fine detail |
| **vignette** | darker corners |
| advanced | exposure, chromatic aberration, gate weave, flicker, halation / bloom size, seed |

The tone curve is a straight contrast line that bends smoothly into black and white (film's toe and
shoulder) instead of clipping. Black and white stay put unless `fade` lifts them. Blur sizes are fractions of
the frame height, so a look holds at any resolution. `comparison` shows one frame, half before and half after.

**Save Film Look as LUT** writes a look's tone and colour as a 33-point `.cube` to `models/luts`, to use in
other apps or as a starting point. The optics are spatial, so a LUT can't hold them.

**Shadows.** `lut_shadow_protect` (default 0.5) blends the LUT back out of the darkest tones, since many
film-stock LUTs crush them. `clean_shadows` smooths flat, near-black areas before grain goes on. It's for
sources whose dark areas are already blocky (a low-bitrate H.264 input, or a video model's VAE), which a
film look's fade would otherwise lift into view.

The looks are this pack's own grades. No third-party LUTs are bundled.

**Free film stock LUTs to try:** the [RawTherapee Film Simulation Collection](https://rawtherapee.com/shared/HaldCLUT.zip)
(422 MB, CC BY-SA 4.0, by Pat David, Pavlov Dmitry and Michael Ezra) has nearly 300 Hald CLUTs that
approximate Kodak, Fuji, Agfa, Ilford, Polaroid and Lomography stocks, including pushed, pulled and faded
variants. Unzip its `Color` and `Black-and-White` folders into `models/luts/` and they appear in the `lut`
list. They are hand-made approximations, not measured film. Credit the authors if you publish with them.

For a film look rather than a grainy one, keep the grain low and let the look carry the image. For example, Film
Look `warm print` with the defaults, then AV1 grain `Fine 35mm` at intensity 0.5–0.7 with chroma off.

## Film Damage

| effect | what it is | encoder cost* |
|---|---|---|
| **dust** | specks and dirt, white (on the negative) or black (on the print), new every frame | +4% at 0.4, +17% at 1.0 |
| **hair** | hairs and fibres that stay a second or two, trembling | ~0% |
| **scratches** | thin vertical projector scratches, drifting and breaking up | +1% |
| **light leaks** | warm glows fading in and out at the frame edge | ~0% |
| **breath** | the print's density and colour drifting frame to frame | ~0% |
| **gate** | the camera gate's rounded corners | +3% |
| **cue marks** | the changeover dot, 4 frames about 8 s before the end | ~0% |
| **letterbox** | 1.66 / 1.85 / 2.00 / 2.39 bars | -4% (2.39) |

\*File size against the same clip with Film Look only, SVT-AV1 crf 28, 16mm, measured on one 768×416
clip. All the defaults together cost +9%. For comparison, grain baked into the pixels cost +1500%.

**Dust, dirt and hairs are real.** They're alpha masks cut from the hand-annotated scans of empty 35 mm frames
in [FilmDamageSimulator](https://github.com/daniela997/FilmDamageSimulator) (Daniela Ivanova et al.,
*Simulating analogue film damage to analyse and improve artefact restoration on high-resolution scans*,
Eurographics 2023, MIT). The atlas `assets/damage_sprites.npz` (1.5 MB) holds 1,500 dust, 900 dirt,
700 short-hair and 398 long-hair artefacts. Its licence and source commit are next to it. Turn
`real_artefacts` off for simple drawn shapes.

**`gauge`** sizes them physically. Dust is the same size whatever the film, so on a smaller frame it looks
bigger: barely visible on 35mm at video resolutions, clear on 16mm, big on Super 8.

Everything is drawn from a schedule for the whole clip from `seed`, so a frame is the same however the
clip is run, and hairs and scratches persist across frames as they do on film.

## VHS / NTSC

**VHS / NTSC** simulates the signal, not a filter over the picture. Frames go onto the real NTSC raster (480
lines, sampled at 4× the colour subcarrier), are encoded to composite video and decoded again, then pass
through the tape format's own bandwidths:

| preset | |
|---|---|
| broadcast NTSC | composite only: dot crawl and rainbows, no tape |
| S-VHS | sharper luma, clean colour |
| VHS SP | the usual home-recording look: chroma smear and delay, sharpening halos |
| VHS EP (worn) | long-play speed: softer, noisier, more jitter and dropouts |
| camcorder | VHS-C at SP, with an oversharpened, slightly hot picture |

`wear` and `tracking` age the tape: time-base jitter, mistracking bands and dropouts. `head_switching` is the
skewed band at the bottom of the picture. `decoder` picks how the TV separates colour (`notch`: more dot
crawl; `comb`: cleaner).

**Tape noise as AV1 grain.** With `noise` on *AV1 grain*, the noise goes out as a grain table, with separate
luma and chroma strengths. Wire `grain` into the encoder with grain mode **from VHS node** and the player
draws it. *Baked* puts the exact noise in the pixels instead, at a much larger file.

**CRT.** `crt_curvature` shows the picture on a curved tube: it bulges outwards inside rounded corners
(`crt_corner`) and darkens slightly towards the edges. The picture is zoomed so the curve never leaves black
edges.

## Long clips: the effects chain

ComfyUI passes video between nodes as whole float32 batches: 25 MB per 1680×1248 frame, and every node in a
chain keeps its own copy. On long clips that runs out of RAM.

So each effect node (Film Look, Film Damage, VHS / NTSC) also has an `fx` input and output. Leave `images`
unconnected, chain the nodes by `fx`, and connect the last one to the encoder's `effects`. The encoder then
pulls the source frames, runs the whole chain a few frames at a time, and encodes as it goes. The output
frames are the same as the batch nodes give (checked by the tests). On a 96-frame 1680×1248 clip the chain
peaked at 5.5 GB against 7.9 GB for the batch nodes, and the saving grows with clip length.

## Frame rate and share copy

**`frame_rate`** on the encoder resamples to `24 (film)`, `23.976`, `25` or `30` with a **`shutter_angle`**.
A film camera exposes each frame for (angle / 360) / fps seconds, so each output frame averages the source
frames inside that window, in linear light: real motion blur from a high-frame-rate source. 48 → 24 fps at
180° averages pairs, and 0° just picks frames.

**`share_copy`** also writes an H.264 `.mp4` next to the AV1 file, for players and apps that don't draw AV1
grain (Discord, most phones). The grain is decoded and drawn into its pixels.
- `Discord (10 MB)` / `Discord Nitro (50 MB)` fit the size limit, retrying at a lower bitrate and scaling to
  720p if needed.
- `H.264 high quality` is CRF 18 with no limit.

## Film Sound

Until the late 1970s most prints carried their sound as an optical track. **Film Sound** gives audio that
character: `optical` band-limiting (down to about 100 Hz–6 kHz), `mono`, `hiss`, `crackle` (clicks and pops),
projector `wow` (a slow ±0.4% pitch sway) and `flutter` (a fast warble), and soft `saturation`.

Wow and flutter are a delay of under a millisecond that never accumulates, and the filters are zero-phase,
so **the sound stays in sync with the picture** and its length never changes. Both are checked by the tests.

## Nodes

**AV1 Film Grain Encode** saves the video to `output/` and shows it in the node. The default container is
**webm**, because ComfyUI's preview is the browser: browsers play webm and mp4, but mostly show mkv black
(desktop players are fine with all three). Files are 10-bit 4:2:0, tagged BT.709 primaries and matrix, sRGB
transfer and limited range, the same as ComfyUI's own Save Video. Its outputs:
- `video`: the encoded file.
- `report`: JSON with the encoder, its command, the size and bitrate, the grain verification and the table used.

After every encode it decodes the file with dav1d and checks that AV1 film grain parameters really are in the
bitstream. If they are missing, the node fails instead of quietly giving you a grain-less file.

**AV1 Film Grain Preview** encodes a few frames and decodes them with dav1d. You see the grain an
AV1 player really draws, not an imitation of it. It outputs the frames `with_grain` and `without_grain`
(the same bitstream decoded with synthesis off).

> Feeding `video` into **Save Video**: keep its codec on **auto**. Auto copies the AV1 stream as it is.
> Choosing a codec re-encodes the video, which drops the grain parameters.

> **Saving clean intermediates with ComfyUI's own nodes:** leave **Create Video**'s `codec` on `none` and pick
> the codec on **Save Video** (`re-encode`, AV1 crf 10–14). A codec set on Create Video encodes right away at
> its default quality (AV1 crf 24), and Save Video's crf then only re-encodes that copy. In dark scenes that
> first encode shows as blocks.

## Grain modes

| mode | what it does | for |
|---|---|---|
| **preset** | a named look (below), with an `intensity` multiplier and optional `chroma_grain` | clean / generated footage |
| **photon noise** | the noise of a digital camera sensor at an `iso`, for a chosen `sensor` size, at the video's resolution | a photographic look |
| **table file** | a film grain table from `models/film_grain/` (validated first), with `intensity` | your own or downloaded grain profiles |
| **from VHS node** | the tape noise table from **VHS / NTSC**'s `grain` output | a tape look |
| **automatic estimation** | the encoder measures the grain the source *already has* and signals it | footage that is already grainy |
| **disabled** | no grain | |

**Automatic estimation does not add grain.** SVT-AV1's `film-grain=N` sets how hard the encoder denoises
the source to *measure* its noise. On a clean ComfyUI render there is almost nothing to measure.
In the benchmark below it gives a grain std of 0.36, which is invisible. For clean footage use a preset, photon
noise or a table.

### Presets

| preset | grain |
|---|---|
| Fine 35mm | subtle, fine, strongest in the mid-tones |
| Coarse 16mm | pronounced, clumpier (AV1's coarsest clean texture), mid-tones |
| Vintage Anime | strong, the same in every tone so flat cel colours show it; monochrome unless `chroma_grain` |
| High ISO 800 / 1600 / 3200 | photon noise of a phone-size (1/2.3-inch) sensor at that ISO and the video's resolution |

These are artistic looks built from AV1's grain parameters. They are **not** emulations of real film stocks.
The ISO presets use the photon noise model, which is physical but describes a *digital sensor*,
not film.

`chroma_grain` puts grain on the colour planes too. Without it the grain is monochrome.

### Photon noise

This is the model libaom's `photon_noise_table` tool and rav1e's `--photon-noise` use: photon shot
noise, read noise and photo-response non-uniformity, for a daylight spectrum, mapped through the
BT.709 transfer function. More pixels on the same sensor means smaller pixels and more noise.

libaom assumes a full-frame sensor, and at video resolutions a full-frame pixel collects so many photons
that even ISO 3200 is faint. That's why the node also has a `sensor` choice (Super 35 down to
phone-size). A smaller sensor is the physically correct way to get visible grain out of the same model.

### Film grain tables

Put tables in **`ComfyUI/models/film_grain/`** (`.tbl` or `.txt`). The format is the one aomenc,
SvtAv1EncApp and rav1e read, starting with `filmgrn1`. Every table is validated against the AV1
specification's ranges (point counts, increasing x, AR lag and coefficients, shifts, 4:2:0 chroma
rules) before encoding, and errors name the field. Only files in that folder can be picked.

No third-party tables are bundled. If you use some, for example
[Boulder08/av1-graintables](https://github.com/Boulder08/av1-graintables) or
[nekotrix/AV1-Photon-Noise-Tables](https://github.com/nekotrix/AV1-Photon-Noise-Tables), check their licences.

## Encoders

What each route can do was checked against the actual encoders (SVT-AV1 4.2, libaom 3.15, FFmpeg 9 /
PyAV 19), not assumed:

| route | estimation | film grain table |
|---|---|---|
| PyAV `libsvtav1` (ComfyUI's own) | ✅ `film-grain`, `film-grain-denoise`, `adaptive-film-grain` | ❌ FFmpeg's wrapper rejects `fgs-table` and **encodes without grain** |
| `SvtAv1EncApp` (standalone binary) | ✅ | ✅ `--fgs-table` |
| PyAV `libaom-av1` | ✅ `denoise-noise-level` | ✅ `film-grain-table` |

- **svt-av1** (default). Estimation runs in-process. Tables (preset / photon noise / table file) need
  **SvtAv1EncApp**, found on `PATH` or through the `SVTAV1ENCAPP` environment variable.
  - Linux: your distro's `svt-av1` package.
  - Windows: `SvtAv1EncApp.exe` from an [SVT-AV1 release](https://gitlab.com/AOMediaCodec/SVT-AV1/-/releases)
    or a community build. Put it on `PATH` or point `SVTAV1ENCAPP` at it.
  - Frames are piped to it, so nothing is written to disk.
- **libaom-av1** appears only if your PyAV was built with libaom. ComfyUI's PyAV wheel isn't; a PyAV
  built against a system FFmpeg can be. It takes tables in-process, and it honours `grain_seed` as the starting seed.
  SVT-AV1 picks its own per-frame seeds, so the svt-av1 encoder has no seed input.

`shadow_detail` (svt-av1) turns on SVT-AV1's variance boost, which spends more bits on flat, low-contrast
areas. Shadows came out about 0.8 dB better, for about 43% more file size.

Neither encoder has a "photon noise" option. Photon noise is a table this pack generates.

The encoder tooltip lists what was detected. A combination the installed encoders can't honour is
refused before encoding, with a hint on what to install or choose.

## Benchmark

Run with `tests/benchmark.py`. The clip is a clean synthetic one (flat colour, a smooth gradient, fine moving detail),
1280×720, 48 frames at 24 fps, SVT-AV1 crf 28, preset 6. Decoded grain is the std of the decoded luma over the flat
region, in 8-bit code values. These numbers are measured on one machine and will vary.

| Encoding method | File size | vs clean | Grain params in bitstream | Decoded grain (std) |
|---|---|---|---|---|
| Clean AV1 | 156.5 kB | 1.00× | 0/48 frames | 0.00 |
| AV1 automatic FGS (`film-grain=12`) | 161.2 kB | 1.03× | 48/48 | 0.36 |
| AV1 photon noise (ISO 3200, phone-size sensor) | 158.1 kB | 1.01× | 48/48 | 4.65 |
| AV1 external FGS table (Coarse 16mm) | 158.7 kB | 1.01× | 48/48 | 6.82 |
| AV1 Vintage Anime preset | 158.7 kB | 1.01× | 48/48 | 8.99 |
| AV1 baked grain (pixel noise matched to Coarse 16mm) | 2511.3 kB | **16.04×** | 0/48 | 2.66 |

The baked encode is 16 times larger and still keeps only about 40% of the grain it was given, because
the encoder smooths most of it away at this crf.

Playback was checked on these files: dav1d (CLI and via FFmpeg) gives different output with
synthesis on and off, and mpv plays them. No other players were tested. A decoder may skip film grain
synthesis and show the clean frames, so check the player you target.

## Limitations

- **AV1 grain can't make big clumps.** The texture filter reaches 3 samples and the grain is tiled
  from 32×32 blocks. Push the correlation further and the tiling shows as a mosaic (checked
  visually), so `Coarse 16mm` is as coarse as AV1 grain gets cleanly.
- Grain parameters are one set for the whole video. Tables can hold several time ranges, but the
  generators write one.
- Odd frame sizes lose their last row or column (4:2:0); the report says so.
- Strengths are calibrated against dav1d's output on flat grey, for tone `flat`: a requested std of 2 / 4 / 8
  measured 2.01 / 4.02 / 8.02 (10-bit) and 2.11 / 4.03 / 8.04 (8-bit, where rounding adds a little).

## Development

```bash
COMFY_ROOT=/path/to/ComfyUI python tests/test_grain.py   # tables, validation, generators
COMFY_ROOT=/path/to/ComfyUI python tests/test_filmlook.py # tone, looks, LUTs, optics
COMFY_ROOT=/path/to/ComfyUI python tests/test_damage_sound.py # damage schedules, gauges, sync of the sound
COMFY_ROOT=/path/to/ComfyUI python tests/test_vhs.py     # NTSC raster, tape presets, noise table, CRT
COMFY_ROOT=/path/to/ComfyUI python tests/test_stream.py  # the effects chain, retiming, share copy
COMFY_ROOT=/path/to/ComfyUI python tests/test_encode.py  # real encodes, decoded and measured; the nodes
COMFY_ROOT=/path/to/ComfyUI python tests/benchmark.py    # the table above
```

`filmlook.py`, `filmdamage.py`, `filmsound.py`, `vhs.py`, `fxchain.py`, `grain.py` and `av1enc.py` don't import ComfyUI. Photon noise follows the model of libaom's
`examples/photon_noise_table.c` (BSD 2-Clause), reimplemented here.
