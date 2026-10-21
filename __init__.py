"""ComfyUI-AV1FilmGrain — film and tape looks for clean frames, encoded to AV1 with decoder-synthesised grain.

pack.json      the pack's identity: namespace, display name, menu category.
pack.py        that identity and the node base class.
filmlook.py    tone curve, colour looks, .cube / Hald LUTs, halation, bloom and other optics.
filmdamage.py  dust, dirt and hair (real scanned artefacts), scratches, leaks, cue marks, gate.
filmsound.py   optical soundtrack: band-limit, hiss, crackle, wow and flutter.
vhs.py         NTSC composite and VHS tape, simulated as signals; its noise as an AV1 grain table.
fxchain.py     effects run inside the encoder a few frames at a time, and film frame-rate conversion.
grain.py       film grain tables: format, validation, photon noise and stylised generators.
av1enc.py      encoder capabilities, the SVT-AV1 / libaom routes, muxing and verification.
assets/        the scanned artefact atlas (FilmDamageSimulator, MIT) and its provenance.
nodes.py       the nodes.
"""

from comfy_api.latest import ComfyExtension, io

from .nodes import (
    AV1FilmGrainEncode,
    AV1FilmGrainPreview,
    FilmDamage,
    FilmLook,
    FilmSound,
    SaveFilmLookLUT,
    VHSTape,
)


class AV1FilmGrainExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FilmLook,
            FilmDamage,
            VHSTape,
            FilmSound,
            AV1FilmGrainEncode,
            AV1FilmGrainPreview,
            SaveFilmLookLUT,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return AV1FilmGrainExtension()


__all__ = ["AV1FilmGrainExtension", "comfy_entrypoint"]
