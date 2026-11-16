"""Image operators shared by the CXR and cross-modal arms.

Why this module exists: the two arms must apply *the same* degradation, not two
implementations that happen to agree. ``08_shift_benchmark.py`` previously owned
``apply_shift`` and the cross-modal runner imported that file by path, which
worked but broke if the script was renamed. Moving the operators here gives one
definition and one import; ``tests/test_cxr_trust.py`` pins the pixel output of
every cell against digests recorded before the move, so the refactor is provably
behaviour-preserving rather than merely plausible.

The blank generators live here too. They are not degradations, but they are the
same kind of object -- a deterministic PIL transform used as an experimental
control -- and the P2 null is only interpretable if the blank is constructed
exactly as specified (docs/58 section 13.4).
"""

from __future__ import annotations

import io
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

DEFAULT_SEVERITIES = {
    "blur": {"mild": 0.5, "moderate": 1.0, "severe": 2.0},
    "noise": {"mild": 0.02, "moderate": 0.05, "severe": 0.10},
    "jpeg": {"mild": 70, "moderate": 40, "severe": 20},
    "brightness": {"mild": 0.8, "moderate": 1.2, "severe": 1.5},
    "contrast": {"mild": 0.7, "moderate": 1.3, "severe": 1.6},
    "resolution": {"mild": 0.5, "moderate": 0.25, "severe": 0.125},
}

BLANK_KINDS = ("blank_phase", "blank_gray", "blank_shuffle")


def apply_shift(image: Image.Image, shift: str, severity: str) -> Image.Image:
    """Apply one parametric degradation. Verbatim from the CXR arm."""
    image = image.convert("RGB")
    if shift == "blur":
        return image.filter(ImageFilter.GaussianBlur(
            radius=float(DEFAULT_SEVERITIES["blur"][severity])))
    if shift == "noise":
        sigma = float(DEFAULT_SEVERITIES["noise"][severity])
        arr = np.asarray(image, dtype=np.float32) / 255.0
        rng = np.random.default_rng(0)
        arr = np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if shift == "jpeg":
        quality = int(DEFAULT_SEVERITIES["jpeg"][severity])
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if shift == "brightness":
        return ImageEnhance.Brightness(image).enhance(
            float(DEFAULT_SEVERITIES["brightness"][severity]))
    if shift == "contrast":
        return ImageEnhance.Contrast(image).enhance(
            float(DEFAULT_SEVERITIES["contrast"][severity]))
    if shift == "resolution":
        factor = float(DEFAULT_SEVERITIES["resolution"][severity])
        w, h = image.size
        small = image.resize((max(1, int(w * factor)), max(1, int(h * factor))),
                             Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    raise ValueError(f"unknown shift: {shift}")


def phase_scramble(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomise phase while preserving the power spectrum *exactly*.

    The naive construction -- draw a uniform phase per pixel, multiply the
    magnitude, take the real part -- is not Hermitian-symmetric, so the real part
    has a different spectrum than the input. That makes "spectrum-matched
    control" an approximate claim, which is an easy thing for a reviewer to
    attack.

    This construction takes the phase from the FFT of a *real* Gaussian field.
    A real field has a Hermitian spectrum by construction, so
    ``|FFT(input)| * exp(i * angle(FFT(noise)))`` is Hermitian and its inverse is
    real to machine precision, while ``|FFT(out)| == |FFT(input)|`` holds
    elementwise, not on average.
    """
    noise = rng.standard_normal(gray.shape)
    phase = np.angle(np.fft.fft2(noise))
    magnitude = np.abs(np.fft.fft2(gray))
    out = np.fft.ifft2(magnitude * np.exp(1j * phase))
    return np.ascontiguousarray(out.real)


def make_blank(image: Image.Image, kind: str, seed: int) -> Image.Image:
    """One per-image blank control.

    A single global blank is not enough for the P2 null: the comparison is
    real-image versus *its own* blank, so the blank has to share the geometry and,
    for the primary variant, the spatial-frequency content. MRI and OCT carry
    large black backgrounds, which makes a black canvas indistinguishable from a
    valid slice; the flat control is therefore mid-gray.
    """
    if kind == "blank_gray":
        return Image.new("RGB", image.size, (128, 128, 128))

    gray = np.asarray(image.convert("L"), dtype=np.float64)
    rng = np.random.default_rng(seed)

    if kind == "blank_shuffle":
        flat = gray.ravel().copy()
        rng.shuffle(flat)
        arr = flat.reshape(gray.shape)
    elif kind == "blank_phase":
        arr = phase_scramble(gray, rng)
    else:
        raise ValueError(f"unknown blank kind: {kind}")

    arr = arr - arr.min()
    span = float(arr.max())
    arr = (arr / span * 255.0) if span > 0 else np.zeros_like(arr)
    rgb = np.repeat(arr.astype(np.uint8)[:, :, None], 3, axis=2)
    return Image.fromarray(rgb)


def all_shift_cells(shifts: Iterable[str] | None = None,
                    severities: Iterable[str] | None = None) -> list[tuple[str, str, float]]:
    """The (condition, severity, parameter) triples of the grid."""
    shifts = list(shifts or DEFAULT_SEVERITIES)
    severities = list(severities or ["mild", "moderate", "severe"])
    return [(s, v, DEFAULT_SEVERITIES[s][v]) for s in shifts for v in severities]
