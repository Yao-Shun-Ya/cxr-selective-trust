"""Contract tests for the shared image operators.

These exist because ``apply_shift`` moved out of ``08_shift_benchmark.py`` into
``src/cxr_trust/inference/shifts.py`` so that the CXR and cross-modal arms share
one implementation. A move is only safe if it is provably behaviour-preserving:
the digests below were recorded from the pre-move implementation on a fixed
synthetic image, so any pixel-level drift in a future edit fails here instead of
silently changing a published result.

Run from ``program/``::

    .venv-modern\\Scripts\\python.exe -m pytest -q tests/test_shifts.py
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from src.cxr_trust.inference.shifts import (
    BLANK_KINDS,
    DEFAULT_SEVERITIES,
    apply_shift,
    make_blank,
    phase_scramble,
)

# Recorded 2026-11-14 from scripts/08_shift_benchmark.py before the operators were
# moved. Do not regenerate these to make a failing test pass.
FROZEN_DIGESTS = {
    "clean": "1e42e1e15e889a63",
    "blur|mild": "6db2af90aafce6e9",
    "blur|moderate": "842b63e2535fe13c",
    "blur|severe": "5019527132e47549",
    "noise|mild": "e2b0c8c229f7c48d",
    "noise|moderate": "21968c25d98aabfc",
    "noise|severe": "1360d61735a1746c",
    "jpeg|mild": "73814def3c03c70e",
    "jpeg|moderate": "2e9c943deddef273",
    "jpeg|severe": "9688062d109f7869",
    "brightness|mild": "2c04aa4a86c28ad4",
    "brightness|moderate": "27092cfdf849de60",
    "brightness|severe": "70585d08c9cafd0d",
    "contrast|mild": "dcafdaf9c9c688c0",
    "contrast|moderate": "655e2e4d230a034b",
    "contrast|severe": "eed0b0e9e6a9ee91",
    "resolution|mild": "5ff1b68749f26cc6",
    "resolution|moderate": "7748f4c975843520",
    "resolution|severe": "e41553ebd1a98b62",
}


def _fixture() -> Image.Image:
    arr = (np.add.outer(np.arange(64), np.arange(64)) % 256).astype(np.uint8)
    base = np.stack([arr, np.flipud(arr), np.roll(arr, 7, axis=1)], axis=2)
    return Image.fromarray(base)


def _digest(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()[:16]


def test_clean_fixture_digest():
    assert _digest(_fixture()) == FROZEN_DIGESTS["clean"]


def test_every_shift_cell_matches_frozen_pixels():
    img = _fixture()
    for shift, severities in DEFAULT_SEVERITIES.items():
        for severity in severities:
            key = f"{shift}|{severity}"
            assert _digest(apply_shift(img, shift, severity)) == FROZEN_DIGESTS[key], key


def test_phase_scramble_preserves_power_spectrum_exactly():
    """The magnitude spectrum must be preserved elementwise, not approximately.

    This is the reason the construction draws its phase from the FFT of a real
    Gaussian field: a uniform per-pixel phase is not Hermitian-symmetric, so its
    real part has a different spectrum and the "matched" control is only matched
    on average.
    """
    rng = np.random.default_rng(3)
    gray = rng.random((64, 64)) * 255.0
    out = phase_scramble(gray, np.random.default_rng(11))

    before = np.abs(np.fft.fft2(gray))
    after = np.abs(np.fft.fft2(out))
    assert np.allclose(before, after, rtol=1e-8, atol=1e-8)
    assert not np.allclose(gray, out)


def test_blank_kinds_are_deterministic_and_distinct():
    img = _fixture()
    for kind in BLANK_KINDS:
        first = make_blank(img, kind, seed=1234)
        again = make_blank(img, kind, seed=1234)
        other = make_blank(img, kind, seed=4321)
        assert first.size == img.size
        assert first.tobytes() == again.tobytes(), kind
        if kind != "blank_gray":
            # The flat control is seed-invariant by design; only the two
            # structured controls must depend on the per-image seed.
            assert first.tobytes() != other.tobytes(), kind


def test_gray_blank_is_not_black():
    """MRI and OCT have large black backgrounds; a black canvas would be a valid
    slice, not a blank, and would bias Delta_null downward."""
    img = _fixture()
    blank = np.asarray(make_blank(img, "blank_gray", seed=0))
    assert np.all(blank == 128)
