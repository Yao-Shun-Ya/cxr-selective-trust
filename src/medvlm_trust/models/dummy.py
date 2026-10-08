"""Deterministic dummy VLM used to test the pipeline without real models."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .base import BaseVLM

from ..inference.schema import PRIMARY_LABELS


class DummyVLM(BaseVLM):
    def __init__(self, name: str = "dummy", seed: int = 0, fail_rate: float = 0.0):
        self.name = name
        self.seed = seed
        self.fail_rate = fail_rate

    def generate(self, image_path: str | Path, prompt: str) -> str:
        key = f"{self.seed}:{Path(image_path).name}:{prompt[:50]}"
        rnd = random.Random(int(hashlib.sha1(key.encode()).hexdigest()[:8], 16))
        if rnd.random() < self.fail_rate:
            return "I cannot determine the findings from this image."

        findings = {}
        probability_positive = {}
        confidence = {}
        for label in PRIMARY_LABELS:
            prob = rnd.random()
            status = "positive" if prob >= 0.5 else "negative"
            findings[label] = status
            probability_positive[label] = round(prob, 4)
            confidence[label] = round(max(prob, 1 - prob), 4)
        obj = {
            "findings": findings,
            "probability_positive": probability_positive,
            "confidence": confidence,
            "rationale": "dummy output for pipeline testing",
            "image_reliance_claim": "dummy visual evidence claim",
        }
        return json.dumps(obj)
