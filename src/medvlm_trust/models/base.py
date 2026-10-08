"""Base interface for VLM wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseVLM(ABC):
    name: str

    @abstractmethod
    def generate(self, image_path: str | Path, prompt: str) -> str:
        """Return raw model output for one image + prompt."""
        raise NotImplementedError
