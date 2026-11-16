"""Injected first on PYTHONPATH so the stub *directory* is importable.

It deliberately does NOT install the stub here. ``sitecustomize`` runs during
``site`` initialisation, before the runner has imported anything, and by then only
one of the two registry module objects exists:

    cxr_trust.models.registry        (root on sys.path)
    src.cxr_trust.models.registry    (root/src on sys.path)

and ``src/`` has no ``__init__.py``, so those are two genuinely separate modules
with separate globals -- patching one does nothing for the other. The first build
of this harness patched the wrong one and the "stubbed" run quietly loaded a 4B
checkpoint.

So this file only makes ``stub_model`` importable; the runner calls
``stub_model.install()`` itself, after all its imports have run. Patching then
covers both module objects.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
