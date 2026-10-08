"""Add the RUNNER_STUB test seam to the four older runners (idempotent).

Why this is a script and not four hand edits: the seam is a testability property,
and testability is a code property that regresses. Any runner that writes
predictions needs it, because without it the kill/restart suite silently loads a
real multi-GB checkpoint instead of the stub — which is exactly what happened the
first time: the "kill" runs completed all their work at full GPU cost and exited
0, so four assertions failed for a reason that had nothing to do with durability.

A seam left in a production script is a small risk, so it is fenced:
  * it only activates when RUNNER_STUB == "1" (never set outside the tests);
  * the runner then asserts the factory really returned the stub, so a
    partially-applied seam fails loudly instead of quietly running the real model.

    .venv-vl\\Scripts\\python.exe scripts\\50_add_runner_stub_seam.py --dry-run
    .venv-vl\\Scripts\\python.exe scripts\\50_add_runner_stub_seam.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEAM = '''

# Test seam: when RUNNER_STUB=1 the model factory is replaced AFTER all imports have
# run. It must be installed here and not from sitecustomize, because src/ has no
# __init__.py, so "src.cxr_trust.*" and "cxr_trust.*" are two separate module
# objects -- a patch applied before this import would miss this one and the runner
# would quietly load a real checkpoint while the test believed it was stubbed.
if os.environ.get("RUNNER_STUB") == "1":  # pragma: no cover - test-only path
    import stub_model

    stub_model.install()
'''

ASSERT = '''
    if os.environ.get("RUNNER_STUB") == "1":
        # Fail loudly if the stub did not take over: a silently un-stubbed test run
        # would load a real checkpoint and still "pass".
        assert type(model).__name__ == "_StubVLM", (
            "RUNNER_STUB=1 but get_model returned "
            + type(model).__name__ + "; the stub did not patch this import path")
'''

TARGETS = {
    "06_image_reliance.py": (
        "from src.cxr_trust.models.registry import MODEL_NAMES, get_model  # noqa: E402",
        "    model = get_model(\n"),
    "08_shift_benchmark.py": (
        "from src.cxr_trust.models.registry import MODEL_NAMES, get_model  # noqa: E402",
        "    model = get_model(\n"),
    "10_run_chexpert_rrg.py": (
        "from src.cxr_trust.models.registry import get_model  # noqa: E402",
        "    model = get_model(args.model, model_path=ROOT / spec[\"path\"]",
    ),
    "21_prompt_ablation.py": (
        "from src.cxr_trust.models.registry import MODEL_NAMES, get_model  # noqa: E402",
        "    model = get_model(\n"),
}


def ensure_os_import(text: str) -> str:
    if re.search(r"^import os$", text, re.M):
        return text
    return re.sub(r"^import json$", "import json\nimport os", text, count=1, flags=re.M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed = 0
    for name, (import_line, model_line) in TARGETS.items():
        p = ROOT / "scripts" / name
        if not p.exists():
            print(f"{name}: MISSING")
            continue
        text = p.read_text(encoding="utf-8")
        before = text
        notes = []

        if "RUNNER_STUB" not in text:
            if import_line not in text:
                print(f"{name}: ! registry import anchor not found, skipped")
                continue
            text = ensure_os_import(text)
            text = text.replace(import_line, import_line + SEAM, 1)
            notes.append("seam added")
        if "_StubVLM" not in text:
            idx = text.find(model_line)
            if idx >= 0:
                end = text.find("\n\n", idx)
                if end < 0:
                    end = len(text)
                text = text[:end] + "\n" + ASSERT + text[end:]
                notes.append("assert added")
            else:
                notes.append("! model line not found, assert skipped")

        if text != before:
            changed += 1
            if not args.dry_run:
                p.write_text(text, encoding="utf-8")
        print(f"{name}: {', '.join(notes) if notes else 'already has the seam'}")

    print(f"\n{changed} file(s) {'would be ' if args.dry_run else ''}updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
