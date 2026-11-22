"""Command-line entry point for ``cxr-audit``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .census import DEFAULT_KEY_FIELDS, census
from .demo import run_demo
from .io import DEFAULT_LABELS, read_jsonl
from .responsiveness import responsiveness
from .selective import selective_audit


def _emit(payload: dict, out: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _labels(value: str) -> list[str]:
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("--labels cannot be empty")
    return labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cxr-audit",
        description="CPU-only reproducibility audit for medical VLM prediction files.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_census = sub.add_parser("census", help="rows, unique keys, duplicates, parse rate")
    p_census.add_argument("path", help="JSONL prediction file")
    p_census.add_argument("--key-fields", default=",".join(DEFAULT_KEY_FIELDS),
                          help="comma-separated key fields (default: image_id,condition,severity)")
    p_census.add_argument("--out", default=None, help="write JSON here instead of stdout")

    p_resp = sub.add_parser("responsiveness", help="distinct/modal vectors and P1 change")
    p_resp.add_argument("path", help="JSONL prediction file")
    p_resp.add_argument("--labels", type=_labels, default=DEFAULT_LABELS,
                        help="comma-separated contract labels")
    p_resp.add_argument("--clean", default="clean")
    p_resp.add_argument("--shift", default="contrast")
    p_resp.add_argument("--severity", default="severe",
                        help="shift severity (empty string matches all severities)")
    p_resp.add_argument("--out", default=None)

    p_sel = sub.add_parser("selective", help="confidence/entropy/margin risk-coverage")
    p_sel.add_argument("path", help="CSV with p_calibrated (or p), conf (or pred), error")
    p_sel.add_argument("--out", default=None)

    sub.add_parser("demo", help="run the self-contained synthetic demo")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "census":
            payload = census(read_jsonl(args.path),
                             [field.strip() for field in args.key_fields.split(",")
                              if field.strip()])
        elif args.command == "responsiveness":
            severity = args.severity if args.severity else None
            payload = responsiveness(read_jsonl(args.path), args.labels,
                                     clean=args.clean, shift=args.shift,
                                     severity=severity)
        elif args.command == "selective":
            payload = selective_audit(pd.read_csv(args.path))
        elif args.command == "demo":
            payload = run_demo()
        else:  # pragma: no cover - argparse enforces the choices
            raise ValueError(f"unknown command {args.command!r}")
        _emit(payload, getattr(args, "out", None))
        return 0
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
