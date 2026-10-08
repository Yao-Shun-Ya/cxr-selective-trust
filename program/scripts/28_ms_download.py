"""Minimal ModelScope file downloader (direct HTTP, no SDK).

Why not use ``modelscope.snapshot_download``?
---------------------------------------------
Two independent problems hit in practice:

1. The bundled ``modelscope_hub`` in this environment raises
   ``AttributeError: 'HubConfig' object has no attribute '_logged_out'`` while
   loading credentials, so every SDK entry point dies before contacting the
   network.
2. Its ignore patterns did not prevent a 1.6 GB ``tf_model.h5`` from being
   fetched (files land as ``*.incomplete`` first, so a pattern anchored on the
   final name misses), and the run had to be killed mid-download.

The ModelScope REST API itself is fast and reliable here, so this script talks to
it directly: list a repo's files, filter them, download over HTTP with resume
support, and stream to disk.

Usage
-----
::

    python scripts/28_ms_download.py --repo AI-ModelScope/clip-vit-large-patch14-336 \
        --out models/clip-vit-large-patch14-336 --exclude '.*\\.(h5|msgpack|ot)$'

    python scripts/28_ms_download.py --repo <id> --out <dir> --list
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
RAW = "https://modelscope.cn/models/{repo}/resolve/master/{path}"


def list_files(repo: str, timeout: int = 30) -> list[dict]:
    url = API.format(repo=repo)
    req = urllib.request.Request(url, headers={"User-Agent": "paper-dl/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    data = payload.get("Data") or {}
    files = data.get("Files") or []
    return [f for f in files if f.get("Type") == "blob" or f.get("Size") is not None]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def download(repo: str, path: str, dest: Path, timeout: int = 60) -> tuple[int, float]:
    url = RAW.format(repo=repo, path=urllib.parse.quote(path))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "paper-dl/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)
    return dest.stat().st_size, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", action="append", default=[],
                    help="regex of paths to skip (repeatable)")
    ap.add_argument("--include", action="append", default=[],
                    help="if given, only paths matching one of these are kept")
    ap.add_argument("--list", action="store_true", help="print the file list and exit")
    args = ap.parse_args()

    files = list_files(args.repo)
    if not files:
        print("no files returned; check the repo id")
        return 1

    excl = [re.compile(p) for p in args.exclude]
    incl = [re.compile(p) for p in args.include] if args.include else None

    keep, skip = [], []
    for f in files:
        p = f["Path"]
        if incl and not any(r.search(p) for r in incl):
            skip.append((p, f.get("Size", 0), "not included"))
            continue
        if any(r.search(p) for r in excl):
            skip.append((p, f.get("Size", 0), "excluded"))
            continue
        keep.append(f)

    total = sum(f.get("Size", 0) for f in keep)
    print(f"repo  : {args.repo}")
    print(f"files : {len(files)} total, {len(keep)} to fetch, {len(skip)} skipped")
    print(f"size  : {human(total)}")
    if args.list:
        for f in keep:
            print(f"   {human(f.get('Size',0)):>10s}  {f['Path']}")
        for p, s, why in skip:
            print(f"   [skip:{why}] {p}")
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    t_all = time.time()
    for f in keep:
        p = f["Path"]
        dest = out / p
        if dest.exists() and dest.stat().st_size == f.get("Size", -1):
            print(f"  [cached] {p}")
            done += f.get("Size", 0)
            continue
        try:
            size, dt = download(args.repo, p, dest)
            done += size
            rate = size / dt / 1e6 if dt > 0 else 0
            print(f"  [{done/total*100:5.1f}%] {p}  {human(size)}  {rate:.1f} MB/s", flush=True)
        except Exception as exc:
            print(f"  FAILED {p}: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            return 1
    print(f"\ndone in {time.time()-t_all:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
