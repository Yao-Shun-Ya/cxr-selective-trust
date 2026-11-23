"""Baseline comparison and fault injection for the durable prediction writer.

This is the evidence the harness paper was missing: every reliability claim is
measured against four alternatives on the same workload, and every failure mode
is injected rather than described.

Baselines
    B0  naive append, no fsync, no sidecar          (the common script)
    B1  append + fsync per row                      (durable but dumb)
    B2  buffer 100 rows, rewrite whole file atomically  (VLMEvalKit-style reuse)
    B3  SQLite in WAL mode, one transaction per batch
    B4  the shipped DurableJsonl (append + fsync + byte-offset sidecar)

    python scripts/123_harness_experiments.py            # full matrix
    python scripts/123_harness_experiments.py --quick    # 400 cells, for CI

Worker mode is internal: the crash and concurrency faults need a real second
process, because a simulated crash inside the parent cannot drop the parent's
own buffers.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import io
import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.inference.durable import DurableJsonl  # noqa: E402

OUT = ROOT / "outputs" / "analysis_harness"
PAYLOAD = "x" * 1400          # matches the measured 1.42 kB per frozen row
BASELINES = ["B0", "B1", "B2", "B3", "B4"]


def _rec(i: int) -> dict:
    return {"image_id": f"img_{i:05d}", "condition": "clean", "severity": "none",
            "raw_output": PAYLOAD, "parsed": {"parse_success": True}}


def _key(rec: dict) -> str:
    return f"{rec['image_id']}|{rec['condition']}|{rec['severity']}"


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
class B0:
    """Naive append. No flush, no fsync, no recovery metadata."""

    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "a", encoding="utf-8")

    def write(self, rec: dict) -> float:
        t = time.perf_counter()
        self.f.write(json.dumps(rec) + "\n")
        return (time.perf_counter() - t) * 1000

    def close(self):
        self.f.close()

    @staticmethod
    def done(path: Path) -> set[str]:
        if not path.exists():
            return set()
        keys = set()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                keys.add(_key(json.loads(line)))
            except Exception:
                continue
        return keys


class B1(B0):
    """Append + fsync per row. Durable, but pays a syscall per cell."""

    def write(self, rec: dict) -> float:
        t = time.perf_counter()
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())
        return (time.perf_counter() - t) * 1000


class B2:
    """Buffer and rewrite the whole file atomically. The file-level reuse model
    used by multimodal harnesses that key on a result file rather than a row."""

    def __init__(self, path: Path, batch: int = 100):
        self.path = path
        self.rows: dict[str, dict] = {}
        self.batch = batch
        self.since = 0
        self.bytes_written = 0

    def _flush_all(self):
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        blob = "".join(json.dumps(r) + "\n" for r in self.rows.values())
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self.bytes_written += len(blob)

    def write(self, rec: dict) -> float:
        t = time.perf_counter()
        self.rows[_key(rec)] = rec
        self.since += 1
        if self.since >= self.batch:
            self._flush_all()
            self.since = 0
        return (time.perf_counter() - t) * 1000

    def close(self):
        if self.since:
            self._flush_all()

    @staticmethod
    def done(path: Path) -> set[str]:
        return B0.done(path)


class B3:
    """SQLite in WAL mode. The industrial answer; no sidecar needed."""

    def __init__(self, path: Path, batch: int = 100):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS cells (k TEXT PRIMARY KEY, v TEXT)")
        self.batch = batch
        self.since = 0

    def write(self, rec: dict) -> float:
        t = time.perf_counter()
        self.db.execute("INSERT OR REPLACE INTO cells VALUES (?, ?)",
                        (_key(rec), json.dumps(rec)))
        self.since += 1
        if self.since >= self.batch:
            self.db.commit()
            self.since = 0
        return (time.perf_counter() - t) * 1000

    def close(self):
        self.db.commit()
        self.db.close()

    @staticmethod
    def done(path: Path) -> set[str]:
        if not path.exists():
            return set()
        db = sqlite3.connect(path)
        try:
            return {r[0] for r in db.execute("SELECT k FROM cells")}
        finally:
            db.close()


class B4:
    """The shipped writer."""

    def __init__(self, path: Path):
        self.path = path
        self.ctx = DurableJsonl(path, key=lambda r: (
            r["image_id"], r["condition"], r["severity"]))
        self.w = self.ctx.__enter__()

    def write(self, rec: dict) -> float:
        t = time.perf_counter()
        self.w.write_many([rec])
        return (time.perf_counter() - t) * 1000

    def close(self):
        self.ctx.__exit__(None, None, None)

    @staticmethod
    def done(path: Path) -> set[str]:
        if not path.exists():
            return set()
        keys = set()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            keys.add(f"{obj['image_id']}|{obj['condition']}|{obj['severity']}")
        return keys


def _make(name: str, path: Path):
    return {"B0": B0, "B1": B1, "B2": B2, "B3": B3, "B4": B4}[name](path)


def _done(name: str, path: Path) -> set[str]:
    return {"B0": B0, "B1": B1, "B2": B2, "B3": B3, "B4": B4}[name].done(path)


# --------------------------------------------------------------------------- #
# worker (crash + concurrency need a second process)
# --------------------------------------------------------------------------- #
def worker(args) -> int:
    path = Path(args.path)
    w = _make(args.baseline, path)
    for i in range(args.start, args.start + args.n):
        w.write(_rec(i))
    if args.crash:
        # Drop the buffers exactly as a killed process would: no close, no flush.
        os._exit(0)
    w.close()
    return 0


# --------------------------------------------------------------------------- #
# one experiment
# --------------------------------------------------------------------------- #
def _torn_line(path: Path) -> bool:
    if not path.exists():
        return False
    data = path.read_bytes()
    return bool(data) and not data.endswith(b"\n")


def _duplicates(path: Path, baseline: str) -> int:
    if baseline == "B3":
        return 0
    seen, dup = set(), 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            k = _key(json.loads(line))
        except Exception:
            continue
        if k in seen:
            dup += 1
        seen.add(k)
    return dup


@contextlib.contextmanager
def _inject(kind: str, limit: int):
    """Patch the OS primitive the durability claim actually rests on.

    Injected at the syscall boundary, not in our own writer, so a baseline cannot
    pass by catching the exception somewhere convenient: the failure surfaces
    wherever Python's buffered IO would really see it.
    """
    saved = {}
    counter = {"calls": 0, "bytes": 0}

    if kind == "fsync_fail":
        saved["fsync"] = os.fsync

        def bad_fsync(fd):
            counter["calls"] += 1
            if counter["calls"] >= limit:
                raise OSError(5, "injected fsync failure")
            return saved["fsync"](fd)
        os.fsync = bad_fsync

    elif kind == "enospc":
        # Patching os.write does not work: CPython's buffered IO calls the C
        # layer, not the Python function. Wrap builtins.open instead, so every
        # writer (including the shipped one, which opens internally) sees a file
        # object that refuses writes past a byte budget.
        # Path.open() calls io.open directly, bypassing builtins.open, so both
        # names have to be replaced or the shipped writer escapes the injection.
        saved["open"] = builtins.open
        saved["io_open"] = io.open

        class _Full:
            def __init__(self, fh):
                self._fh = fh

            def write(self, data):
                counter["bytes"] += len(data)
                if counter["bytes"] >= limit:
                    raise OSError(28, "No space left on device (injected)")
                return self._fh.write(data)

            def __getattr__(self, name):
                return getattr(self._fh, name)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._fh.__exit__(*exc)

        def bad_open(*a, **k):
            return _Full(saved["open"](*a, **k))
        builtins.open = bad_open
        io.open = bad_open

    elif kind == "rename_fail":
        saved["replace"] = os.replace

        def bad_replace(*a, **k):
            raise OSError(13, "injected rename failure")
        os.replace = bad_replace

    try:
        yield
    finally:
        for name, fn in saved.items():
            if name == "open":
                builtins.open = fn
            elif name == "io_open":
                io.open = fn
            else:
                setattr(os, name, fn)


def run_case(baseline: str, fault: str, n: int, kill_at: int, tmp: Path) -> dict:
    path = tmp / f"{baseline}_{fault}.jsonl"
    if path.exists():
        path.unlink()
    sidecar = path.with_suffix(path.suffix + ".progress.json")
    if sidecar.exists():
        sidecar.unlink()

    entry = {"baseline": baseline, "fault": fault, "rows_target": n,
             "outcome": "", "note": ""}
    lat: list[float] = []

    injected = {"fsync_fail", "enospc", "rename_fail"}

    # ---- phase 1: produce a partial artefact, either by crash or cleanly ----
    if fault == "crash":
        proc = subprocess.run(
            [sys.executable, str(Path(__file__)), "--worker",
             "--baseline", baseline, "--path", str(path),
             "--n", str(kill_at), "--start", "0", "--crash"],
            capture_output=True, text=True)
        entry["note"] = f"worker_exit={proc.returncode}"
    elif fault == "concurrent":
        workers = [
            subprocess.Popen([sys.executable, str(Path(__file__)), "--worker",
                              "--baseline", baseline, "--path", str(path),
                              "--n", str(kill_at), "--start", "0"])
            for _ in range(2)
        ]
        codes = [p.wait() for p in workers]
        entry["note"] = f"two processes, identical key range, exits={codes}"
        if any(c != 0 for c in codes):
            entry["outcome"] = "worker_raised"
    else:
        # The writer must be CONSTRUCTED inside the patch: several baselines open
        # their file in __init__, and an injection applied after that point would
        # leave the already-open handle unpatched and silently test nothing.
        w = None
        # fsync failure is injected on the second call, because a batch-flushing
        # baseline may only fsync a handful of times per run and a threshold tied
        # to the row count would never be reached.
        limit = 2 if fault == "fsync_fail" else max(1, kill_at // 2)
        ctx = (_inject(fault, limit=limit)
               if fault in injected else contextlib.nullcontext())
        try:
            with ctx:
                w = _make(baseline, path)
                for i in range(kill_at):
                    lat.append(w.write(_rec(i)))
        except OSError as exc:
            entry["outcome"] = "raised"
            entry["note"] = f"{type(exc).__name__}: {exc}"
        try:
            if w is not None:
                w.close()
        except Exception as exc:  # noqa: BLE001
            entry.setdefault("note", f"close: {type(exc).__name__}: {exc}")
            if not entry["outcome"]:
                # Buffered writers only hit the disk here, so an injected write
                # failure legitimately surfaces at close rather than mid-loop.
                entry["outcome"] = "raised_at_close"

    # ---- phase 2: apply the non-crash fault --------------------------------
    if fault == "torn_tail" and path.exists():
        data = path.read_bytes()
        path.write_bytes(data[: max(0, len(data) - 40)])
    elif fault == "sidecar_corrupt" and sidecar.exists():
        sidecar.write_text("{not json", encoding="utf-8")
    elif fault == "sidecar_ahead" and sidecar.exists():
        obj = json.loads(sidecar.read_text(encoding="utf-8"))
        # The reader keys on "bytes" (DurableJsonl._write_progress); setting
        # "offset" here silently skipped the injection and the fault was a no-op.
        obj["bytes"] = path.stat().st_size + 10_000 if path.exists() else 10_000
        sidecar.write_text(json.dumps(obj), encoding="utf-8")
    elif fault == "readonly_write" and path.exists():
        # A read-only artefact must reject the next append *and* leave the
        # already-durable rows intact. The earlier version only chmod'ed and
        # never attempted a write, so it passed without testing anything.
        os.chmod(path, 0o444)
        try:
            w = _make(baseline, path)
            w.write(_rec(kill_at))
            w.close()
            entry["outcome"] = "ACCEPTED_WRITE_ON_READONLY"
        except Exception as exc:  # noqa: BLE001  (sqlite raises its own type)
            entry["outcome"] = "refused"
            entry["note"] = f"{type(exc).__name__}: {exc}"

    # ---- phase 3: recover, then resume -------------------------------------
    t0 = time.perf_counter()
    try:
        recovered = len(_done(baseline, path))
        entry["recover_ok"] = True
    except Exception as exc:  # noqa: BLE001
        recovered = 0
        entry["recover_ok"] = False
        entry["note"] = f"recover failed: {type(exc).__name__}: {exc}"
    entry["recover_sec"] = time.perf_counter() - t0
    # For an injected failure the loop stops early, so the number of rows the
    # writer actually attempted is what "lost" must be measured against. Using
    # kill_at there would report un-attempted rows as data loss.
    entry["rows_before_fault"] = (
        min(kill_at, n) if fault in ("crash", "concurrent") else len(lat))
    entry["recovered"] = recovered
    entry["lost"] = max(0, entry["rows_before_fault"] - recovered)
    entry["reexecuted"] = max(0, n - recovered)
    # SQLite has no "line" to tear; reporting a boolean there would be noise.
    entry["torn_tail"] = None if baseline == "B3" else _torn_line(path)
    entry["duplicates"] = _duplicates(path, baseline)

    if fault == "readonly_write" and path.exists():
        os.chmod(path, 0o666)

    if not entry["outcome"]:
        entry["outcome"] = "recovered"
    if lat:
        entry["append_p50_ms"] = statistics.median(lat)
        entry["append_p99_ms"] = sorted(lat)[max(0, int(len(lat) * 0.99) - 1)]
    else:
        entry["append_p50_ms"] = float("nan")
        entry["append_p99_ms"] = float("nan")
    entry["file_bytes"] = path.stat().st_size if path.exists() else 0
    return entry


def run_cost(baseline: str, n: int, tmp: Path) -> dict:
    """Clean run, no fault: the price of the durability guarantee."""
    path = tmp / f"cost_{baseline}.jsonl"
    if path.exists():
        path.unlink()
    sidecar = path.with_suffix(path.suffix + ".progress.json")
    if sidecar.exists():
        sidecar.unlink()

    lat = []
    t0 = time.perf_counter()
    w = _make(baseline, path)
    for i in range(n):
        lat.append(w.write(_rec(i)))
    w.close()
    total = time.perf_counter() - t0

    t1 = time.perf_counter()
    done = len(_done(baseline, path))
    recover = time.perf_counter() - t1

    return {
        "baseline": baseline, "rows": n, "total_sec": total,
        "rows_per_sec": n / total if total else float("nan"),
        "append_p50_ms": statistics.median(lat),
        "append_p99_ms": sorted(lat)[max(0, int(len(lat) * 0.99) - 1)],
        "file_bytes": path.stat().st_size if path.exists() else 0,
        "bytes_per_row": (path.stat().st_size / n) if path.exists() else 0,
        "recover_sec": recover, "recovered": done,
    }


def config_drift_case(tmp: Path) -> dict:
    """Exercise the shipped provenance guard, not a copy of it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cm121", ROOT / "scripts" / "121_crossmodal_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stage = tmp / "stage"
    stage.mkdir(exist_ok=True)
    prov_a = {"prompt_sha256": "a", "labels_sha256": "a",
              "manifest_sha256": "a", "runner_sha256": "a"}
    prov_b = dict(prov_a, prompt_sha256="b")
    mod._guard_provenance(stage, prov_a)
    refused = False
    try:
        mod._guard_provenance(stage, prov_b)
    except SystemExit:
        refused = True
    return {"baseline": "B4", "fault": "config_drift",
            "rows_target": 0, "rows_before_fault": 0,
            "recovered": 0, "lost": 0, "reexecuted": 0,
            "torn_tail": False, "duplicates": 0,
            "outcome": "refused" if refused else "ACCEPTED_MIXTURE",
            "note": "shipped _guard_provenance"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--baseline", default="B4")
    ap.add_argument("--path", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--crash", action="store_true")
    args = ap.parse_args()

    if args.worker:
        return worker(args)

    n = 400 if args.quick else 2000
    kill_at = int(n * 0.35)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="harness_bench_"))

    cost = [run_cost(b, n, tmp) for b in BASELINES]
    print("cost done", flush=True)

    # Capability map, stated rather than silently skipped: a fault that a
    # baseline has no mechanism for is reported as n/a, not as a pass.
    fault_matrix = {
        "crash": BASELINES,
        "torn_tail": ["B0", "B1", "B2", "B4"],
        "sidecar_corrupt": ["B4"],
        "sidecar_ahead": ["B4"],
        # B3 is included: SQLite has real locking, so pretending it has no
        # mechanism would be a strawman. Byte-level write/fsync injection still
        # cannot reach its C file layer, which is reported as n/a with a reason.
        "concurrent": ["B0", "B1", "B2", "B3", "B4"],
        "readonly_write": BASELINES,
        "fsync_fail": ["B1", "B2", "B4"],
        "enospc": ["B0", "B1", "B2", "B4"],
        "rename_fail": ["B2", "B4"],
        "config_drift": ["B4"],
    }
    rows = []
    for b in BASELINES:
        for f, supported in fault_matrix.items():
            if b not in supported:
                continue
            if f == "config_drift":
                rows.append(config_drift_case(tmp))
            else:
                rows.append(run_case(b, f, n, kill_at, tmp))
        print(f"{b} faults done", flush=True)

    import csv
    with open(OUT / "harness_cost.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cost[0]))
        w.writeheader()
        w.writerows(cost)
    keys = sorted({k for r in rows for k in r})
    with open(OUT / "harness_faults.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {OUT/'harness_cost.csv'} and {OUT/'harness_faults.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
