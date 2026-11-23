"""Crash-safe append-only JSONL streaming for every experiment runner.

Why this module exists
----------------------
Every stage runner used to do this::

    with out_path.open("a" if done else "w", encoding="utf-8") as f:
        for batch in ...:
            ...run the model...
            for row in batch:
                f.write(json.dumps(row) + "\\n")
            f.flush()

Three things go wrong with that under a power cut, which on this project is a
realistic event (the full run is >10 days of continuous GPU time):

1. ``f.flush()`` only hands bytes to the operating system. On Windows they can
   still sit in the file cache, so a kill -9 or a power loss loses the tail.
   ``os.fsync`` is what actually forces them out.
2. Resume was keyed on ``parsed.parse_success``. A row written from a failed
   generation has ``parse_success == false``, so it is *not* considered done and
   is retried -- but a row written from a *truncated* JSON reply is treated the
   same way as a row written from a genuinely empty reply, and there was no way
   to distinguish "never attempted" from "attempted and failed".
3. A partial line at the end of the file (the classic torn write) was silently
   dropped by ``load_done``. That is the right behaviour for reading, but it
   means the file and the set of completed work disagree, and the disagreement
   grows every restart.

What this module does instead
-----------------------------
Every record is appended, then **fsynced**, then the durable byte offset is
recorded in a sidecar ``<name>.progress.json`` written atomically
(write temp -> ``os.replace``). On open, the reader:

* truncates the JSONL back to the last durable offset, so a torn tail cannot
  survive a crash, and
* reports exactly which ``(image_id, condition)`` keys are already present.

``retry_failed`` controls whether rows whose generation raised are re-attempted.
The default is to retry them, because a transient CUDA error should not become a
permanent hole in the data; pass ``retry_failed=False`` when every failure is
known to be deterministic (for example a prompt the model can never satisfy).

The file is opened in line-buffered mode as well, so a normal Ctrl-C still loses
at most the in-flight batch even on a filesystem where fsync is a no-op.

Usage::

    from src.cxr_trust.inference.durable import DurableJsonl

    with DurableJsonl(out_path, key=lambda r: (r["image_id"], r["condition"])) as out:
        pending = [j for j in jobs if out.needs(j.image_id, j.condition)]
        for batch in chunks(pending, batch_size):
            rows = run(batch)                 # may raise; nothing is written
            out.write_many(rows)              # append + fsync + progress marker
        print(out.stats())
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


def _fsync_file(fh) -> None:
    """Flush Python buffers and force the OS to write through to the device."""
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except (OSError, AttributeError, ValueError):
        # fsync can be unavailable or meaningless on some filesystems; a flush is
        # still strictly better than nothing.
        pass


def read_jsonl(path: Path) -> list[dict]:
    """Read every complete JSON object from a JSONL file.

    A trailing partial line (torn write) is skipped rather than raising, because
    this is also used to count what already exists.
    """
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


class DurableJsonl:
    """Append-only JSONL with an atomic progress sidecar.

    Parameters
    ----------
    path
        The JSONL file. The sidecar is ``<path>.progress.json``.
    key
        Maps a record to its ``(image_id, condition)`` identity tuple.
    retry_failed
        When True (default), rows whose ``error`` field is non-empty are treated
        as *not* done and will be attempted again on the next run.
    """

    def __init__(self, path: Path, key: Callable[[dict], tuple],
                 retry_failed: bool = True) -> None:
        self.path = Path(path)
        self.progress_path = self.path.with_suffix(self.path.suffix + ".progress.json")
        self.key = key
        self.retry_failed = retry_failed
        self.done: set[tuple] = set()
        self.records: list[dict] = []
        self._fh = None
        self._offset = 0
        self._appended = 0
        self._truncated_bytes = 0
        self._deduped = 0
        self._replaced = 0
        self._index: dict[tuple, int] = {}
        self.progress_error: str | None = None

    # ---------------------------------------------------------------- lifecycle
    def _load_progress(self) -> dict:
        if not self.progress_path.exists():
            return {}
        try:
            return json.loads(self.progress_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_progress(self, n_records: int) -> None:
        """Atomically replace the sidecar so it can never be half-written."""
        body = {
            "jsonl": self.path.name,
            "bytes": self._offset,
            "records": n_records,
            "appended_this_run": self._appended,
            "truncated_bytes_on_open": self._truncated_bytes,
        }
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.progress_path.parent),
                                   prefix=self.progress_path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(body, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.progress_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def __enter__(self) -> "DurableJsonl":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        progress = self._load_progress()

        if self.path.exists():
            size = self.path.stat().st_size
            durable = progress.get("bytes")
            if isinstance(durable, int) and 0 <= durable <= size:
                if size > durable:
                    # Everything past the last durable marker is suspect: it may
                    # be a complete row that was written but not fsynced, or a
                    # torn row. Dropping it is the safe choice -- the work is
                    # simply redone -- because keeping a torn row would corrupt
                    # every downstream aggregate.
                    with self.path.open("r+b") as fh:
                        fh.truncate(durable)
                    self._truncated_bytes = size - durable
                self._offset = durable
            else:
                # No sidecar (first run after this upgrade, or it was deleted):
                # trust the file as far as its complete lines go, and record that
                # offset from now on.
                self._offset = self._usable_size()
            self.records = read_jsonl(self.path)
        else:
            self._offset = 0
            self.records = []

        # Collapse superseded attempts. A row that failed is retried on the next
        # run, and retrying appends; without this the file accumulates a duplicate
        # for every retry of every failed cell, and -- worse -- it would then hold
        # BOTH the failed attempt and the successful one for the same key. Any
        # reader that takes the last row per key still works, but any reader that
        # counts rows, or takes the first, silently reports the failure.
        #
        # This runs unconditionally: retry_failed only decides whether a *failed*
        # row blocks the next attempt, not whether duplicates are cleaned up.
        self.records = self._dedupe(self.records)
        self._index = {}
        for i, rec in enumerate(self.records):
            k = self._key_of(rec)
            if k is not None:
                self._index[k] = i

        for rec in self.records:
            if rec.get("error") and self.retry_failed:
                continue
            try:
                self.done.add(self.key(rec))
            except (KeyError, TypeError):
                continue

        self._fh = self.path.open("ab")
        return self

    def _dedupe(self, records: list[dict]) -> list[dict]:
        """Keep one record per key: the last successful one, else the last attempt.

        Records whose key cannot be computed are kept as they are rather than
        dropped, so a malformed line can never be silently deleted. Order is
        preserved, so the file still reads as a log of the work that was kept.
        """
        best: dict[tuple, int] = {}
        unkeyed: list[int] = []
        for idx, rec in enumerate(records):
            try:
                k = self.key(rec)
            except (KeyError, TypeError):
                unkeyed.append(idx)
                continue
            failed = bool(rec.get("error"))
            prev = best.get(k)
            if prev is None:
                best[k] = idx
                continue
            prev_failed = bool(records[prev].get("error"))
            # prefer a success over a failure; otherwise prefer the newer attempt
            if (prev_failed and not failed) or (prev_failed == failed):
                best[k] = idx

        keep = sorted(set(best.values()) | set(unkeyed))
        # nothing to collapse and nothing keyless: leave the file untouched
        if len(keep) == len(records):
            return records
        out = [records[i] for i in keep]
        self._rewrite(out)
        # how many duplicate/superseded lines this open removed, not how many
        # records remain -- those are different numbers and conflating them hid a
        # real bug during development.
        self._deduped = len(records) - len(out)
        return out

    def _rewrite(self, records: list[dict]) -> None:
        """Atomically replace the JSONL with ``records`` and reset the offset.

        On Windows ``os.replace`` fails with WinError 5 (access denied) if the
        destination is still open by this process, so the append handle is closed
        first and reopened afterwards. That is also the correct order on POSIX:
        a write handle opened before the swap would keep pointing at the unlinked
        inode and silently append into nothing.
        """
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent),
                                   prefix=self.path.name + ".", suffix=".rewrite")
        had_handle = self._fh is not None
        if had_handle:
            _fsync_file(self._fh)
            self._fh.close()
            self._fh = None
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload.encode("utf-8"))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        finally:
            if had_handle:
                self._fh = self.path.open("ab")
        self._offset = len(payload.encode("utf-8"))
        # `_deduped` is owned by _dedupe (it counts removed lines, not kept ones);
        # _rewrite only resets the byte offset so appends land at the new end.

    def _usable_size(self) -> int:
        """Byte length of the file up to and including its last complete line."""
        total = 0
        with self.path.open("rb") as fh:
            for line in fh:
                if line.endswith(b"\n"):
                    total += len(line)
                else:
                    break
        return total

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            _fsync_file(self._fh)
            self._fh.close()
            self._fh = None
        # Record the durable offset even on an exception, so a crash inside a
        # stage still leaves a resumable state.  A failing sidecar is surfaced
        # rather than swallowed: it means the next resume falls back to the
        # last complete line, which is safe but costs re-work.
        try:
            self._write_progress(len(self.records))
        except OSError as exc:
            self.progress_error = str(exc)
            print(f"[durable] WARNING: could not write progress sidecar "
                  f"{self.progress_path}: {exc}", file=sys.stderr)
        return None

    # ------------------------------------------------------------------ reading
    def needs(self, *key_parts) -> bool:
        """True when this key has no durable result yet.

        Takes the key as separate components rather than a fixed pair: a shift cell
        is keyed by (image, condition, severity) and an ablation cell by (image,
        variant, condition), and the first version of this method accepted exactly
        two arguments, which forced those callers to reach into ``self.done``
        directly. Accepting ``*key_parts`` lets every caller say what its key
        actually is, and converts a passing tuple into the same shape.
        """
        if len(key_parts) == 1 and isinstance(key_parts[0], tuple):
            return tuple(str(x) for x in key_parts[0]) not in self.done
        return tuple(str(x) for x in key_parts) not in self.done

    def pending(self, jobs: Iterable[Any],
                key_of: Callable[[Any], tuple]) -> list[Any]:
        """Filter ``jobs`` (any objects) to those not yet completed."""
        out = []
        for job in jobs:
            image_id, condition = key_of(job)
            if self.needs(image_id, condition):
                out.append(job)
        return out

    # ------------------------------------------------------------------ writing
    def write_many(self, rows: Sequence[dict]) -> int:
        """Append rows, fsync, then mark the new durable offset. Returns count.

        A row whose key is already present **replaces** the stored attempt instead
        of appending beside it. Appending is the fast path and is used for every
        first-time result; replacement costs one atomic rewrite and only happens
        when a previously failed cell is retried, which is rare by construction.
        Without this, retrying a failed cell inside a single run appended a second
        row for the same key and the file ended up holding both the failure and
        the success.
        """
        if not rows:
            return 0
        if self._fh is None:
            raise RuntimeError("DurableJsonl used outside its context manager")

        # collapse repeats INSIDE this batch first, last one wins
        ordered: list[dict] = []
        index: dict[tuple, int] = {}
        for rec in rows:
            k = self._key_of(rec)
            if k is None:
                ordered.append(rec)
                continue
            if k in index:
                ordered[index[k]] = rec
            else:
                index[k] = len(ordered)
                ordered.append(rec)

        def store(rec: dict) -> bool:
            """Replace the existing attempt for this key in place, or append."""
            k = self._key_of(rec)
            if k is not None and k in self._index:
                idx = self._index[k]
                self.records[idx] = rec
                return True
            self.records.append(rec)
            if k is not None:
                self._index[k] = len(self.records) - 1
            return False

        replaces = 0
        appends: list[dict] = []
        for rec in ordered:
            k = self._key_of(rec)
            if k is not None and k in self._index:
                store(rec)
                replaces += 1
            else:
                appends.append(rec)

        if replaces:
            # A mixed batch appends new keys *and* supersedes old ones.  The
            # append-only path would write the new rows but leave the replaced
            # failure on disk, so the file would still hold the failed attempt.
            # Add the new records to the in-memory view, then rewrite once.
            for rec in appends:
                self.records.append(rec)
                k = self._key_of(rec)
                if k is not None:
                    self._index[k] = len(self.records) - 1
            self._rewrite(self.records)
        elif appends:
            payload = b"".join(
                (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in appends)
            self._fh.write(payload)
            _fsync_file(self._fh)
            self._offset += len(payload)
            for rec in appends:
                self.records.append(rec)
                k = self._key_of(rec)
                if k is not None:
                    self._index[k] = len(self.records) - 1

        for rec in ordered:
            if rec.get("error") and self.retry_failed:
                continue
            k = self._key_of(rec)
            if k is not None:
                self.done.add(k)

        self._appended += len(appends)
        self._replaced += replaces
        self._write_progress(len(self.records))
        return len(ordered)

    def _key_of(self, rec: dict):
        try:
            return self.key(rec)
        except (KeyError, TypeError):
            return None

    def write(self, row: dict) -> int:
        return self.write_many([row])

    # -------------------------------------------------------------------- stats
    def stats(self) -> dict:
        failed = sum(1 for r in self.records if r.get("error"))
        return {
            "file": str(self.path),
            "records_on_disk": len(self.records),
            "already_done_this_run": len(self.done),
            "failed_rows_on_disk": failed,
            "appended_this_run": self._appended,
            "replaced_this_run": self._replaced,
            "truncated_bytes_on_open": self._truncated_bytes,
            "rows_collapsed_on_open": self._deduped,
            "durable_bytes": self._offset,
        }
