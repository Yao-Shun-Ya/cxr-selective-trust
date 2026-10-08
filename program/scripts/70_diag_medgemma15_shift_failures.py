"""One-off: characterise the 3 medgemma_15 shift rows that fail to parse after the
2026-10-08 uniform 2048-token re-run.

The floor in 66_run_prerun_retry.ps1 asserts 1900/1900 on the rationale "2048 clears
every trace truncation". That rationale came from the PRECISION stage (bf16, reliance
prompt), whose longest reply was 1495 tokens. The shift stage is a different prompt
AND 4-bit, and it still fails 3 cells. This script answers the only question that
decides the fix: are these replies TRUNCATED at the 2048 cap (a budget problem ->
raise it) or has the model switched into its reasoning channel and degenerated into a
repetition loop that never opens the JSON object (a model pathology -> report it)?

Three independent checks, all GPU-free:

  1. does the reply contain a single '{' at all?
  2. is the tail EXACTLY periodic? Under greedy decoding (the study protocol:
     do_sample=False, num_beams=1) an exact cycle is an ABSORBING state -- the same
     tokens map to the same next tokens forever, so no token budget can leave it.
  3. is the PRE-RE-RUN backup reply a PREFIX of the new reply? Greedy decoding is
     deterministic, so a prefix means the new reply is literally the old generation
     continued: the extra tokens went into extending the trajectory, not changing it.

Not part of the pipeline.
"""
import json
from collections import Counter
from pathlib import Path

CUR = Path("outputs/pretest/nih_eval/medgemma_15_4b_it/shift/shift_predictions.jsonl")
BAK = Path("outputs/pretest_retry_backup/medgemma_15_4b_it_fullrerun_20261008/shift/shift_predictions.jsonl")


def load(path):
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[(r.get("image_id"), r.get("condition"), r.get("severity"))] = r
    return out


def find_period(s, max_len=3000, max_p=1500):
    """Smallest p such that the last max_len chars are exactly periodic with period p."""
    tail = s[-min(len(s), max_len):]
    n = len(tail)
    for p in range(1, min(max_p, n // 2) + 1):
        if all(tail[i] == tail[i % p] for i in range(p, n)):
            return p, tail[-p:]
    return None, None


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


cur = load(CUR)
bak = load(BAK) if BAK.exists() else {}
print("current rows=%d   backup rows=%d" % (len(cur), len(bak)))
print("row keys:", sorted(next(iter(cur.values())).keys()))
print()

fails = [r for r in cur.values() if not (r.get("parsed") or {}).get("parse_success")]
print("parse failures: %d of %d" % (len(fails), len(cur)))
print("=" * 78)

for r in sorted(fails, key=lambda x: (x.get("condition") or "", x.get("severity") or "")):
    raw = r.get("raw_output") or ""
    key = (r.get("image_id"), r.get("condition"), r.get("severity"))
    print("%s/%s   image=%s   rawlen=%d chars" % (r.get("condition"), r.get("severity"),
                                                  r.get("image_id"), len(raw)))
    print("  starts with '<unused94>thought' : %s" % raw.startswith("<unused94>"))
    print("  contains '{' anywhere           : %s" % ("{" in raw))
    print("  contains '\"findings\"'           : %s" % ('"findings"' in raw))
    p, unit = find_period(raw)
    if p:
        print("  tail EXACTLY periodic           : period=%d chars" % p)
        print("    repeating unit: %r" % unit)
    else:
        print("  tail exactly periodic           : no (period > 1500 chars)")

    b = bak.get(key)
    if b is None:
        print("  backup: (no row for this cell)")
    else:
        braw = b.get("raw_output") or ""
        print("  backup rawlen=%d  backup_has_'{'=%s" % (len(braw), "{" in braw))
        print("  backup IS PREFIX of new         : %s" % raw.startswith(braw))
        print("    longest common prefix = %d of %d backup chars" % (lcp(braw, raw), len(braw)))
        pb, _ = find_period(braw)
        print("    backup tail periodic          : %s" % (("period=%d" % pb) if pb else "no"))
    print("=" * 78)

by_img = Counter(r.get("image_id") for r in fails)
print("failures per image:", dict(by_img))
marker_rows = [r for r in cur.values()
               if (r.get("raw_output") or "").startswith("<unused94>")]
print("rows that opened the thought channel : %d" % len(marker_rows))
print("  ...of those, that still parsed     : %d"
      % sum(1 for r in marker_rows if (r.get("parsed") or {}).get("parse_success")))