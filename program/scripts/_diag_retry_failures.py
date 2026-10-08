"""One-off: inspect the rows that failed to parse after the 2026-10-08 retry.

Not part of the pipeline. Prints, per stage, the parse-fail count by condition and
the tail of each failed reply, so the failure MODE is visible (digit loop vs
truncated thought trace vs something else) instead of only the rate.
"""
import json
from collections import Counter

TARGETS = [
    ("chexagent_8b/shift",
     "outputs/pretest/nih_eval/chexagent_8b/shift/shift_predictions.jsonl"),
    ("chexagent_8b/reliance",
     "outputs/pretest/nih_eval/chexagent_8b/reliance/image_reliance_predictions.jsonl"),
    ("medgemma_15/precision(bf16)",
     "outputs/pretest/nih_eval/medgemma_15_4b_it/reliance_bf16/image_reliance_predictions.jsonl"),
    ("medgemma_15/shift",
     "outputs/pretest/nih_eval/medgemma_15_4b_it/shift/shift_predictions.jsonl"),
    ("medgemma_15/reliance",
     "outputs/pretest/nih_eval/medgemma_15_4b_it/reliance/image_reliance_predictions.jsonl"),
]

for name, path in TARGETS:
    n = fail = 0
    cond = Counter()
    modes = Counter()
    loops = 0
    tails = []
    try:
        fh = open(path, encoding="utf-8")
    except OSError as exc:
        print("=" * 74)
        print("%s: CANNOT OPEN (%s)" % (name, exc))
        continue
    for line in fh:
        line = line.strip()
        if not line:
            continue
        n += 1
        r = json.loads(line)
        if (r.get("parsed") or {}).get("parse_success"):
            continue
        fail += 1
        c = "%s/%s" % (r.get("condition"), r.get("severity"))
        cond[c] += 1
        raw = r.get("raw_output") or ""
        if "<br>" in raw:
            modes["html_br"] += 1
        elif len(raw.strip()) < 30:
            modes["bare_label"] += 1
        elif "//" in raw:
            modes["js_comment"] += 1
        elif len(raw) > 3000:
            modes["long_other"] += 1
        else:
            modes["other"] += 1
        tail = raw[-60:]
        digits = [ch for ch in tail if ch.isdigit()]
        if len(digits) >= 40 and len(set(digits)) <= 2:
            loops += 1
        if len(tails) < 1:
            tails.append((c, r.get("image_id"), len(raw), raw))
    print("=" * 74)
    print("%s: rows=%d parsed=%d fail=%d (digit-loop tails=%d)"
          % (name, n, n - fail, fail, loops))
    print("  fail by condition/severity:", dict(sorted(cond.items())))
    print("  fail MODES:", dict(modes.most_common()))
    for c, iid, L, t in tails:
        print("  --- FULL RAW %s  %s  rawlen=%d ---" % (c, iid, L))
        print(t)
        print("  --- END ---")