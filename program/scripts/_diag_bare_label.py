"""One-off: what exactly are the chexagent 'bare label' failures?"""
import json
from collections import Counter

path = "outputs/pretest/nih_eval/chexagent_8b/shift/shift_predictions.jsonl"
short = Counter()
for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    if (r.get("parsed") or {}).get("parse_success"):
        continue
    raw = (r.get("raw_output") or "").strip()
    if len(raw) < 30:
        short[repr(raw)] += 1
print("distinct short (bare-label) replies and their counts:")
for k, v in short.most_common():
    print(f"  {v:3d} x {k}")