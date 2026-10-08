import json, collections
from pathlib import Path

R = Path(r"d:\Paper\program\outputs\pretest\nih_eval")
labels = ['atelectasis','cardiomegaly','consolidation','edema','pleural_effusion','pneumonia','pneumothorax']
cells = ['clean__none','blur__severe','brightness__severe','contrast__severe','jpeg__severe','noise__severe','resolution__severe']

def rows(model):
    return [json.loads(l) for l in open(R/model/"shift"/"shift_predictions.jsonl", encoding="utf-8") if l.strip()]

print("=== positive pathology instances (GT) per cell, per model ===")
for m in sorted(p.name for p in R.iterdir() if p.is_dir()):
    by = collections.defaultdict(int)
    for r in rows(m):
        g = r.get("ground_truth") or {}
        cell = f"{r['condition']}__{r['severity']}"
        by[cell] += sum(1 for lb in labels if g.get("label_"+lb) == "positive")
    # total across all cells
    tot = sum(by.values())
    line = "  ".join(f"{c.split('__')[1] if '__' in c else c}:{by.get(c,0)}" for c in cells)
    print(f"  {m:22} total_pos={tot}  [{line}]")

print()
print("=== qwen25 shift status distribution ===")
st = collections.Counter()
for r in rows("qwen25_vl_7b_instruct"):
    fs = (r.get("parsed") or {}).get("findings") or {}
    for lb, fd in fs.items():
        st[(fd or {}).get("status")] += 1
print("  ", dict(st))

print()
print("=== chexagent shift status distribution ===")
st = collections.Counter()
for r in rows("chexagent_8b"):
    fs = (r.get("parsed") or {}).get("findings") or {}
    for lb, fd in fs.items():
        st[(fd or {}).get("status")] += 1
print("  ", dict(st))