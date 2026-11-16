#!/usr/bin/env python
"""109 - CheXbert cross-check of the Open-i report layer.

The paper's Open-i omission/hallucination table uses a deliberately simple
rule-based labeller applied to both the reference and the generated report, so
a labeller bias cancels in the ranking.  This script replaces the labeller with
CheXbert (Smit et al., EMNLP 2020) and asks whether the model ranking survives.

Inputs (frozen predictions):
  outputs/matrix/openi/<model>/shift/image_reliance_predictions.jsonl
  data/manifests/openi_rrg.csv  (reference_text)

Outputs:
  outputs/analysis_chexbert/openi_chexbert_reports.csv
  outputs/analysis_chexbert/openi_chexbert_metrics.csv

The CheXbert code and checkpoint live in program/third_party/CheXbert:
  git clone https://github.com/stanfordmlgroup/CheXbert
  checkpoint: StanfordAIMI/RRG_scorers/chexbert.pth
  vocab:      bert-base-uncased/vocab.txt

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\109_chexbert_openi.py
  .\\.venv-vl\\Scripts\\python.exe scripts\\109_chexbert_openi.py --limit 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "manifests" / "openi_rrg.csv"
MATRIX = ROOT / "outputs" / "matrix" / "openi"
CHEX = ROOT / "third_party" / "CheXbert"
CKPT = CHEX / "checkpoints" / "chexbert.pth"
VOCAB = CHEX / "checkpoints" / "bert_base" / "vocab.txt"
OUT = ROOT / "outputs" / "analysis_chexbert"

MODELS = ["medgemma_4b_it", "medgemma_15_4b_it", "internvl3_5_8b",
          "qwen3_vl_8b_instruct", "qwen25_vl_7b_instruct", "chexagent_8b"]

# CheXbert output order (src/constants.py)
CONDITIONS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
              "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
              "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
              "Support Devices", "No Finding"]
TARGETS = {
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
    "consolidation": "Consolidation",
    "edema": "Edema",
    "pleural_effusion": "Pleural Effusion",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
    "no_finding": "No Finding",
}


class CheXbert(nn.Module):
    """Same head layout as third_party/CheXbert/src/models/bert_labeler.py.

    BertModel is built from a local BertConfig instead of from_pretrained, so
    the checkpoint supplies every weight and no network access is needed.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.bert = BertModel(BertConfig())
        self.dropout = nn.Dropout(p)
        hidden = self.bert.pooler.dense.in_features
        self.linear_heads = nn.ModuleList([nn.Linear(hidden, 4) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hidden, 2))

    def forward(self, ids, mask):
        h = self.bert(ids, attention_mask=mask)[0][:, 0, :]
        h = self.dropout(h)
        return [head(h) for head in self.linear_heads]


@torch.inference_mode()
def label_reports(texts: list[str], model, tok, device: str,
                  batch_size: int = 18) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=512,
                  return_tensors="pt")
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        pred = torch.stack([x.argmax(dim=1) for x in logits], dim=1).cpu().numpy()
        out.append(pred)
    return np.concatenate(out, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=18)
    args = ap.parse_args()

    man = pd.read_csv(MAN, low_memory=False)
    man = man[["image_id", "reference_text"]].dropna(subset=["reference_text"])
    if args.limit:
        man = man.head(args.limit)

    rows = [{"image_id": r.image_id, "kind": "reference", "model": "reference",
             "text": str(r.reference_text)} for r in man.itertuples()]
    keep = set(man.image_id)
    for slug in MODELS:
        p = MATRIX / slug / "shift" / "rrg_predictions.jsonl"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if str(r.get("condition")) != "original":
                    continue
                if args.limit and str(r.get("image_id")) not in keep:
                    continue
                rows.append({"image_id": str(r.get("image_id")), "kind": "generated",
                             "model": slug, "text": str(r.get("raw_output") or "")})

    df = pd.DataFrame(rows)
    print(f"reports: {len(df)} ({df.model.nunique()} sources)", flush=True)

    tok = BertTokenizer(vocab_file=str(VOCAB), do_lower_case=True)
    model = CheXbert()
    ck = torch.load(CKPT, map_location="cpu")
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    pred = label_reports(df.text.tolist(), model, tok, device, args.batch_size)
    for j, cond in enumerate(CONDITIONS):
        df[cond] = pred[:, j]

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "openi_chexbert_reports.csv", index=False, encoding="utf-8-sig")

    # Omission/hallucination against the reference CheXbert labels.
    ref = df[df.kind == "reference"].set_index("image_id")
    metrics = []
    for slug in MODELS:
        gen = df[(df.kind == "generated") & (df.model == slug)].set_index("image_id")
        common = sorted(set(gen.index) & set(ref.index))
        if not common:
            continue
        om = ha = pos_ref = 0
        for iid in common:
            for lab, cond in TARGETS.items():
                rp = int(ref.loc[iid, cond]) == 1
                gp = int(gen.loc[iid, cond]) == 1
                pos_ref += int(rp)
                om += int(rp and not gp)
                ha += int((not rp) and gp)
        metrics.append({"model": slug, "n_studies": len(common),
                        "reference_positive_slots": pos_ref,
                        "omissions": om, "hallucinations": ha,
                        "omission_rate": round(om / pos_ref, 4) if pos_ref else None})
    mdf = pd.DataFrame(metrics).sort_values("omissions")
    mdf.to_csv(OUT / "openi_chexbert_metrics.csv", index=False, encoding="utf-8-sig")
    print(mdf.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
