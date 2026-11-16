#!/usr/bin/env python
"""94 - DenseNet-121 supervised baseline on the available NIH images.

Data reality
------------
The repository holds 5,958 NIH PNGs, not the full 112k ChestX-ray14 release.
They cover 3,871 `internal_train`, 735 `source_val`, and 1,352 `internal_test`
images, with **no patient overlap** between the three splits.  This is a small
but honest baseline: it is trained only on `internal_train`, never on the
evaluation rows, and evaluated on the same 1,506-image study set used by the
VLM paper (plus the 1,054 `internal_test` subset).

Why it matters
--------------
A VLM macro AUROC of 0.50--0.63 can be read as "the task is too hard" rather
than "the model is weak".  A supervised baseline trained on the same labels
separates those explanations.  The baseline is deliberately simple
(DenseNet-121, 224px, BCE, fixed 8 epochs, no early stopping) so it cannot be
accused of being tuned on the test split.

Usage
-----
Dry run (CPU only, prints split/label counts)::

    .\\.venv-vl\\Scripts\\python.exe scripts\\94_train_densenet_baseline.py --dry-run

Train + evaluate::

    .\\.venv-vl\\Scripts\\python.exe scripts\\94_train_densenet_baseline.py \
        --epochs 8 --batch-size 32 --pretrained

Outputs `outputs/densenet_baseline/`:
  metrics.json, per_label_metrics.csv, test_predictions.csv, best_model.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "nih_full.csv"
IMAGES = ROOT / "data" / "nih" / "images"
EVAL_MANIFEST = ROOT / "data" / "manifests" / "nih_eval_1506.csv"
OUT = ROOT / "outputs" / "densenet_baseline"

LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"]
PATHOLOGY = [l for l in LABELS if l != "no_finding"]


def load_rows() -> pd.DataFrame:
    d = pd.read_csv(MANIFEST, low_memory=False)
    d["path"] = [IMAGES / r for r in d["image_relpath"]]
    d["exists"] = [p.exists() for p in d["path"]]
    return d[d["exists"]].copy()


def label_matrix(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        (d[f"label_{l}"].astype(str).str.lower() == "positive").to_numpy()
        for l in LABELS
    ]).astype(np.float32)


class NihDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tf):
        self.frame = frame.reset_index(drop=True)
        self.tf = tf
        self.y = label_matrix(frame)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        row = self.frame.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.tf(img), torch.from_numpy(self.y[i])


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    out = {}
    for i, l in enumerate(LABELS):
        yy, pp = y[:, i], p[:, i]
        out[l] = float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else float("nan")
    out["macro_pathology_auroc"] = float(np.nanmean([out[l] for l in PATHOLOGY]))
    out["macro_all8_auroc"] = float(np.nanmean([out[l] for l in LABELS]))
    return out


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(p, bins[1:-1], right=False)
    total = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def build_model(pretrained: bool):
    import torch.nn as nn
    from torchvision import models
    weights = None
    if pretrained:
        try:
            from torchvision.models import DenseNet121_Weights
            weights = DenseNet121_Weights.IMAGENET1K_V1
        except Exception:
            weights = "IMAGENET1K_V1"
    try:
        m = models.densenet121(weights=weights)
    except Exception as exc:  # noqa: BLE001 - offline fallback must not kill the run
        print(f"[warn] pretrained weights unavailable ({exc}); training from scratch")
        m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, len(LABELS))
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = load_rows()
    train = d[d.split == "internal_train"]
    val = d[d.split == "source_val"]
    test_all = d[d.split.isin(["internal_test", "source_val"])]
    eval_ids = set(pd.read_csv(EVAL_MANIFEST)["image_id"])
    test_eval = d[d.image_id.isin(eval_ids)]
    test_internal = d[d.split == "internal_test"]

    print(f"train {len(train)} | val {len(val)} | eval {len(test_eval)} | "
          f"internal_test {len(test_internal)}")
    print("train patients", train.patient_id.nunique(),
          "test patients", test_internal.patient_id.nunique(),
          "overlap", len(set(train.patient_id) & set(test_internal.patient_id)))
    pos = label_matrix(train).sum(axis=0)
    print("train positives", dict(zip(LABELS, pos.astype(int))))
    if args.dry_run:
        return 0

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = build_model(args.pretrained).to(device)
    pos_weight = torch.tensor(
        (len(train) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(NihDataset(train, train_tf), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, pin_memory=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * len(x)
        print(f"epoch {epoch}/{args.epochs} loss {running/len(train):.4f} "
              f"elapsed {time.time()-t0:.1f}s")

    model.eval()
    torch.save(model.state_dict(), OUT / "best_model.pt")

    def predict(frame):
        ds = NihDataset(frame, eval_tf)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        probs = []
        with torch.inference_mode():
            for x, _ in dl:
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = model(x.to(device, non_blocking=True))
                probs.append(torch.sigmoid(logits).float().cpu().numpy())
        return np.concatenate(probs, axis=0)

    p_eval = predict(test_eval)
    y_eval = label_matrix(test_eval)
    p_int = predict(test_internal)
    y_int = label_matrix(test_internal)

    per_label = []
    for i, l in enumerate(LABELS):
        for name, y, p in [("eval_1506", y_eval, p_eval), ("internal_test", y_int, p_int)]:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y[:, i], p[:, i])) if len(np.unique(y[:, i])) == 2 else float("nan")
            per_label.append({"split": name, "label": l, "auroc": auc,
                              "positives": int(y[:, i].sum()),
                              "ece": ece(y[:, i], p[:, i])})
    per_label_df = pd.DataFrame(per_label)
    per_label_df.to_csv(OUT / "per_label_metrics.csv", index=False)

    result = {
        "train_n": len(train), "val_n": len(val),
        "eval_n": len(test_eval), "internal_test_n": len(test_internal),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "pretrained": bool(args.pretrained),
        "eval_1506": metrics(y_eval, p_eval),
        "internal_test": metrics(y_int, p_int),
        "prevalence": {l: float(y_eval[:, i].mean()) for i, l in enumerate(LABELS)},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(p_eval, columns=LABELS).assign(
        image_id=test_eval["image_id"].to_numpy()).to_csv(
        OUT / "test_predictions.csv", index=False)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
