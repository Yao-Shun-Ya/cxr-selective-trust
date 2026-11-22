"""Analyse the frozen-encoder probe: patient-grouped CV and split-held-out AUROC.

The probe is the readout control for the main claim. If the frozen representation
carries the label, a linear classifier on pooled vision features should separate
it even when the prompted VLM sits at chance. We report both a patient-grouped
5-fold CV over the 1,506 evaluation studies and the pre-declared
source_val -> internal_test split.

    python scripts/128_analyze_probe.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "probe"
LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax"]
MODELS = ["medgemma_15_4b_it", "qwen3_vl_8b_instruct", "internvl3_5_8b"]


def _load(model: str) -> tuple[np.ndarray, list[str]]:
    path = OUT / model / "features.jsonl"
    ids, vecs = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        ids.append(obj["image_id"])
        vecs.append(np.frombuffer(base64.b64decode(obj["embedding_b64"]),
                                  dtype=np.float32))
    return np.stack(vecs), ids


def _macro(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for j in range(y.shape[1]):
        ok = ~np.isnan(p[:, j])
        if ok.sum() < 2 or len(np.unique(y[ok, j])) < 2:
            continue
        vals.append(roc_auc_score(y[ok, j], p[ok, j]))
    return float(np.mean(vals)) if vals else float("nan")


def _cluster_boot(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                  n_boot: int = 1000) -> tuple[float, float]:
    uniq = np.unique(groups)
    rows = {u: np.where(groups == u)[0] for u in uniq}
    rng = np.random.default_rng(20261118)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows[u] for u in pick])
        vals.append(_macro(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _fit_predict(Xtr, ytr, Xte):
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    out = np.full(len(Xte), np.nan)
    if ytr.sum() < 5 or ytr.sum() == len(ytr):
        return out
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def main() -> None:
    manifest = pd.read_csv(ROOT / "data" / "manifests" / "nih_eval_1506.csv")
    manifest = manifest[manifest["split"].isin(["internal_test", "source_val"])].copy()
    manifest["image_id"] = manifest["image_id"].astype(str)
    labels = np.array([manifest[f"label_{lab}"].eq("positive").to_numpy()
                       for lab in LABELS]).T.astype(int)
    groups = manifest["patient_id"].astype(str).to_numpy()
    split = manifest["split"].astype(str).to_numpy()

    rows = []
    summary = {}
    for model in MODELS:
        X, ids = _load(model)
        order = {k: i for i, k in enumerate(ids)}
        keep = [i for i, k in enumerate(manifest["image_id"]) if k in order]
        if len(keep) != len(manifest):
            print(f"[warn] {model}: {len(manifest)-len(keep)} images without features")
        Xm = X[[order[manifest["image_id"].iloc[i]] for i in keep]]
        ym, gm, sm = labels[keep], groups[keep], split[keep]

        # patient-grouped 5-fold CV
        oof = np.full(ym.shape, np.nan)
        gkf = GroupKFold(n_splits=5)
        for tr, te in gkf.split(Xm, groups=gm):
            for j in range(len(LABELS)):
                oof[te, j] = _fit_predict(Xm[tr], ym[tr, j], Xm[te])
        macro = _macro(ym, oof)
        lo, hi = _cluster_boot(ym, oof, gm)

        # pre-declared split: source_val -> internal_test
        tr = np.where(sm == "source_val")[0]
        te = np.where(sm == "internal_test")[0]
        pred = np.full((len(te), len(LABELS)), np.nan)
        for j in range(len(LABELS)):
            pred[:, j] = _fit_predict(Xm[tr], ym[tr, j], Xm[te])
        macro_split = _macro(ym[te], pred)

        summary[model] = {"macro_cv": macro, "ci": [lo, hi],
                          "macro_source_val_to_test": macro_split,
                          "n": int(len(ym)), "dim": int(Xm.shape[1])}
        for j, lab in enumerate(LABELS):
            ok = ~np.isnan(oof[:, j])
            rows.append({
                "model": model, "label": lab,
                "auroc_cv": (float(roc_auc_score(ym[ok, j], oof[ok, j]))
                             if ok.sum() and len(np.unique(ym[ok, j])) > 1 else np.nan),
                "n_pos": int(ym[:, j].sum()), "n": int(len(ym)),
                "macro_cv": macro, "ci_lo": lo, "ci_hi": hi,
                "macro_split": macro_split,
            })
        print(f"{model}: macro CV {macro:.3f} [{lo:.3f}, {hi:.3f}], "
              f"split {macro_split:.3f}, dim {Xm.shape[1]}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "probe_auroc.csv", index=False)
    (OUT / "probe_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT/'probe_auroc.csv'} and probe_summary.json")


if __name__ == "__main__":
    main()
