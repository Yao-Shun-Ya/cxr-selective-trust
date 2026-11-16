"""Cross-modal analysis: P1, P2, P4, collapse, CXR-vocabulary leakage, P3.

Reads the JSONL written by ``121_crossmodal_benchmark.py`` and writes one CSV per
metric under ``outputs/analysis_crossmodal/``.

Metric definitions follow docs/58 section 13.3 and the CXR arm:
  D_text(a,b) = 1 - Jaccard(positive sets)   (0 when both sets are empty)
  D_prob(a,b) = mean |p_a - p_b| over labels both sides emit
  Delta_null  = D(real, blank) - D(real, real')
A parse failure is carried as the token ``unparsed`` rather than dropped: a
contract failure is a behavioural change, and P4 reports its rate separately.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_crossmodal"

CXR_TERMS = [
    "cardiomegaly", "pleural effusion", "pneumothorax", "consolidation",
    "atelectasis", "pulmonary edema", "mediastinal", "cardiopulmonary",
    "chest x-ray", "chest radiograph", "no acute abnormality",
]

POSITIVE_SETS = {
    "derm": {"melanoma", "basal_cell_carcinoma", "actinic_keratosis"},
    "oct": {"choroidal_neovascularization", "diabetic_macular_edema", "drusen"},
}

def _load(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        parsed = obj.get("parsed") or {}
        findings = parsed.get("findings") or {}
        rows.append({
            "image_id": obj["image_id"],
            # Identity comes from the record, not the directory: a renamed or
            # reused output directory must not be able to merge two models into
            # one analysis cell.
            "model": obj.get("model") or path.parent.name,
            "modality": obj.get("modality", ""),
            "schema": obj.get("schema", ""),
            "cluster_id": obj.get("cluster_id", "") or obj["image_id"],
            "cluster_level": obj.get("cluster_level", "image"),
            "condition": obj["condition"],
            "severity": obj["severity"],
            "raw_output": obj.get("raw_output", ""),
            "parse_success": bool(parsed.get("parse_success")),
            # Fraction of the contract's labels that the reply names explicitly.
            # parse_success alone is too lenient: a one-key JSON object passes.
            "label_completeness": float(obj.get("label_completeness", float("nan"))),
            "findings": findings,
            "label": (obj.get("ground_truth") or {}).get("label", ""),
        })
    return pd.DataFrame(rows)


def _status_vec(findings: dict, labels: list[str], ok: bool) -> tuple:
    if not ok:
        return tuple(["unparsed"] * len(labels))
    return tuple((findings.get(l) or {}).get("status", "not_mentioned") for l in labels)


def _prob_vec(findings: dict, labels: list[str]) -> list[float | None]:
    return [(findings.get(l) or {}).get("probability_positive") for l in labels]


def _d_text(a: tuple, b: tuple) -> float:
    pa = {i for i, s in enumerate(a) if s == "positive"}
    pb = {i for i, s in enumerate(b) if s == "positive"}
    if not pa and not pb:
        return 0.0
    return 1.0 - len(pa & pb) / len(pa | pb)


def _d_prob(a, b) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return 0.0
    return float(np.mean([abs(x - y) for x, y in pairs]))


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _cluster_boot(values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000):
    uniq = np.unique(clusters)
    rng = np.random.default_rng(20261114)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in pick])
        if len(idx):
            means.append(values[idx].mean())
    if not means:
        return (float("nan"), float("nan"))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def analyze(path: Path, labels: list[str],
            pair: dict[str, str]) -> dict[str, pd.DataFrame]:
    df = _load(path)
    triples = df[["model", "modality", "schema"]].drop_duplicates()
    if len(triples) != 1:
        raise ValueError(f"{path} mixes {len(triples)} configurations: "
                         f"{triples.to_dict('records')}")
    model, modality, schema = (
        df["model"].iloc[0], df["modality"].iloc[0], df["schema"].iloc[0])
    cluster_level = df["cluster_level"].iloc[0]

    df["vec"] = [
        _status_vec(f, labels, ok) for f, ok in zip(df["findings"], df["parse_success"])
    ]
    df["prob"] = [_prob_vec(f, labels) for f in df["findings"]]
    df["cxr_leak"] = df["raw_output"].str.lower().apply(
        lambda s: any(t in s for t in CXR_TERMS))

    clean = df[df["condition"] == "clean"].set_index("image_id")
    cells = df[~df["condition"].str.startswith("blank_")]

    # ---- P1 responsiveness and P4 contract coverage, per grid cell ----------
    recs = []
    for (cond, sev), grp in cells.groupby(["condition", "severity"]):
        if cond == "clean":
            continue
        joined = grp[grp["image_id"].isin(clean.index)]
        changed = [
            _d_text(clean.loc[r["image_id"], "vec"], r["vec"]) > 0
            for _, r in joined.iterrows()
        ]
        k = int(np.sum(changed))
        n = len(changed)
        lo, hi = _wilson(k, n)
        recs.append({
            "model": model, "modality": modality, "schema": schema,
            "cluster_level": cluster_level,
            "condition": cond, "severity": sev,
            "p1_rate": k / n if n else float("nan"), "n": n,
            "p1_lo": lo, "p1_hi": hi,
            "parse_rate": float(joined["parse_success"].mean()) if n else float("nan"),
            "label_completeness": (float(joined["label_completeness"].mean())
                                   if n else float("nan")),
            "contract_violation_rate": (float((joined["label_completeness"] < 1).mean())
                                        if n else float("nan")),
            "modal_share": (Counter(joined["vec"]).most_common(1)[0][1] / n) if n else float("nan"),
            "cxr_leak_rate": float(joined["cxr_leak"].mean()) if n else float("nan"),
        })
    p1 = pd.DataFrame(recs)

    # ---- P2 null separation -------------------------------------------------
    recs = []
    for kind in [c for c in df["condition"].unique() if c.startswith("blank_")]:
        blank = df[df["condition"] == kind].set_index("image_id")
        ids = [i for i in blank.index if i in clean.index and pair.get(i) in clean.index]
        if not ids:
            continue
        d_blank = np.array([_d_text(clean.loc[i, "vec"], blank.loc[i, "vec"]) for i in ids])
        d_pair = np.array([_d_text(clean.loc[i, "vec"], clean.loc[pair[i], "vec"]) for i in ids])
        dp_blank = np.array([_d_prob(clean.loc[i, "prob"], blank.loc[i, "prob"]) for i in ids])
        dp_pair = np.array([_d_prob(clean.loc[i, "prob"], clean.loc[pair[i], "prob"]) for i in ids])
        delta = d_blank - d_pair
        delta_p = dp_blank - dp_pair
        cluster_map = dict(zip(df["image_id"], df["cluster_id"]))
        clusters = np.array([cluster_map.get(i, i) for i in ids])
        lo, hi = _cluster_boot(delta, clusters)
        recs.append({
            "model": model, "modality": modality, "schema": schema,
            "cluster_level": cluster_level,
            "n_clusters": int(len(set(clusters))),
            "blank_kind": kind, "n": len(ids),
            "D_real_blank": float(d_blank.mean()),
            "D_real_realprime": float(d_pair.mean()),
            "delta_null": float(delta.mean()), "delta_lo": lo, "delta_hi": hi,
            "delta_null_prob": float(delta_p.mean()),
            "parse_rate_blank": float(blank.loc[ids, "parse_success"].mean()),
        })
    p2 = pd.DataFrame(recs)

    # ---- P3 operating point (modalities with a clinical negative class) -----
    p3 = pd.DataFrame()
    if modality in POSITIVE_SETS:
        pos = POSITIVE_SETS[modality]
        neg_labels = sorted(set(labels) - pos)
        rows = []
        for cond, sev in [("clean", "none"), ("noise", "severe"), ("resolution", "severe")]:
            sub = df[(df["condition"] == cond) & (df["severity"] == sev)]
            neg = sub[sub["label"].isin(neg_labels)]
            called = [any((f.get(l) or {}).get("status") == "positive" for l in pos)
                      for f in neg["findings"]]
            rows.append({
                "model": model, "modality": modality, "schema": schema,
                "cluster_level": cluster_level,
                "condition": cond, "severity": sev,
                "n_negative": len(neg),
                "positive_call_rate_on_negative": float(np.mean(called)) if len(neg) else float("nan"),
            })
        p3 = pd.DataFrame(rows)

    return {"p1": p1, "p2": p2, "p3": p3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "outputs" / "crossmodal"))
    ap.add_argument("--labels", default=None,
                    help="JSON mapping modality -> label list; defaults to installed vocab")
    args = ap.parse_args()

    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.cxr_trust.inference.prompts_crossmodal import MODALITY_LABELS

    OUT.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[pd.DataFrame]] = {"p1": [], "p2": [], "p3": []}

    for path in sorted(Path(args.root).glob("*/*/*/shift_predictions.jsonl")):
        modality = path.parents[2].name
        manifest = ROOT / "data" / "manifests" / f"crossmodal_{modality}.csv"
        if not manifest.exists():
            print(f"[skip] no manifest for {modality}")
            continue
        man = pd.read_csv(manifest)
        pair = dict(zip(man["image_id"], man["pair_id"]))
        labels = MODALITY_LABELS[modality]
        print(f"[run] {path.parent}")
        res = analyze(path, labels, pair)
        for k, v in res.items():
            if len(v):
                buckets[k].append(v)

    for k, frames in buckets.items():
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df.to_csv(OUT / f"crossmodal_{k}.csv", index=False)
            print(f"[wrote] {OUT / f'crossmodal_{k}.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
