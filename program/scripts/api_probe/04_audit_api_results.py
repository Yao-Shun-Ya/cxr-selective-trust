"""Step 4: reuse the project's audit definitions on the API predictions.

P1 responsiveness, P2 null separation (with the phase-scrambled control), P3
operating point and P4 contract coverage, grouped by cohort. Distance helpers are
imported from ``122_analyze_crossmodal.py`` so the API arm and the local arm
cannot drift apart in their definitions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "api_probe"


def _load_helpers():
    spec = importlib.util.spec_from_file_location(
        "cm122", ROOT / "scripts" / "122_analyze_crossmodal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_helpers()
LABELS = {
    "cxr": ["atelectasis", "cardiomegaly", "consolidation", "edema",
            "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"],
    "mri": ["meningioma", "glioma", "pituitary_tumor"],
    "oct": ["choroidal_neovascularization", "diabetic_macular_edema",
            "drusen", "normal"],
    "derm": ["actinic_keratosis", "basal_cell_carcinoma", "benign_keratosis",
             "dermatofibroma", "melanoma", "melanocytic_nevus", "vascular_lesion"],
    "openi": ["atelectasis", "cardiomegaly", "consolidation", "edema",
              "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"],
}
MALIGNANT = {"melanoma", "basal_cell_carcinoma", "actinic_keratosis"}
PATHOLOGY = ["atelectasis", "cardiomegaly", "consolidation", "edema",
             "pleural_effusion", "pneumonia", "pneumothorax"]


def load(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        f = (o.get("parsed") or {}).get("findings") or {}
        rows.append({
            "probe_id": o["probe_id"], "cohort": o["cohort"],
            "dataset": o["dataset"], "image_id": o["image_id"],
            "condition": o["condition"], "schema": o["schema"],
            "cluster_id": o.get("cluster_id") or o["image_id"],
            "cluster_level": o.get("cluster_level", "image"),
            "raw": o.get("raw_output") or "", "error": o.get("error") or "",
            "parsed_ok": bool((o.get("parsed") or {}).get("parse_success")),
            "completeness": float(o.get("label_completeness") or 0.0),
            "usage": o.get("usage") or {}, "latency": float(o.get("latency_s") or 0),
            "gt": o.get("ground_truth") or {},
            "findings": f,
        })
    return pd.DataFrame(rows)


def vec(findings: dict, labels: list[str], ok: bool) -> tuple:
    if not ok:
        return tuple(["unparsed"] * len(labels))
    return tuple((findings.get(l) or {}).get("status", "not_mentioned")
                 for l in labels)


def prob(findings: dict, labels: list[str]) -> list:
    return [(findings.get(l) or {}).get("probability_positive") for l in labels]


def p1(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cohort, g in df.groupby("cohort"):
        labels = LABELS[cohort]
        g = g.copy()
        g["vec"] = [vec(f, labels, ok) for f, ok in zip(g["findings"], g["parsed_ok"])]
        clean = g[g["condition"].isin(["clean", "native", "real"])].set_index("image_id")
        for cond, sub in g.groupby("condition"):
            if cond in ("clean", "native", "real") or cond.startswith("blank"):
                continue
            joined = sub[sub["image_id"].isin(clean.index)]
            if joined.empty:
                continue
            changed = [H._d_text(clean.loc[r.image_id, "vec"], r.vec) > 0
                       for r in joined.itertuples()]
            k, n = int(np.sum(changed)), len(changed)
            lo, hi = H._wilson(k, n)
            out.append({"cohort": cohort, "condition": cond, "n": n,
                        "p1_rate": k / n, "p1_lo": lo, "p1_hi": hi,
                        "distinct_answers": joined["vec"].nunique(),
                        "modal_share": (Counter(joined["vec"]).most_common(1)[0][1] / n),
                        "parse_rate": float(joined["parsed_ok"].mean())})
    return pd.DataFrame(out)


def _d_words(a: str, b: str) -> float:
    """1 - Jaccard over lowercased word sets. Prose has no status vector."""
    wa = {w.strip(".,;:()[]") for w in (a or "").lower().split() if w.strip()}
    wb = {w.strip(".,;:()[]") for w in (b or "").lower().split() if w.strip()}
    if not wa and not wb:
        return 0.0
    return 1.0 - len(wa & wb) / len(wa | wb)


def p2(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    # Schema is held at v0 here: the v1 blank is a separate contract question.
    df = df[df["schema"].isin(["v0", "report"])]
    for (cohort, dataset), g in df.groupby(["cohort", "dataset"]):
        labels = LABELS[cohort]
        g = g.copy()
        if cohort == "openi":
            g["vec"] = [tuple() for _ in range(len(g))]
        else:
            g["vec"] = [vec(f, labels, ok)
                        for f, ok in zip(g["findings"], g["parsed_ok"])]
        clean = g[g["condition"].isin(["clean", "native", "real"])].sort_values("image_id")
        if clean.empty:
            continue
        ids = list(clean["image_id"])
        partner = {a: b for a, b in zip(ids, ids[1:] + ids[:1])}
        clean = clean.set_index("image_id")
        for blank_kind, sub in g[g["condition"].str.startswith("blank")].groupby("condition"):
            sub = sub.set_index("image_id")
            both = [i for i in sub.index if i in clean.index and partner.get(i) in clean.index]
            if not both:
                continue
            if cohort == "openi":
                d_blank = np.array([_d_words(clean.loc[i, "raw"], sub.loc[i, "raw"])
                                    for i in both])
                d_pair = np.array([_d_words(clean.loc[i, "raw"], clean.loc[partner[i], "raw"])
                                   for i in both])
            else:
                d_blank = np.array([H._d_text(clean.loc[i, "vec"], sub.loc[i, "vec"])
                                    for i in both])
                d_pair = np.array([H._d_text(clean.loc[i, "vec"], clean.loc[partner[i], "vec"])
                                   for i in both])
            delta = d_blank - d_pair
            clusters = np.array([clean.loc[i, "cluster_id"] for i in both])
            lo, hi = H._cluster_boot(delta, clusters)
            out.append({"cohort": cohort, "dataset": dataset, "blank": blank_kind,
                        "n": len(both),
                        "D_real_blank": float(d_blank.mean()),
                        "D_real_realprime": float(d_pair.mean()),
                        "delta_null": float(delta.mean()),
                        "delta_lo": lo, "delta_hi": hi})
    return pd.DataFrame(out)


def p3(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (cohort, dataset), g in df.groupby(["cohort", "dataset"]):
        if cohort in ("openi", "mri"):
            continue
        clean = g[g["condition"].isin(["clean", "native"])]
        if clean.empty:
            continue
        for cond, sub in [("clean", clean)] + [
                (c, g[g["condition"] == c]) for c in
                ("contrast_severe", "noise_severe", "resolution_severe")
                if c in set(g["condition"])]:
            pos, neg = _split_pos_neg(cohort, sub)
            if not pos and not neg:
                continue
            out.append({
                "cohort": cohort, "dataset": dataset, "condition": cond,
                "n_pos": len(pos), "n_neg": len(neg),
                "positive_call_rate_on_neg": float(np.mean(neg)) if neg else float("nan"),
                "recall_on_pos": float(np.mean(pos)) if pos else float("nan"),
            })
    return pd.DataFrame(out)


def _split_pos_neg(cohort: str, sub: pd.DataFrame):
    pos, neg = [], []
    for r in sub.itertuples():
        f, gt = r.findings, r.gt
        if cohort == "cxr" and r.dataset == "nih":
            truth = str(gt.get("label_pneumonia", "")) == "positive"
            called = (f.get("pneumonia") or {}).get("status") == "positive"
        elif cohort == "cxr" and r.dataset == "padchest":
            truth = any(str(gt.get(f"label_{x}", "")) == "positive" for x in PATHOLOGY)
            called = any((f.get(x) or {}).get("status") == "positive" for x in PATHOLOGY)
        elif cohort == "derm":
            truth = str(gt.get("label", "")) in MALIGNANT
            called = any((f.get(x) or {}).get("status") == "positive" for x in MALIGNANT)
        elif cohort == "oct":
            truth = str(gt.get("label", "")) != "normal"
            called = any((f.get(x) or {}).get("status") == "positive"
                         for x in ("choroidal_neovascularization",
                                   "diabetic_macular_edema", "drusen"))
        else:
            continue
        (pos if truth else neg).append(1 if called else 0)
    return pos, neg


def p4(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (cohort, schema), g in df.groupby(["cohort", "schema"]):
        out.append({"cohort": cohort, "schema": schema, "n": len(g),
                    "parse_rate": float(g["parsed_ok"].mean()),
                    "label_completeness": float(g["completeness"].mean()),
                    "error_rows": int((g["error"] != "").sum()),
                    "empty_replies": int((g["raw"].str.strip() == "").sum())})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="deepseek_v4_1_flash")
    ap.add_argument("--input", default=None)
    args = ap.parse_args()
    base = OUT / args.slug
    path = Path(args.input) if args.input else base / "predictions.jsonl"
    df = load(path)
    print(f"loaded {len(df)} rows from {path}")
    for name, fn in (("p1_responsiveness", p1), ("p2_null", p2),
                     ("p3_operating_point", p3), ("p4_contract", p4)):
        res = fn(df)
        res.to_csv(base / f"{name}.csv", index=False)
        print(f"\n== {name} ==\n{res.to_string(index=False)}")
    usage = pd.DataFrame([{
        "cohort": r.cohort, "latency_s": r.latency,
        "prompt_tokens": r.usage.get("prompt_tokens"),
        "completion_tokens": r.usage.get("completion_tokens"),
        "error": r.error,
    } for r in df.itertuples()])
    usage.to_csv(base / "usage.csv", index=False)
    print("\n== usage ==")
    print(usage.groupby("cohort")[["latency_s", "prompt_tokens",
                                   "completion_tokens"]].mean().to_string())
    print("errors:", int((usage["error"] != "").sum()))


if __name__ == "__main__":
    main()
