"""Responsiveness audit: is a layer's "image reliance" even measurable?

Why this runs BEFORE the geographic-shift analysis
--------------------------------------------------
A blank-control contrast is only interpretable if the model's output actually
varies across real images. If a model emits the same string for every one of
1,500 radiographs, then ``J(original, blank) == 1`` says nothing about whether
it reads the image -- there is no variation to explain. On 2026-10-23 the
PadChest layer turned out to be exactly that case for five of six models
(1--20 distinct outputs for 1,500 images), and the Open-i blank contrast turned
out to be a *template switch* rather than an image-reliance reading (every
model's blank output collapses to one fixed string).

What is measured, per (layer, model)
------------------------------------
1. **Diversity of real answers**: ``n_unique_text`` / ``top_share_text`` (with a
   Wilson 95% CI) and, for structured layers, ``n_unique_status`` -- the ordered
   status vector, so formatting cannot inflate diversity.
2. **Blank contrast against its own null**. Two metric spaces, because they
   answer different questions:
   * text Jaccard: ``j_real_blank`` vs ``j_real_real``;
   * structured probability vector: ``mean|P_real - P_blank|`` vs the same
     distance between two DIFFERENT real images.

   ``gap = value(real, blank) - value(real, other real)``. **Negative** means the
   blank is *more* distinct than a random real image (some separation); **near
   zero** means the blank is indistinguishable from a random real report (no
   evidence of reliance); **positive** means the blank is *more* similar than a
   random real report (the model answers with its prior).
3. **What the blank separation means**: if the modal blank reply is a refusal
   ("uninterpretable", "no image data"), the contrast measures *absence
   detection*, not evidence extraction. Recorded as ``blank_top_is_refusal``.

The collapse rule is **descriptive, not fitted**: ``top_share_text >= 0.5`` or
``n_unique_text <= 5``. It never filters data; it only says whether a reliance
number may be read as reliance.

Outputs (``--output-dir``)
--------------------------
``responsiveness_by_cell.csv``  one row per (layer, model)
``responsiveness_summary.json`` flags, denominators, declared rule

CPU only; reads only files the three layers already produced.

Usage::

    python scripts/78_responsiveness_audit.py --output-dir outputs/analysis_responsiveness
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    "medgemma_4b_it",
    "medgemma_15_4b_it",
    "internvl3_5_8b",
    "qwen3_vl_8b_instruct",
    "chexagent_8b",
    "qwen25_vl_7b_instruct",
]

LAYERS: dict[str, dict[str, Any]] = {
    "nih": {
        "path": "nih_eval/{m}/reliance/image_reliance_predictions.jsonl",
        "real_condition": "original",
        "blank_condition": "blank",
        "structured": True,
    },
    "padchest": {
        "path": "padchest/{m}/shift/image_reliance_predictions.jsonl",
        "real_condition": "original",
        "blank_condition": "blank",
        "structured": True,
    },
    "openi": {
        "path": "openi/{m}/shift/rrg_predictions.jsonl",
        "real_condition": "original",
        "blank_condition": "blank",
        "structured": False,
    },
}

LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"]

COLLAPSE_TOP_SHARE = 0.50
COLLAPSE_MIN_UNIQUE = 5
MAX_PAIRS = 400
REFUSAL_RE = re.compile(
    r"uninterpretable|no image|lack of image|unable to (?:interpret|assess)|"
    r"cannot (?:interpret|assess)|image (?:is )?(?:missing|not (?:provided|available))",
    re.I)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", (text or "").lower()))


def jaccard(a: str, b: str) -> float:
    A, B = tokens(a), tokens(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def status_vector(parsed: dict) -> tuple | None:
    findings = (parsed or {}).get("findings") or {}
    out = []
    for lb in LABELS:
        st = (findings.get(lb) or {}).get("status")
        if st not in {"positive", "negative", "uncertain"}:
            return None
        out.append(st)
    return tuple(out)


def prob_vector(parsed: dict) -> tuple | None:
    findings = (parsed or {}).get("findings") or {}
    out = []
    for lb in LABELS:
        p = (findings.get(lb) or {}).get("probability_positive")
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            return None
        out.append(float(p))
    return tuple(out)


def mean_abs(a: tuple, b: tuple) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def read_layer(path: Path) -> dict[str, dict[str, dict]]:
    """condition -> image_id -> row."""
    out: dict[str, dict[str, dict]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            out.setdefault(str(obj.get("condition")), {})[
                str(obj.get("image_id"))] = obj
    return out


def diversity(texts: list[str], statuses: list[tuple | None]) -> dict:
    texts = [norm(t) for t in texts]
    n = len(texts)
    if n == 0:
        return dict(n=0, n_unique_text=0, top_share_text=None, top_ci=None,
                    n_usable_status=0, n_unique_status=0,
                    top_share_status=None, top_status_ci=None)
    counts: dict[str, int] = {}
    for t in texts:
        counts[t] = counts.get(t, 0) + 1
    top_text, top_n = max(counts.items(), key=lambda kv: kv[1])
    d = dict(n=n, n_unique_text=len(counts),
             top_share_text=round(top_n / n, 4),
             top_ci=[round(v, 4) for v in wilson(top_n, n)],
             top_text_example=top_text[:180])
    usable = [s for s in statuses if s is not None]
    if usable:
        sc: dict[tuple, int] = {}
        for s in usable:
            sc[s] = sc.get(s, 0) + 1
        top_s, top_sn = max(sc.items(), key=lambda kv: kv[1])
        d.update(n_usable_status=len(usable), n_unique_status=len(sc),
                 top_share_status=round(top_sn / len(usable), 4),
                 top_status_ci=[round(v, 4) for v in wilson(top_sn, len(usable))])
    else:
        d.update(n_usable_status=0, n_unique_status=0,
                 top_share_status=None, top_status_ci=None)
    return d


def contrasts(real_map: dict, blank_map: dict, seed: int = 0) -> dict:
    """Text and probability contrasts, each against a real-vs-real null."""
    ids = sorted(set(real_map) & set(blank_map))
    if not ids:
        return dict(n_pairs=0, j_real_blank=None, j_real_real=None, gap_text=None,
                    n_real_real=0, dprob_real_blank=None, dprob_real_real=None,
                    gap_prob=None, n_prob_pairs=0, n_prob_real_real=0)

    j_rb = sum(jaccard(real_map[i].get("raw_output", ""),
                       blank_map[i].get("raw_output", "")) for i in ids) / len(ids)

    rng = random.Random(seed)
    all_real = sorted(real_map)
    pairs: list[tuple[str, str]] = []
    for _ in range(MAX_PAIRS):
        a, b = rng.choice(all_real), rng.choice(all_real)
        if a != b:
            pairs.append((a, b))
    j_rr = (sum(jaccard(real_map[a].get("raw_output", ""),
                        real_map[b].get("raw_output", "")) for a, b in pairs) / len(pairs)
            if pairs else None)

    # Probability space: only over images where BOTH sides parsed to 8 probabilities.
    prob_ids = [i for i in ids
                if prob_vector(real_map[i].get("parsed") or {}) is not None
                and prob_vector(blank_map[i].get("parsed") or {}) is not None]
    d_rb = None
    d_rr = None
    if prob_ids:
        d_rb = sum(mean_abs(prob_vector(real_map[i]["parsed"]),
                            prob_vector(blank_map[i]["parsed"]))
                   for i in prob_ids) / len(prob_ids)
        valid = [i for i in all_real
                 if prob_vector(real_map[i].get("parsed") or {}) is not None]
        ppairs = [(a, b) for a, b in pairs if a in valid and b in valid]
        if ppairs:
            d_rr = sum(mean_abs(prob_vector(real_map[a]["parsed"]),
                                prob_vector(real_map[b]["parsed"]))
                       for a, b in ppairs) / len(ppairs)

    return dict(
        n_pairs=len(ids),
        j_real_blank=round(j_rb, 4),
        j_real_real=(round(j_rr, 4) if j_rr is not None else None),
        gap_text=(round(j_rb - j_rr, 4) if j_rr is not None else None),
        n_real_real=len(pairs),
        n_prob_pairs=len(prob_ids),
        dprob_real_blank=(round(d_rb, 4) if d_rb is not None else None),
        dprob_real_real=(round(d_rr, 4) if d_rr is not None else None),
        gap_prob=(round(d_rb - d_rr, 4) if (d_rb is not None and d_rr is not None)
                  else None),
        n_prob_real_real=len(ppairs) if prob_ids else 0,
    )


def audit_layer(layer: str, model: str, matrix: Path) -> dict | None:
    spec = LAYERS[layer]
    data = read_layer(matrix / spec["path"].format(m=model))
    real_map = data.get(spec["real_condition"], {})
    blank_map = data.get(spec["blank_condition"], {})
    if not real_map and not blank_map:
        return None

    real_text = [r.get("raw_output") or "" for r in real_map.values()]
    real_status = [status_vector(r.get("parsed") or {}) for r in real_map.values()]
    blank_text = [r.get("raw_output") or "" for r in blank_map.values()]
    blank_status = [status_vector(r.get("parsed") or {}) for r in blank_map.values()]

    rd = diversity(real_text, real_status)
    bd = diversity(blank_text, blank_status)
    con = contrasts(real_map, blank_map)

    top_share = rd["top_share_text"] or 0.0
    collapsed = bool(top_share >= COLLAPSE_TOP_SHARE
                     or rd["n_unique_text"] <= COLLAPSE_MIN_UNIQUE)
    refusal = bool(REFUSAL_RE.search(bd.get("top_text_example") or ""))

    # Which metric space carries the blank effect, if any.
    if con["gap_prob"] is not None:
        primary = "probability"
        effect = con["gap_prob"]           # positive = blank moves P more than a real pair
    else:
        primary = "text"
        effect = (None if con["gap_text"] is None else -con["gap_text"])

    return {
        "layer": layer, "model": model, "structured": spec["structured"],
        "real_n": rd["n"],
        "real_n_unique_text": rd["n_unique_text"],
        "real_top_share_text": rd["top_share_text"],
        "real_top_share_ci": rd["top_ci"],
        "real_n_usable_status": rd["n_usable_status"],
        "real_n_unique_status": rd["n_unique_status"],
        "real_top_share_status": rd["top_share_status"],
        "real_top_share_status_ci": rd["top_status_ci"],
        "real_top_text_example": rd["top_text_example"][:180],
        "blank_n": bd["n"],
        "blank_n_unique_text": bd["n_unique_text"],
        "blank_top_share_text": bd["top_share_text"],
        "blank_top_text_example": bd["top_text_example"][:180],
        "blank_top_is_refusal": refusal,
        **con,
        "effect_space": (primary if (effect is not None and not collapsed) else None),
        "blank_effect": effect,
        "output_collapsed": collapsed,
        "blank_contrast_usable": bool(not collapsed),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-root", default=str(ROOT / "outputs" / "matrix"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = Path(args.matrix_root)

    rows = []
    for layer in LAYERS:
        for model in args.models:
            r = audit_layer(layer, model, matrix)
            if r is None:
                print(f"[--] no predictions for {layer}/{model}", flush=True)
            else:
                rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "responsiveness_by_cell.csv", index=False, encoding="utf-8-sig")

    summary: dict[str, Any] = {
        "declared_rule": {
            "collapse_top_share_text_gte": COLLAPSE_TOP_SHARE,
            "collapse_n_unique_text_lte": COLLAPSE_MIN_UNIQUE,
            "note": "descriptive, not fitted; never filters data",
        },
        "null_models": {
            "text": "mean Jaccard between two DIFFERENT real images",
            "probability": "mean |P(a) - P(b)| between two DIFFERENT real images",
            "pairs": f"up to {MAX_PAIRS} seeded random pairs per cell",
        },
        "n_cells": len(df),
        "by_layer": {},
    }
    for layer, g in df.groupby("layer"):
        summary["by_layer"][layer] = {
            "n_models": int(len(g)),
            "n_output_collapsed": int(g["output_collapsed"].sum()),
            "models_collapsed": sorted(g.loc[g["output_collapsed"], "model"].tolist()),
            "n_blank_contrast_usable": int(g["blank_contrast_usable"].sum()),
            "n_blank_is_refusal": int(g["blank_top_is_refusal"].sum()),
            "median_top_share_text": float(g["real_top_share_text"].median()),
            "median_gap_text": (None if g["gap_text"].isna().all()
                                else float(g["gap_text"].median())),
            "median_gap_prob": (None if g["gap_prob"].isna().all()
                                else float(g["gap_prob"].median())),
        }
    (out_dir / "responsiveness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    cols = ["layer", "model", "real_n", "real_n_unique_text", "real_top_share_text",
            "gap_text", "dprob_real_blank", "dprob_real_real", "gap_prob",
            "blank_top_is_refusal", "output_collapsed", "blank_contrast_usable"]
    print(df[cols].to_string(index=False), flush=True)
    print(f"\nwrote {out_dir / 'responsiveness_by_cell.csv'}")
    print(json.dumps(summary["by_layer"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
