"""Build the evaluation-only NIH manifest from the 6K-strategy split manifest.

Why this exists
---------------
``nih_shared_5000.csv`` carries three splits: ``internal_train`` (3,268),
``internal_test`` (1,142) and ``source_val`` (590). The stage scripts select rows
with ``--limit``, which takes the **first N rows of the file** -- not a sample.
Because the file is ordered by image_id, ``--limit 300`` yielded only 138
evaluation images and 12 pneumonia positives, and a large ``--limit`` would have
pulled training rows into the reported metrics without any warning.

This script writes the evaluation set explicitly: ``internal_test`` plus
``source_val`` = 1,506 scored images, plus 226 ``internal_train`` donor rows used
only to repair ``no_finding`` (1,732 manifest rows total), which is the number the
6K plan (docs/41) states.

It also repairs a second defect in the upstream sampling
--------------------------------------------------------
``stratified_select`` in ``23_build_stratified_manifest.py`` balances the seven
disease labels but treats ``no_finding`` as "handled separately" and never gives
it a positive target. The tail-filling step then orders candidates by *novelty*
-- how many still-under-fed labels they cover -- so images that are positive for
several diseases sort first and plain normals sort last, exactly where the
budget runs out. Result: ``nih_shared_5000.csv`` contains **0 of 5,000** rows
with ``label_no_finding == "positive"``, while the full pool (``nih_full.csv``)
has 60,361 of 112,120. A constant label makes AUROC and every selective metric
for ``no_finding`` undefined.

--no-finding-min fixes that by enforcing a floor, which means also converting
some evaluation rows the sampler picked as exclusively-negative into rows from
the training pool. Everything else is left untouched, and the swap is reported.

Usage::

    python scripts/40_build_eval_manifest.py
    python scripts/40_build_eval_manifest.py --no-finding-min 0.20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

STD_LABELS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "pleural_effusion", "pneumonia", "pneumothorax", "no_finding",
]
EVAL_SPLITS = ["internal_test", "source_val"]


def repair_no_finding(sel: pd.DataFrame, pool: pd.DataFrame, frac: float,
                      seed: int, prefer_splits=None) -> tuple[pd.DataFrame, dict]:
    """Raise the ``no_finding`` positive rate in ``sel`` to at least ``frac``.

    The sampler in ``23_build_stratified_manifest.py`` never gives
    ``no_finding`` a positive target, and its tail-filling step orders candidates
    by how many *still-under-fed disease labels* they cover -- so plain normals
    sort last and the budget runs out before any are taken. The result is a
    constant-negative label, which leaves AUROC and every selective metric for
    ``no_finding`` undefined while the row counts still look healthy.

    Rows are swapped rather than added, so the evaluation size and the split
    proportions stay exactly as requested. Donors come from the same pool, and
    the rows they replace are the least load-bearing ones available: first rows
    that are positive for no disease at all, otherwise rows whose rarest positive
    label is still common in the set.

    An earlier version of this function could not run at all, because *every* row
    the sampler selects is positive for at least one disease -- that is precisely
    what the sampler optimises for. The fallback below exists for that case.
    """
    col = "label_no_finding"
    n = len(sel)
    want = int(np.ceil(frac * n))
    have = int((sel[col] == "positive").sum())
    info = {"target": want, "before": have, "after": have, "swapped": 0,
            "drop_strategy": "none"}
    if have >= want:
        return sel, info

    need = want - have
    chosen_ids = set(sel["image_id"].astype(str))

    def _donors(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[(frame[col] == "positive") &
                     (~frame["image_id"].astype(str).isin(chosen_ids))]

    in_split = None
    if prefer_splits and "split" in pool.columns:
        pool_in = pool[pool["split"].isin(list(prefer_splits))]
        cand = _donors(pool_in)
        if len(cand) >= need:
            in_split = "yes"
            donors = cand
            info["donor_scope"] = "evaluation splits only"
        else:
            donors = _donors(pool)
            in_split = "no"
            info["donor_scope"] = (f"whole extracted pool ({len(cand)} available "
                                   f"inside the evaluation splits)")
    else:
        donors = _donors(pool)
        info["donor_scope"] = "whole extracted pool"

    disease_cols = [c for c in (f"label_{lb}" for lb in STD_LABELS if lb != "no_finding")
                    if c in sel.columns]

    # Preference 1: rows positive for nothing else -- removing them costs no
    # disease signal at all.
    all_negative = pd.Series(True, index=sel.index)
    for c in disease_cols:
        all_negative &= (sel[c] != "positive")
    drop_pool = sel[(sel[col] != "positive") & all_negative]
    info["drop_strategy"] = "spare-rows"

    # Preference 2 (the usual case): rank by how expendable each row is. A row is
    # expendable when its rarest positive label is still well represented, so
    # dropping it barely moves the least-supported metric.
    if len(drop_pool) < need:
        freq = {c: int((sel[c] == "positive").sum()) for c in disease_cols}
        # rows that are no_finding-negative, ordered by rarest-positive-label
        # frequency descending == most expendable first
        candidates = sel[sel[col] != "positive"].copy()
        rarest = np.full(len(candidates), 10 ** 9, dtype=float)
        for c in disease_cols:
            pos = (candidates[c] == "positive").to_numpy()
            rarest = np.where(pos, np.minimum(rarest, freq[c]), rarest)
        candidates = candidates.assign(_rarest=rarest)
        candidates = candidates.sort_values(
            by=["_rarest"], ascending=False, kind="mergesort")
        drop_pool = candidates.head(max(need, len(drop_pool)))
        info["drop_strategy"] = "least-load-bearing"

    if len(donors) < need or len(drop_pool) < need:
        info["warning"] = (
            f"cannot meet the floor: {len(donors)} donors and {len(drop_pool)} "
            f"droppable rows available for {need} swaps")
        need = min(need, len(donors), len(drop_pool))

    if need <= 0:
        return sel, info

    # Deterministic, and spread the drops instead of always taking the same tail.
    rng = np.random.default_rng(seed)
    if info["drop_strategy"] == "least-load-bearing" and len(drop_pool) > need:
        head = drop_pool.head(min(len(drop_pool), need * 2))
        drop_idx = rng.choice(head.index.to_numpy(), size=need, replace=False)
    else:
        pool_idx = drop_pool.index.to_numpy()
        if len(pool_idx) > need:
            rng.shuffle(pool_idx)
            pool_idx = pool_idx[:need]
        drop_idx = pool_idx
    donor_idx = rng.choice(donors.index.to_numpy(), size=need, replace=False)

    out = pd.concat([sel.drop(index=drop_idx), pool.loc[donor_idx]], axis=0)
    out = out[list(sel.columns)].reset_index(drop=True)
    info["after"] = int((out[col] == "positive").sum())
    info["swapped"] = int(need)
    info["dropped_label_counts"] = {
        c.replace("label_", ""): int((sel.loc[drop_idx, c] == "positive").sum())
        for c in disease_cols
    }
    info["after_label_counts"] = {
        c.replace("label_", ""): int((out[c] == "positive").sum()) for c in disease_cols
    }
    return out, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/manifests/nih_shared_5000.csv")
    ap.add_argument("--pool", default="data/manifests/nih_full.csv",
                    help="full manifest used to find no_finding-positive donors")
    ap.add_argument("--output", default="data/manifests/nih_eval_1732.csv")
    ap.add_argument("--splits", nargs="+", default=EVAL_SPLITS)
    ap.add_argument("--no-finding-min", type=float, default=0.20,
                    help="minimum share of no_finding-positive rows (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-json", default="data/manifests/nih_eval_1732_report.json")
    args = ap.parse_args()

    src = ROOT / args.source
    if not src.exists():
        print(f"[error] source manifest not found: {src}")
        return 2
    df = pd.read_csv(src)
    if "split" not in df.columns:
        print(f"[error] {src.name} has no 'split' column; cannot select an evaluation set")
        return 2

    keep = df[df["split"].isin(args.splits)].copy()
    dropped = len(df) - len(keep)
    if keep.empty:
        print(f"[error] no rows matched splits {args.splits}; available: "
              f"{sorted(df['split'].unique())}")
        return 2

    nf_info = {"target": 0, "before": 0, "after": 0, "swapped": 0, "disabled": True}
    if args.no_finding_min > 0:
        pool_path = ROOT / args.pool
        if not pool_path.exists():
            print(f"[warn] pool {pool_path} not found; skipping the no_finding repair")
        else:
            pool = pd.read_csv(pool_path)
            if "label_no_finding" not in pool.columns:
                print("[warn] pool has no label_no_finding; skipping the repair")
            else:
                # The pool spans the whole 112k NIH manifest, but only ~6k images
                # are extracted on disk. Donors must exist locally, or the
                # manifest points at files that are not there.
                images_root = ROOT / "data" / "nih" / "images"
                if images_root.exists():
                    before = len(pool)
                    keep_mask = [ (images_root / rel).exists()
                                  for rel in pool["image_relpath"].astype(str) ]
                    pool = pool[pd.Series(keep_mask, index=pool.index)].copy()
                    print(f"pool restricted to extracted images: {before} -> {len(pool)}")
                # Prefer donors already inside the evaluation splits, so the
                # repair does not move rows across the split boundary at all.
                keep, nf_info = repair_no_finding(
                    keep, pool, args.no_finding_min, args.seed,
                    prefer_splits=args.splits)
                nf_info["disabled"] = False

    # Keep the source column order so the file is a drop-in subset.
    keep = keep[list(df.columns)].reset_index(drop=True)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(out, index=False)

    positives = {lb: int((keep[f"label_{lb}"] == "positive").sum()) for lb in STD_LABELS}
    body = {
        "source": src.name,
        "output": out.name,
        "rows": int(len(keep)),
        "dropped_rows": int(dropped),
        "dropped_reason": f"not in {args.splits}",
        "splits": keep["split"].value_counts().to_dict(),
        "unique_patients": int(keep["patient_id"].nunique()),
        "positives_per_label": positives,
        "no_finding_repair": nf_info,
    }

    missing = 0
    images_root = ROOT / "data" / "nih" / "images"
    if images_root.exists():
        missing = sum(1 for rel in keep["image_relpath"] if not (images_root / rel).exists())
        body["images_missing_on_disk"] = int(missing)

    if args.report_json:
        rp = ROOT / args.report_json
        rp.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {out} : {len(keep)} rows (dropped {dropped} non-evaluation rows)")
    print(f"  splits    : {body['splits']}")
    print(f"  patients  : {body['unique_patients']}")
    print(f"  positives : {positives}")
    if not nf_info.get("disabled"):
        print(f"  no_finding: {nf_info['before']} -> {nf_info['after']} "
              f"of {len(keep)} (target {nf_info['target']}, swapped {nf_info['swapped']})")
        if nf_info.get("warning"):
            print(f"  [warn] {nf_info['warning']}")
    if images_root.exists():
        print(f"  images    : {len(keep) - missing} present, {missing} missing")
    if positives["pneumonia"] < 100:
        print(f"[warn] only {positives['pneumonia']} pneumonia positives: bootstrap CIs "
              f"on that label will be wide")
    # Regression guard. `nih_shared_5000.csv` ships with label_no_finding constant,
    # which made every selective metric for that label undefined while the row
    # counts still looked healthy. If a future regeneration reintroduces a constant
    # label column, this output must not be promoted to an evaluation manifest.
    constant = [lb for lb, n in positives.items() if n == 0]
    if constant:
        print(f"[error] the built manifest has constant label column(s): {constant}")
        print("        their metrics would be undefined; do not use this for evaluation")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
