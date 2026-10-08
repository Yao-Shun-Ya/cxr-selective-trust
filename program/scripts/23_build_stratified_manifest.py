#!/usr/bin/env python
"""Stratified manifest builder for multi-VLM evaluation.

Why stratified, not random
--------------------------
NIH ChestX-ray14 is heavily long-tailed. A uniform 500-image draw can land a
handful of ``pneumothorax`` positives, and a macro AUROC averaged over labels
with n_positive = 3 is noise dressed as a result. The fix is not more images --
it is choosing images so that every label has enough positives to estimate
anything.

This builder also does the thing the current pilot cannot: it puts **every**
downstream experiment (shift benchmark, image reliance, prompt ablation,
selective prediction) on **one shared image list**, so the signals can be
combined on a real sample. Right now the shift set (100 images) and the
reliance set (300 images) overlap by zero test images, which is what blocks the
full-framework row in `19_accept_clarify_abstain.py`.

Design
------
* Splits are preserved exactly as they are in the source manifest: a patient is
  never moved between ``source_val`` and ``internal_test``.
* Within each split, images are chosen to hit a per-label positive target
  (``--min-positives``) for the disease labels, then filled with whatever else
  is needed to reach ``--size``.
* Multi-image patients are taken whole where possible, so patient-level
  clustering stays meaningful.
* The output records, per label and per split, the resulting positive counts so
  the paper can state the sampling honestly.

Usage
-----
::

    python scripts/23_build_stratified_manifest.py \
        --source data/manifests/nih_pilot_1000.csv \
        --output data/manifests/nih_shared_500.csv \
        --size 500 --min-positives 25 --seed 0

Add ``--extract`` to also unpack the chosen images from the NIH tarballs, and
``--report-json`` to dump the composition table for the paper's appendix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# labels the paper reports; no_finding is treated separately because it is
# near-complementary to the disease labels and would dominate any draw
DISEASE_LABELS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "pleural_effusion", "pneumonia", "pneumothorax",
]
ALL_REPORTED = [*DISEASE_LABELS, "no_finding"]


def label_matrix(df: pd.DataFrame, labels=DISEASE_LABELS) -> np.ndarray:
    cols = [f"label_{lb}" for lb in labels]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"manifest is missing label columns: {missing}")
    return np.column_stack([(df[c].astype(str) == "positive").to_numpy() for c in cols])


def allocate_per_split(counts: pd.Series, size: int) -> dict[str, int]:
    """Split the total budget across splits proportionally to available images."""
    total = int(counts.sum())
    if total == 0:
        return {}
    raw = {k: size * int(v) / total for k, v in counts.items()}
    alloc = {k: int(np.floor(v)) for k, v in raw.items()}
    # hand out the remainder to the largest fractional parts
    rest = size - sum(alloc.values())
    for k, _ in sorted(raw.items(), key=lambda kv: kv[1] - np.floor(kv[1]), reverse=True):
        if rest <= 0:
            break
        if alloc[k] < int(counts[k]):
            alloc[k] += 1
            rest -= 1
    return alloc


def allocate_per_label(pos_counts: pd.Series, min_pos: int) -> dict[str, int]:
    """Positive budget per disease label: at least min_pos, scaled by rarity.

    Rarer findings get a relative boost so that scarce labels are not the ones
    that end up under-powered.
    """
    if pos_counts.sum() == 0:
        return {k: 0 for k in pos_counts.index}
    inv = 1.0 / np.maximum(pos_counts.to_numpy(dtype=float), 1.0)
    weight = inv / inv.sum()
    target = {k: max(min_pos, int(round(min_pos * 2 * w * len(pos_counts))))
              for k, w in zip(pos_counts.index, weight)}
    return {k: min(v, int(pos_counts[k])) for k, v in target.items()}


def stratified_select(df: pd.DataFrame, size: int, min_pos: int, seed: int,
                      keep_patients_whole: bool = True) -> tuple[pd.DataFrame, dict]:
    """Pick `size` images so that every disease label reaches `min_pos` positives.

    Two failure modes are guarded against explicitly, because both are easy to
    hit and both silently ruin the evaluation set:

    1. **Greedy filling starves a split.** Taking "whatever is left" from the
       largest split leaves ``internal_test`` too small to estimate anything, so
       each split gets a proportional budget filled *only* from itself.
    2. **Rarity is not uniform.** Findings such as pneumonia are rare; images
       that are positive for them must be chosen deliberately, scarcest label
       first, or that label ends up with zero positives and its metric undefined.

    Labels whose target cannot be met are reported as unmet, never silently
    under-sampled.
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    Y = label_matrix(df)

    sizes = df["split"].value_counts()
    alloc = allocate_per_split(sizes, size)

    selected: list[int] = []
    unmet: dict[str, dict[str, int]] = {}
    global_have = {lb: 0 for lb in DISEASE_LABELS}

    for split in sizes.index:
        idx = df.index[df["split"] == split].to_numpy()
        Ys = Y[idx]
        budget = min(int(alloc.get(split, 0)), len(idx))
        chosen: set[int] = set()
        split_unmet: dict[str, int] = {}

        # positive targets, scarcest label first so it is not crowded out
        pos_counts = {lb: int(Ys[:, j].sum()) for j, lb in enumerate(DISEASE_LABELS)}
        for lb in sorted(DISEASE_LABELS, key=lambda x: pos_counts[x]):
            j = DISEASE_LABELS.index(lb)
            want_global = max(0, min_pos - global_have[lb])
            want_split = min(want_global, int(round(min_pos * len(idx) / len(df))))
            if want_split <= 0:
                continue
            have = int(Ys[sorted(chosen), j].sum()) if chosen else 0
            pool = [int(i) for i in np.flatnonzero(Ys[:, j] == 1) if i not in chosen]
            take = min(want_split - have, len(pool), budget - len(chosen))
            if take > 0:
                rng.shuffle(pool)
                chosen.update(pool[:take])
            got = int(Ys[sorted(chosen), j].sum()) if chosen else 0
            if got < want_split:
                split_unmet[lb] = want_split - got

        # fill the remaining budget from THIS split only, preferring images that
        # are positive for still-under-fed labels
        if len(chosen) < budget:
            rest = [i for i in range(len(idx)) if i not in chosen]

            def novelty(i, _Y=Ys, _chosen=chosen, _min=min_pos):
                if not _chosen:
                    return int(_Y[i].sum())
                have = _Y[sorted(_chosen), :].sum(axis=0)
                return int(np.sum((_Y[i] == 1) & (have < _min)))

            rest.sort(key=lambda i: (-novelty(i), i))
            chosen.update(rest[:budget - len(chosen)])

        for lb in DISEASE_LABELS:
            global_have[lb] += int(Ys[sorted(chosen), DISEASE_LABELS.index(lb)].sum())
        if split_unmet:
            unmet[split] = split_unmet
        selected.extend(int(idx[i]) for i in sorted(chosen))

    sel = df.loc[sorted(selected)].copy()

    # Keep multi-image patients whole, but only as far as the budget allows.
    # (Unbounded expansion is a real trap here: adding every sibling image of
    # every selected patient can pull in the entire manifest.)
    if keep_patients_whole and len(sel) < size:
        room = size - len(sel)
        pids = sel["patient_id"].astype(str).unique()
        extra = df[df["patient_id"].astype(str).isin(pids) & ~df.index.isin(sel.index)]
        if len(extra):
            # prefer patients that already contribute most images, deterministically
            extra = extra.sort_values(["patient_id", "image_id"])
            sel = pd.concat([sel, extra.head(room)]).sort_index()

    # hard budget guarantee
    if len(sel) > size:
        sel = sel.head(size)

    report = {
        "requested_size": int(size),
        "achieved_size": int(len(sel)),
        "min_positives_target": int(min_pos),
        "seed": int(seed),
        "by_split": sel["split"].value_counts().to_dict(),
        "by_label_positive": {
            lb: int((sel[f"label_{lb}"].astype(str) == "positive").sum())
            for lb in ALL_REPORTED if f"label_{lb}" in sel.columns
        },
        "by_label_positive_per_split": {
            lb: {sp: int((gg[f"label_{lb}"].astype(str) == "positive").sum())
                 for sp, gg in sel.groupby("split", sort=False)}
            for lb in ALL_REPORTED if f"label_{lb}" in sel.columns
        },
        "unique_patients": int(sel["patient_id"].astype(str).nunique()),
        "unmet": unmet,
    }
    return sel, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/manifests/nih_pilot_1000.csv")
    ap.add_argument("--output", default="data/manifests/nih_shared_500.csv")
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--min-positives", type=int, default=25,
                    help="target positives per disease label inside the selection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-json", default=None,
                    help="write the composition report (for the paper appendix)")
    ap.add_argument("--extract", action="store_true",
                    help="also extract the selected images from the NIH tarballs")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_absolute():
        src = ROOT / src
    df = pd.read_csv(src)
    if "patient_id" not in df.columns:
        print("[error] source manifest has no patient_id; cannot preserve splits by patient")
        return 1
    print(f"source: {src}  rows={len(df)}")
    print("  available by split:", df['split'].value_counts().to_dict())

    sel, report = stratified_select(df, args.size, args.min_positives, args.seed)

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out, index=False)

    print(f"\nwrote {out}  rows={len(sel)}")
    print(f"  by split : {report['by_split']}")
    print(f"  patients : {report['unique_patients']}")
    print("  positives per label (target >= %d):" % args.min_positives)
    for lb in DISEASE_LABELS:
        n = report["by_label_positive"].get(lb, 0)
        flag = "OK " if n >= args.min_positives else "LOW"
        print(f"    [{flag}] {lb:18s} {n:4d}   per split {report['by_label_positive_per_split'].get(lb)}")

    # A label can miss its target because the whole source pool is too small --
    # that is a data limitation, not a sampling bug, and it must be visible.
    pool = {lb: int((df[f"label_{lb}"].astype(str) == "positive").sum())
            for lb in DISEASE_LABELS if f"label_{lb}" in df.columns}
    short = {lb: pool.get(lb, 0) for lb in DISEASE_LABELS
             if report["by_label_positive"].get(lb, 0) < args.min_positives}
    if short:
        print("\n  !! targets NOT met -- available positives in the whole source pool:")
        for lb, avail in short.items():
            need = max(0, args.min_positives - report["by_label_positive"].get(lb, 0))
            verdict = ("source pool is exhausted; expand the pool or lower the target"
                       if avail < args.min_positives else "raise --size or the per-split budget")
            print(f"     {lb:18s} got {report['by_label_positive'].get(lb,0):3d}, "
                  f"need {need:3d} more, pool has {avail:3d}  -> {verdict}")
    if report.get("unmet"):
        print(f"\n  per-split shortfalls: {json.dumps(report['unmet'])}")

    if args.report_json:
        rp = Path(args.report_json)
        if not rp.is_absolute():
            rp = ROOT / rp
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {rp}")

    if args.extract:
        import subprocess
        cmd = [sys.executable, str(ROOT / "scripts" / "02_extract_nih_images.py"),
               "--manifest", str(out)]
        print("\nextracting:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
