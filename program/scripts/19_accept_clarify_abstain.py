#!/usr/bin/env python
"""Shift-aware selective trust: composite risk + cost-sensitive three-way policy.

What problem this solves
------------------------
The paper claims a framework integrating **calibration, uncertainty, shift
score, and image reliance**. Until now the code used confidence alone, and
Figures 1/9 were conceptual. This script makes the framework a measurable
object and answers the only question a reviewer will actually ask:

    Does adding shift and image-reliance evidence to confidence improve the
    decision, not just the ranking?

Signals (all computable without ground truth)
---------------------------------------------
``uncertainty``
    ``1 - mean|2p-1|`` over calibrated probabilities: how far the model sits
    from the decision boundary.
``image_reliance``
    Mean |ΔP| between the original image and uninformative ones (blank / blur /
    shuffle), from ``06_image_reliance.py`` output. This is the Visual Reliance
    Signal: a model that truly reads the image must change its answer.
``shift_score``
    Spread of P across image-quality conditions, from ``08_shift_benchmark.py``
    output.
``disagreement``
    Cross-model disagreement at the same image, when a second model's
    predictions are supplied.

Method
------
1. Temperature-scale per finding on ``source_val``; apply to the target split.
2. Build per-image signal rows from whichever sources are supplied.
3. Fit a **calibrated** logistic map signals -> P(error) on ``source_val`` only.
   Weights are fitted, not hand-picked, so they cannot be accused of being tuned
   on the test split.
4. Compare signal subsets head-to-head on the target split: AUROC for detecting
   errors, and ``coverage@risk`` -- because a better AUROC with no gain in
   usable coverage is exactly the situation we are in, and hiding that would be
   the easiest way to overclaim.
5. Fit the three-way ``accept / clarify / abstain`` thresholds on ``source_val``
   by minimizing a **declared** cost model, then evaluate on the target split.
6. Bootstrap all headline numbers by patient cluster.

Usage
-----
::

    python scripts/19_accept_clarify_abstain.py \
        --predictions outputs/overnight/medgemma_4b_it_1000/predictions.jsonl \
        --reliance    outputs/reliance/medgemma_300 \
        --shift       outputs/shift_benchmark/medgemma_4b_it_100 \
        --manifest    data/manifests/nih_pilot_1000.csv \
        --output-dir  outputs/trust_v1/medgemma_4b_it \
        --n-boot 1000

Every source is optional; the script reports which ones were actually used and
restricts fitting to images where all required signals exist (no imputation).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.evaluation.trust import (  # noqa: E402
    DEFAULT_LABELS,
    CostModel,
    LogisticErrorModel,
    SIGNAL_NAMES,
    _expected_cost,
    evaluate_policy,
    fit_three_way_policy,
    image_reliance_signal,
    shift_score,
    status_flip_signal,
    uncertainty_signal,
)

# signal subsets compared head-to-head; the last one is the paper's framework
SUBSETS = {
    "confidence_only": (),
    "plus_image_reliance": ("image_reliance",),
    "plus_shift_score": ("shift_score",),
    "plus_reliance_and_shift": ("image_reliance", "shift_score"),
    "full_framework": ("image_reliance", "shift_score", "disagreement"),
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_predictions(path: Path) -> pd.DataFrame:
    """One row per image: ground truth vector, statuses, raw probabilities."""
    rows = []
    for obj in _read_jsonl(path):
        gt = obj.get("ground_truth") or {}
        findings = obj.get("findings") or {}
        if not findings:
            continue
        y, p, st = [], [], []
        usable = True
        for lb in DEFAULT_LABELS:
            g = gt.get(f"label_{lb}")
            if g not in {"positive", "negative"}:
                usable = False
                break
            y.append(1.0 if g == "positive" else 0.0)
            f = findings.get(lb) or {}
            pp = f.get("probability_positive")
            p.append(float(pp) if isinstance(pp, (int, float)) and not isinstance(pp, bool) else np.nan)
            st.append(f.get("status"))
        if not usable:
            continue
        pred = np.asarray([1.0 if s == "positive" else 0.0 for s in st], dtype=float)
        yv = np.asarray(y, dtype=float)
        rows.append({
            "image_id": obj.get("image_id"),
            "split": obj.get("split"),
            "y": yv,
            "p": np.asarray(p, dtype=float),
            "status": st,
            "error_all": float((pred != yv).any()),          # image is wrong if ANY finding is wrong
            "error_mean": float((pred != yv).mean()),
        })
    return pd.DataFrame(rows)


def load_reliance(dir_path: Path, labels=DEFAULT_LABELS) -> pd.DataFrame:
    """Per-image image-reliance signals from 06_image_reliance.py output."""
    recs = {}
    for p in sorted(dir_path.glob("*/image_reliance_predictions.jsonl")) + \
             sorted(dir_path.glob("image_reliance_predictions.jsonl")):
        for obj in _read_jsonl(p):
            recs[(str(obj.get("image_id")), str(obj.get("condition")))] = obj.get("parsed") or {}
    if not recs:
        return pd.DataFrame()
    rows = []
    for iid in sorted({k[0] for k in recs}):
        orig = recs.get((iid, "original"))
        if not orig or not orig.get("parse_success", True):
            continue
        others = [recs[(iid, c)] for c in ("blank", "blur", "shuffle") if (iid, c) in recs]
        rows.append({
            "image_id": iid,
            "image_reliance": image_reliance_signal(orig, others, labels),
            "status_flip": status_flip_signal(orig, others, labels),
        })
    return pd.DataFrame(rows)


def load_shift(dir_path: Path, labels=DEFAULT_LABELS) -> pd.DataFrame:
    """Per-image shift score from 08_shift_benchmark.py output."""
    files = sorted(dir_path.glob("shift_predictions.jsonl")) + \
            sorted(dir_path.glob("*/shift_predictions.jsonl"))
    by_image: dict[str, list[np.ndarray]] = {}
    for p in files:
        for obj in _read_jsonl(p):
            parsed = obj.get("parsed") or {}
            if not parsed.get("parse_success", True):
                continue
            findings = parsed.get("findings") or {}
            vec = np.asarray([
                (findings.get(lb) or {}).get("probability_positive")
                if isinstance((findings.get(lb) or {}).get("probability_positive"), (int, float))
                else np.nan
                for lb in labels
            ], dtype=float)
            by_image.setdefault(str(obj.get("image_id")), []).append(vec)
    rows = []
    for iid, vecs in by_image.items():
        s = shift_score(np.vstack(vecs))
        rows.append({"image_id": iid, "shift_score": s["spread"],
                     "shift_valley": s["valley"], "n_shift_conditions": len(vecs)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    m = np.isfinite(p) & np.isfinite(y)
    y, p = y[m], p[m]
    if len(y) < 5 or len(np.unique(y)) < 2:
        return 1.0
    z = _logit(p)

    def nll(log_t):
        q = np.clip(_sigmoid(z / np.exp(log_t)), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(50.0)), method="bounded")
    return float(np.exp(res.x)) if res.success else 1.0


def calibrate(df: pd.DataFrame, src_split: str, min_pos: int = 5) -> tuple[pd.DataFrame, dict]:
    """Per-finding temperature fitted on src_split; applied to every row."""
    src = df[df["split"] == src_split]
    temps = {}
    for i, lb in enumerate(DEFAULT_LABELS):
        y = np.asarray([r[i] for r in src["y"]])
        p = np.asarray([r[i] for r in src["p"]])
        temps[lb] = fit_temperature(y, p) if np.nansum(y) >= min_pos else 1.0
    tvec = np.asarray([temps[lb] for lb in DEFAULT_LABELS])

    out = df.copy()
    out["p_cal"] = [
        np.asarray([_sigmoid(_logit(pp) / t) if np.isfinite(pp) else np.nan
                    for pp, t in zip(row, tvec)], dtype=float)
        for row in df["p"]
    ]
    out["uncertainty"] = [uncertainty_signal({"findings": _pack(pc)}, DEFAULT_LABELS)
                          for pc in out["p_cal"]]
    return out, temps


def _pack(pvec: np.ndarray) -> dict:
    return {lb: {"probability_positive": (float(v) if np.isfinite(v) else None)}
            for lb, v in zip(DEFAULT_LABELS, pvec)}


def _confidence_from_row(statuses, p_cal: np.ndarray) -> float:
    """Mean confidence in the decisions the model actually made.

    Mirrors the convention documented in the paper's Methods so that the
    confidence-only baseline is exactly the score the paper currently uses:
    ``conf = p`` for a positive call, ``1 - p`` for a negative call, and 0.5
    when the model declined to decide. Uses the *calibrated* probability, which
    is what a deployed system would have.
    """
    vals = []
    for st, pc in zip(statuses, p_cal):
        if not np.isfinite(pc):
            continue
        if st == "positive":
            vals.append(float(pc))
        elif st == "negative":
            vals.append(float(1.0 - pc))
        else:
            vals.append(0.5)
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# unit of analysis
# --------------------------------------------------------------------------- #
def explode_to_findings(df: pd.DataFrame, labels=DEFAULT_LABELS) -> pd.DataFrame:
    """One row per (image, finding) -- the unit a single prediction lives at.

    Why this exists: at image level, "correct" means *every* one of eight
    non-independent findings is right, so MedGemma's image error rate is 94.7%
    and no cost model can justify accepting anything. That is a true statement
    about whole-report automation, but it is not a fair test of whether the
    framework works -- it is a test of whether one specific aggregation is
    hopeless. Finding level is where selective prediction is conventionally
    evaluated, so both are reported.
    """
    rows = []
    for _, r in df.iterrows():
        y = np.asarray(r["y"], float)
        p = np.asarray(r["p"], float)
        pc = np.asarray(r["p_cal"], float)
        for i, lb in enumerate(labels):
            st = r["status"][i]
            if not np.isfinite(y[i]):
                continue
            if st == "positive":
                pred, conf = 1.0, (pc[i] if np.isfinite(pc[i]) else np.nan)
            elif st == "negative":
                pred, conf = 0.0, (1.0 - pc[i] if np.isfinite(pc[i]) else np.nan)
            else:
                pred, conf = 0.0, 0.5      # a declined call is treated as negative, at 0.5
            rows.append({
                "image_id": r["image_id"], "label": lb,
                "patient_id": r.get("patient_id", r["image_id"]),
                "split": r["split"],
                "y": float(y[i]),
                "error_all": float(pred != y[i]),
                "confidence": float(conf) if np.isfinite(conf) else np.nan,
                "image_reliance": r.get("image_reliance", np.nan),
                "shift_score": r.get("shift_score", np.nan),
                "disagreement": r.get("disagreement", np.nan),
                # per-finding versions of the reliability signals
                "status_flip": r.get("status_flip", np.nan),
                "shift_valley": r.get("shift_valley", np.nan),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def cluster_indices(ids: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(ids, kind="stable")
    s = ids[order]
    b = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    return [order[b[i]:b[i + 1]] for i in range(len(b) - 1)]


def _auc(err, risk):
    err = np.asarray(err, float); risk = np.asarray(risk, float)
    m = np.isfinite(err) & np.isfinite(risk)
    err, risk = err[m], risk[m]
    if len(err) < 2 or len(np.unique(err)) < 2:
        return float("nan")
    order = np.argsort(risk)
    ranks = np.empty(len(risk), float)
    ranks[order] = np.arange(1, len(risk) + 1)
    n1 = err.sum(); n0 = len(err) - n1
    return float((ranks[err == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _coverage_at_risk(err, risk, budget):
    err = np.asarray(err, float); risk = np.asarray(risk, float)
    m = np.isfinite(err) & np.isfinite(risk)
    err, risk = err[m], risk[m]
    if len(err) == 0:
        return float("nan")
    o = np.argsort(risk, kind="stable")
    e = err[o]
    cum = np.cumsum(e) / np.arange(1, len(e) + 1)
    ok = np.where(cum <= budget + 1e-12)[0]
    return float((ok[-1] + 1) / len(e)) if len(ok) else 0.0


def percentile_ci(vals, level=0.95):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"),) * 3
    return (float(np.mean(v)),
            float(np.percentile(v, (1 - level) / 2 * 100)),
            float(np.percentile(v, (1 + level) / 2 * 100)))


def bootstrap_metrics(err, risk, clusters, n_boot, rng, budgets=(0.05, 0.10, 0.20)):
    rows = []
    for idx in _iter_boot(clusters, n_boot, rng):
        e, r = err[idx], risk[idx]
        row = {"auc": _auc(e, r)}
        for b in budgets:
            row[f"cov@{int(b*100)}"] = _coverage_at_risk(e, r, b)
        rows.append(row)
    return pd.DataFrame(rows)


def _iter_boot(clusters, n_boot, rng):
    n = len(clusters)
    for _ in range(n_boot):
        yield np.concatenate([clusters[i] for i in rng.integers(0, n, size=n)])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--reliance", default=None, help="dir with image_reliance_predictions.jsonl")
    ap.add_argument("--shift", default=None, help="dir with shift_predictions.jsonl")
    ap.add_argument("--disagreement-predictions", default=None,
                    help="second model's predictions.jsonl; adds a disagreement signal")
    ap.add_argument("--manifest", default="data/manifests/nih_pilot_1000.csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--source-val-split", default="source_val")
    ap.add_argument("--test-split", default="internal_test")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c-accept-error", type=float, default=10.0)
    ap.add_argument("--c-clarify", type=float, default=1.0)
    ap.add_argument("--c-abstain", type=float, default=2.0)
    ap.add_argument("--risk-budgets", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--unit", nargs="+", default=["finding", "image"],
                    choices=["finding", "image"],
                    help="unit of analysis; 'finding' = per (image,finding) prediction, "
                         "'image' = whole report correct only if every finding is correct")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- load + merge ---------------------------------------------------
    df = load_predictions(Path(args.predictions))
    if df.empty:
        print("[error] no usable predictions")
        return 1
    sources = {"predictions": str(args.predictions)}

    # patient clustering
    man_path = Path(args.manifest)
    if not man_path.is_absolute():
        man_path = ROOT / man_path
    id_col = "image_id"
    if man_path.exists():
        man = pd.read_csv(man_path, usecols=lambda c: c in {"image_id", "patient_id"})
        if "patient_id" in man.columns:
            df = df.merge(man.drop_duplicates("image_id")[["image_id", "patient_id"]],
                          on="image_id", how="left")
            df["patient_id"] = df["patient_id"].fillna(df["image_id"]).astype(str)
            id_col = "patient_id"
    if "patient_id" not in df.columns:
        df["patient_id"] = df["image_id"].astype(str)

    if args.reliance:
        rl = load_reliance(Path(args.reliance))
        if not rl.empty:
            df = df.merge(rl, on="image_id", how="left")
            sources["reliance"] = args.reliance
            print(f"[reliance] {rl['image_reliance'].notna().sum()} images with a VRS value")

    if args.shift:
        sh = load_shift(Path(args.shift))
        if not sh.empty:
            df = df.merge(sh, on="image_id", how="left")
            sources["shift"] = args.shift
            print(f"[shift] {sh['shift_score'].notna().sum()} images with a shift score")

    if args.disagreement_predictions:
        other = load_predictions(Path(args.disagreement_predictions))
        if not other.empty:
            omap = {r["image_id"]: r for _, r in other.iterrows()}
            dis = []
            for _, r in df.iterrows():
                o = omap.get(r["image_id"])
                if o is None:
                    dis.append(np.nan)
                    continue
                a = np.asarray([1.0 if s == "positive" else 0.0 for s in r["status"]])
                b = np.asarray([1.0 if s == "positive" else 0.0 for s in o["status"]])
                dis.append(float((a != b).mean()))
            df["disagreement"] = dis
            sources["disagreement"] = args.disagreement_predictions
            print(f"[disagreement] {np.isfinite(df['disagreement']).sum()} images")

    for s in SIGNAL_NAMES:
        if s not in df.columns:
            df[s] = np.nan

    # ---- calibration ----------------------------------------------------
    # Confidence is defined AFTER temperature scaling, from the calibrated
    # probability of whichever decision the model actually made:
    #     confidence = p_cal if predicted positive else 1 - p_cal
    # and "uncertain" / missing statuses get 0.5 (the existing convention).
    df, temps = calibrate(df, args.source_val_split)
    df["confidence"] = [
        _confidence_from_row(st, pc) for st, pc in zip(df["status"], df["p_cal"])
    ]
    pd.DataFrame([{"label": k, "temperature": v} for k, v in temps.items()]).to_csv(
        out_dir / "temperature_by_label.csv", index=False)

    src = df[df["split"] == args.source_val_split].copy()
    tgt = df[df["split"] == args.test_split].copy()
    if src.empty or tgt.empty:
        print(f"[error] need both splits: {args.source_val_split}={len(src)}, "
              f"{args.test_split}={len(tgt)}")
        return 1

    # ---- per-unit analysis ----------------------------------------------
    cm = CostModel(c_accept_error=args.c_accept_error,
                   c_clarify=args.c_clarify, c_abstain=args.c_abstain)

    units = {}
    for unit in args.unit:
        data = df if unit == "image" else explode_to_findings(df)
        u_src = data[data["split"] == args.source_val_split].copy()
        u_tgt = data[data["split"] == args.test_split].copy()
        if u_src.empty or u_tgt.empty or len(u_src) < 20 or len(u_tgt) < 20:
            print(f"[skip] unit={unit}: source_val={len(u_src)} test={len(u_tgt)}")
            continue

        print(f"\n=== unit={unit}  "
              f"(source_val n={len(u_src)}, test n={len(u_tgt)}, "
              f"base error={u_tgt['error_all'].mean():.3f}) ===")
        print("--- signal subset comparison (fitted on source_val, evaluated on test) ---")
        comparison, boots = [], {}
        for name, extra in SUBSETS.items():
            cols = ["confidence", *extra]
            s_ok = u_src.dropna(subset=cols)
            t_ok = u_tgt.dropna(subset=cols)
            if len(s_ok) < 20 or len(t_ok) < 20:
                comparison.append({"unit": unit, "subset": name,
                                   "n_source_val": int(len(s_ok)), "n_test": int(len(t_ok)),
                                   "status": "insufficient_rows"})
                print(f"  {name:26s} SKIPPED (source_val={len(s_ok)}, test={len(t_ok)})")
                continue
            Xs = s_ok[cols].to_numpy(float); ys = s_ok["error_all"].to_numpy(float)
            Xt = t_ok[cols].to_numpy(float); yt = t_ok["error_all"].to_numpy(float)
            model = LogisticErrorModel().fit(Xs, ys, feature_names=cols)
            risk_t = model.predict_proba(Xt)

            row = {
                "unit": unit, "subset": name, "features": "+".join(cols),
                "n_source_val": int(len(s_ok)), "n_test": int(len(t_ok)),
                "n_dropped_source_val": int(len(u_src) - len(s_ok)),
                "n_dropped_test": int(len(u_tgt) - len(t_ok)),
                "test_base_error": float(yt.mean()),
                "auc": _auc(yt, risk_t),
                "model_coefficients": model.coefficients(),
            }
            for b in args.risk_budgets:
                row[f"coverage_at_{int(b*100)}pct_risk"] = _coverage_at_risk(yt, risk_t, b)
            comparison.append(row)

            cl = cluster_indices(t_ok["patient_id"].astype(str).to_numpy())
            bs = bootstrap_metrics(yt, risk_t, cl, args.n_boot, rng, tuple(args.risk_budgets))
            boots[name] = bs
            row["auc_ci"] = dict(zip(("mean", "lo", "hi"), percentile_ci(bs["auc"])))
            for b in args.risk_budgets:
                row[f"coverage_at_{int(b*100)}pct_risk_ci"] = dict(
                    zip(("mean", "lo", "hi"), percentile_ci(bs[f"cov@{int(b*100)}"])))
            print(f"  {name:26s} n_test={len(t_ok):5d}  AUROC={row['auc']:.3f}"
                  f"  cov@5%={row['coverage_at_5pct_risk']:.3f}"
                  f"  cov@10%={row['coverage_at_10pct_risk']:.3f}"
                  f"  cov@20%={row['coverage_at_20pct_risk']:.3f}")

        print(f"--- three-way policy, cost model: {cm.describe()} ---")
        policy_rows = []
        for name, extra in SUBSETS.items():
            cols = ["confidence", *extra]
            s_ok = u_src.dropna(subset=cols); t_ok = u_tgt.dropna(subset=cols)
            if len(s_ok) < 20 or len(t_ok) < 20:
                continue
            model = LogisticErrorModel().fit(s_ok[cols].to_numpy(float),
                                             s_ok["error_all"].to_numpy(float),
                                             feature_names=cols)
            risk_s = model.predict_proba(s_ok[cols].to_numpy(float))
            risk_t = model.predict_proba(t_ok[cols].to_numpy(float))
            pol = fit_three_way_policy(risk_s, s_ok["error_all"].to_numpy(float), cm,
                                       fitted_on=args.source_val_split)
            ev = evaluate_policy(pol.decide(risk_t), t_ok["error_all"].to_numpy(float),
                                 cm, risk_t)
            ev.update({"unit": unit, "subset": name, "policy": "three_way",
                       "tau_accept": pol.tau_accept, "tau_abstain": pol.tau_abstain,
                       "source_val_expected_cost": pol.expected_cost})
            policy_rows.append(ev)
            print(f"  {name:26s} accept={ev['accept_rate']:.3f} "
                  f"clarify={ev['clarify_rate']:.3f} abstain={ev['abstain_rate']:.3f}  "
                  f"accepted_err={ev['accepted_error_rate']:.3f} "
                  f"(base {ev['overall_error_rate']:.3f})")

        ref = {
            "accept_all": evaluate_policy(np.zeros(len(u_tgt), dtype=int),
                                          u_tgt["error_all"].to_numpy(float), cm),
            "abstain_all": evaluate_policy(np.full(len(u_tgt), 2, dtype=int),
                                           u_tgt["error_all"].to_numpy(float), cm),
        }

        # ---- matched-sample comparison -----------------------------------
        # Subsets with extra signals have FEWER usable rows, so comparing them
        # against confidence_only on all rows would confound "the signal helps"
        # with "the easy cases happen to carry the signal". Refit every subset
        # on the intersection of rows where BOTH it and the baseline have data,
        # then paired-bootstrap the difference over the same patients.
        #
        # Note on the pilot data: the shift signal exists only for 100 images and
        # the image-reliance signal for 300 different images, so their three-way
        # intersection is empty. That is why comparisons are pairwise -- and why
        # scaling both onto one shared manifest is a prerequisite for the
        # full-framework row, not a nice-to-have.
        matched = {"comparisons": []}
        for name, extra in SUBSETS.items():
            if name == "confidence_only" or not extra:
                continue
            cols = ["confidence", *extra]
            base_cols = ["confidence"]
            need = sorted(set(cols) | set(base_cols))
            m_src = u_src.dropna(subset=need)
            m_tgt = u_tgt.dropna(subset=need)
            if len(m_src) < 20 or len(m_tgt) < 20:
                continue
            print(f"--- matched comparison on {len(m_tgt)} test rows "
                  f"(source_val {len(m_src)}): confidence vs confidence+{'+'.join(extra)} ---")
            yt_m = m_tgt["error_all"].to_numpy(float)
            mdl_full = LogisticErrorModel().fit(
                m_src[cols].to_numpy(float), m_src["error_all"].to_numpy(float),
                feature_names=cols)
            mdl_base = LogisticErrorModel().fit(
                m_src[base_cols].to_numpy(float), m_src["error_all"].to_numpy(float),
                feature_names=base_cols)
            r_full = mdl_full.predict_proba(m_tgt[cols].to_numpy(float))
            r_base = mdl_base.predict_proba(m_tgt[base_cols].to_numpy(float))

            m_clusters = cluster_indices(m_tgt["patient_id"].astype(str).to_numpy())
            boot_idx = list(_iter_boot(m_clusters, args.n_boot, rng))
            d_auc = [_auc(yt_m[i], r_full[i]) - _auc(yt_m[i], r_base[i]) for i in boot_idx]
            d_cov = {b: [_coverage_at_risk(yt_m[i], r_full[i], b)
                         - _coverage_at_risk(yt_m[i], r_base[i], b) for i in boot_idx]
                     for b in args.risk_budgets}

            rec = {
                "added_signal": "+".join(extra),
                "n_test": int(len(m_tgt)), "n_source_val": int(len(m_src)),
                "base_error": float(yt_m.mean()),
                "auc_confidence_only": _auc(yt_m, r_base),
                "auc_with_signal": _auc(yt_m, r_full),
                "coefficients_with_signal": mdl_full.coefficients(),
            }
            mean, lo, hi = percentile_ci(d_auc)
            rec["delta_auc"] = {"mean": mean, "lo": lo, "hi": hi,
                                "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi)
                                                      and (lo > 0 or hi < 0))}
            for b in args.risk_budgets:
                mean, lo, hi = percentile_ci(d_cov[b])
                rec[f"delta_coverage_at_{int(b*100)}pct_risk"] = {
                    "mean": mean, "lo": lo, "hi": hi,
                    "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi)
                                          and (lo > 0 or hi < 0))}
                rec[f"coverage_at_{int(b*100)}pct_risk_confidence_only"] = _coverage_at_risk(
                    yt_m, r_base, b)
                rec[f"coverage_at_{int(b*100)}pct_risk_with_signal"] = _coverage_at_risk(
                    yt_m, r_full, b)
            matched["comparisons"].append(rec)
            print(f"    AUROC      {rec['auc_confidence_only']:.3f} -> {rec['auc_with_signal']:.3f}"
                  f"   Δ={rec['delta_auc']['mean']:+.3f} "
                  f"[{rec['delta_auc']['lo']:+.3f},{rec['delta_auc']['hi']:+.3f}]"
                  f"{'  *excludes 0*' if rec['delta_auc']['excludes_zero'] else ''}")
            for b in args.risk_budgets:
                d = rec[f"delta_coverage_at_{int(b*100)}pct_risk"]
                print(f"    cov@{int(b*100)}% risk  "
                      f"{rec[f'coverage_at_{int(b*100)}pct_risk_confidence_only']:.3f}"
                      f" -> {rec[f'coverage_at_{int(b*100)}pct_risk_with_signal']:.3f}"
                      f"   Δ={d['mean']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]"
                      f"{'  *excludes 0*' if d['excludes_zero'] else ''}")

        units[unit] = {
            "comparison": comparison,
            "policy": policy_rows,
            "reference": ref,
            "matched_sample": matched,
            "n_source_val": int(len(u_src)),
            "n_test": int(len(u_tgt)),
            "base_error_test": float(u_tgt["error_all"].mean()),
        }
        pd.DataFrame(comparison).to_csv(out_dir / f"signal_subset_comparison_{unit}.csv",
                                        index=False)
        pd.DataFrame(policy_rows).to_csv(out_dir / f"three_way_policy_results_{unit}.csv",
                                         index=False)
        if matched.get("comparisons"):
            flat = []
            for rec in matched["comparisons"]:
                row = {k: v for k, v in rec.items() if not isinstance(v, dict)}
                for k, v in rec.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            row[f"{k}__{kk}"] = vv
                flat.append(row)
            pd.DataFrame(flat).to_csv(
                out_dir / f"matched_paired_differences_{unit}.csv", index=False)

    # ---- persist --------------------------------------------------------
    # drop the pre-unit rollup written by earlier versions; per-unit files are authoritative
    stale = out_dir / "signal_subset_comparison.csv"
    if stale.exists():
        stale.unlink()

    summary = {
        "sources": sources,
        "units": units,
        "calibration_temperatures": {k: float(v) for k, v in temps.items()},
        "cost_model": cm.describe(),
        "risk_budgets": args.risk_budgets,
        "interpretation_notes": [
            "unit=finding is one row per (image, finding) prediction.",
            "unit=image counts a report correct only if every finding is correct; "
            "the 8 labels are not independent (no_finding is near-complementary to "
            "the 7 disease labels), so image-level error is a report-correctness rate.",
            "All signals are ground-truth-free; ground truth is used only to fit the "
            "logistic map and the policy thresholds, both on source_val.",
            "Rows missing any required signal are dropped, never imputed; the number "
            "dropped is reported per subset.",
            "CAUTION -- unmatched subset comparison is not a fair head-to-head: subsets "
            "with extra signals have fewer usable rows, so a higher AUROC there can be "
            "a sample-composition effect. Use matched_sample.comparisons, which refits "
            "both the baseline and the augmented model on the same rows and paired-"
            "bootstraps the difference over the same patients.",
            "On the current pilot the shift signal covers 100 images and the image-"
            "reliance signal 300 different images, so the three-way intersection is "
            "empty; full-framework rows require scaling both onto one shared manifest.",
        ],
        "provenance": {
            "script": "scripts/19_accept_clarify_abstain.py",
            "predictions": str(args.predictions),
            "n_boot": args.n_boot, "seed": args.seed,
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
        },
    }
    (out_dir / "trust_v1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    df.drop(columns=["y", "p", "p_cal"], errors="ignore").to_csv(
        out_dir / "image_signals.csv", index=False)

    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
