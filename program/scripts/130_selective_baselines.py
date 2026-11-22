"""Named selective-prediction baselines from the frozen finding-level items.

The paper's confidence score is status-aware: calibrated p for a positive call,
1-p for a negative call, 0.5 for a declined call. A reviewer can ask whether the
negative coverage result is an artefact of that particular score. This script
recomputes the same risk-coverage metrics under two probability-only baselines:

* Bernoulli entropy of the calibrated probability (higher = less confident);
* probability margin |p - 0.5| (higher = more confident).

Both are computed on the same 8,432 finding items and use stable sorting, which
is the convention that reproduces Table 3's confidence coverage exactly.

    python scripts/130_selective_baselines.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_selective_baselines"
MODELS = ["medgemma_4b_it", "medgemma_15_4b_it", "internvl3_5_8b",
          "qwen3_vl_8b_instruct", "qwen25_vl_7b_instruct"]


def _cov_at(error: np.ndarray, score: np.ndarray, budget: float) -> float:
    """Coverage at a risk budget; stable sort matches the paper's pipeline."""
    order = np.argsort(-score, kind="stable")
    err = error[order]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    ok = np.where(risk <= budget + 1e-12)[0]
    return float((ok[-1] + 1) / len(err)) if len(ok) else 0.0


def _metrics(error: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    return (float(roc_auc_score(error, -score)),
            _cov_at(error, score, 0.05),
            _cov_at(error, score, 0.10))


def main() -> None:
    rows = []
    for model in MODELS:
        path = ROOT / "outputs" / "selective_v2" / model / "risk_items_finding.csv"
        df = pd.read_csv(path)
        p = df["p_calibrated"].clip(1e-6, 1 - 1e-6).to_numpy()
        err = df["error"].to_numpy()
        ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        margin = np.abs(p - 0.5)
        for signal, score in (("confidence", df["conf"].to_numpy()),
                              ("entropy", -ent),
                              ("margin", margin)):
            auroc, cov5, cov10 = _metrics(err, score)
            rows.append({"model": model, "signal": signal, "auroc": auroc,
                         "cov5": cov5, "cov10": cov10, "n": len(df)})
    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "selective_baselines.csv", index=False)
    print(out.to_string(index=False))
    print(f"wrote {OUT/'selective_baselines.csv'}")


if __name__ == "__main__":
    main()
