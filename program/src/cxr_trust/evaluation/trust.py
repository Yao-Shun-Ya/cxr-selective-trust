"""Shift-aware selective trust: composite risk, calibration, and decision policy.

The gap this module closes
--------------------------
`papers/main_v2/sections/03_methods.tex` claims a framework that integrates
**calibration, uncertainty, shift score, and image reliance**, but the code it
describes computes risk from *confidence alone*::

    confidence = P(positive) if predicted positive else 1 - P(positive); risk = 1 - confidence

The other three terms never enter the decision. Everything below exists so the
paper's claim matches the code.

Three deliberate design rules
-----------------------------
1. **Nothing may use ground truth at inference time.** Every signal here is
   computable from the input image and the model alone, so the score can be
   used on unlabelled data. Ground truth is used only to *fit* the mapping from
   signals to error probability, and that fitting happens on ``source_val``.

2. **The mapping to a probability is calibrated, not hand-weighted.** A logistic
   map from signals to P(error) is fitted on ``source_val``. Hand-picked
   weights would be a hyperparameter with no principled justification, and would
   invite exactly the "you tuned it on the test set" objection.

3. **Decision thresholds come from an explicit cost model.** The three-way
   policy is derived from declared clinical costs (missing a finding costs more
   than deferring one), not from a threshold chosen to make the curve look good.
   Cost parameters are declared before evaluation and reported alongside results.

The honest position
-------------------
On the current pilot data the composite score improves error *ranking*
(image-level AUROC 0.814 -> 0.853 on n=61) but does **not** yet reach a usable
``coverage@5% risk``. This module exists partly to measure that gap rigorously
and partly to make the framework a real, falsifiable object rather than a
diagram.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# --------------------------------------------------------------------------- #
# signal names, in the order they appear in the design matrix
# --------------------------------------------------------------------------- #
SIGNAL_NAMES = ("uncertainty", "image_reliance", "shift_score", "disagreement")

DEFAULT_LABELS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "pleural_effusion", "pneumonia", "pneumothorax", "no_finding",
]


# --------------------------------------------------------------------------- #
# per-image, ground-truth-free signals
# --------------------------------------------------------------------------- #
def probability_vector(parsed: dict, labels: Sequence[str] = DEFAULT_LABELS) -> np.ndarray:
    """P(positive) per label, NaN where the model did not supply a probability."""
    findings = (parsed or {}).get("findings") or {}
    out = []
    for lb in labels:
        v = (findings.get(lb) or {}).get("probability_positive")
        out.append(float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan)
    return np.asarray(out, dtype=float)


def status_vector(parsed: dict, labels: Sequence[str] = DEFAULT_LABELS) -> np.ndarray:
    """1.0 positive, 0.0 negative, NaN for uncertain / not_mentioned / absent."""
    findings = (parsed or {}).get("findings") or {}
    out = []
    for lb in labels:
        st = (findings.get(lb) or {}).get("status")
        out.append(1.0 if st == "positive" else 0.0 if st == "negative" else np.nan)
    return np.asarray(out, dtype=float)


def image_reliance_signal(original_parsed: dict,
                          perturbed_parsed: Sequence[dict],
                          labels: Sequence[str] = DEFAULT_LABELS) -> float:
    """Visual Reliance Signal.

    Mean absolute change in P(positive) when the image is replaced by an
    uninformative one (blank / heavy blur / shuffled other patient). A model that
    truly reads the image must change its answer; a model running on a language
    prior will not. **Requires no ground truth**, so it is available at
    deployment for the price of one extra forward pass.

    Returns NaN when no perturbed condition supplied usable probabilities.
    """
    base = probability_vector(original_parsed, labels)
    if np.all(np.isnan(base)):
        return float("nan")
    deltas = []
    for p in perturbed_parsed:
        other = probability_vector(p, labels)
        if np.all(np.isnan(other)):
            continue
        deltas.append(np.nanmean(np.abs(base - other)))
    if not deltas:
        return float("nan")
    return float(np.mean(deltas))


def status_flip_signal(original_parsed: dict,
                       perturbed_parsed: Sequence[dict],
                       labels: Sequence[str] = DEFAULT_LABELS) -> float:
    """Fraction of labels whose positive/negative status flips under perturbation.

    Complements the probability-level VRS: a model can hold its probabilities
    steady yet still flip a decision, or move probabilities a lot without ever
    crossing the decision boundary.
    """
    base = status_vector(original_parsed, labels)
    occ = ~np.isnan(base)
    if not occ.any():
        return float("nan")
    flips = []
    for p in perturbed_parsed:
        other = status_vector(p, labels)
        both = occ & ~np.isnan(other)
        if both.any():
            flips.append(float(np.mean(base[both] != other[both])))
    return float(np.mean(flips)) if flips else float("nan")


def uncertainty_signal(parsed: dict, labels: Sequence[str] = DEFAULT_LABELS) -> float:
    """Mean distance of the calibrated probability from the decision boundary.

    ``1 - mean|2p - 1|``: 0 when every label is asserted with full confidence,
    1 when every label sits exactly at 0.5. Uses the calibrated probability, so
    call it after temperature scaling.
    """
    p = probability_vector(parsed, labels)
    p = p[~np.isnan(p)]
    if len(p) == 0:
        return float("nan")
    return float(1.0 - np.mean(np.abs(2.0 * p - 1.0)))


# --------------------------------------------------------------------------- #
# shift score (heterogeneity across perturbation conditions)
# --------------------------------------------------------------------------- #
def shift_score(vectors: np.ndarray) -> dict:
    """Aggregate several probability vectors of one image into a stability read.

    Computed *without* ground truth. ``spread`` is the mean per-label standard
    deviation across conditions -- high spread means the model's belief depends
    strongly on the perturbation, i.e. the image-quality channel is doing work.

    Parameters
    ----------
    vectors : (n_conditions, n_labels) array; NaN allowed.

    Returns
    -------
    dict with ``spread``, ``mean``, ``valley``.
    ``valley`` needs >= 2 conditions and measures the drop at the midpoint,
    which is what a genuinely inverted-U sensitivity curve would show.
    """
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or v.shape[0] == 0:
        return {"spread": float("nan"), "mean": float("nan"), "valley": float("nan")}
    out = {
        "spread": float(np.nanmean(np.nanstd(v, axis=0))) if v.shape[0] > 1 else float("nan"),
        "mean": float(np.nanmean(v)),
        "valley": float("nan"),
    }
    if v.shape[0] >= 2:
        per_cond = np.nanmean(v, axis=1)
        if np.isfinite(per_cond).sum() >= 2:
            out["valley"] = float(np.nanmax(per_cond) - np.nanmean(per_cond))
    return out


def model_disagreement(pred: np.ndarray, y: np.ndarray) -> float:  # pragma: no cover - unused
    raise NotImplementedError("disagreement across models is assembled by the caller")


# --------------------------------------------------------------------------- #
# logistic calibration of signals -> P(error)
# --------------------------------------------------------------------------- #
class LogisticErrorModel:
    """Map ground-truth-free signals to a calibrated P(error).

    Fitted by Newton-Raphson on standardized features with an L2 penalty, which
    is stable on the few hundred rows a pilot provides. Deliberately simple:
    the point is a *calibrated* score, not a flexible learner that could absorb
    the test set.

    Features are standardized using ``source_val`` statistics and those exact
    statistics are reused at predict time, so the transformation is fixed.
    """

    def __init__(self, l2: float = 1.0):
        self.l2 = float(l2)
        self.mu_: Optional[np.ndarray] = None
        self.sigma_: Optional[np.ndarray] = None
        self.w_: Optional[np.ndarray] = None
        self.feature_names_: list[str] = []
        self.n_iter_run_: int = 0
        self.converged_: bool = False
        self.final_grad_norm_: float = float("nan")

    # -- internals -------------------------------------------------------
    @staticmethod
    def _design(X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), X])

    def _standardize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if fit:
            if np.isnan(X).any():
                raise ValueError("fit() received NaN features; drop or impute before fitting")
            self.mu_ = np.nanmean(X, axis=0)
            self.sigma_ = np.nanstd(X, axis=0)
            self.sigma_ = np.where(self.sigma_ > 1e-12, self.sigma_, 1.0)
        Z = (X - self.mu_) / self.sigma_
        return np.nan_to_num(Z, nan=0.0)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    # -- public API ------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, n_iter: int = 200,
            feature_names: Optional[Sequence[str]] = None,
            tol: float = 1e-10) -> "LogisticErrorModel":
        """Newton-Raphson with step halving.

        The intercept is left unpenalized so the fitted probabilities stay
        calibrated (mean predicted P(error) equals the observed error rate), which
        is what lets the cost thresholds be interpreted as real risk levels.
        """
        y = np.asarray(y, dtype=float)
        Z = self._standardize(X, fit=True)
        A = self._design(Z)
        d = A.shape[1]
        self.feature_names_ = list(feature_names or range(d - 1))
        w = np.zeros(d)
        pen_diag = np.diag(np.r_[0.0, np.full(d - 1, self.l2)])
        self.converged_ = False

        for it in range(n_iter):
            p = self._sigmoid(A @ w)
            grad = A.T @ (p - y) / len(y) + (pen_diag @ w) / len(y)
            H = (A * (p * (1 - p))[:, None]).T @ A / len(y) + pen_diag / len(y)
            self.final_grad_norm_ = float(np.linalg.norm(grad))
            if self.final_grad_norm_ < tol:
                self.converged_ = True
                self.n_iter_run_ = it
                break
            try:
                step = np.linalg.solve(H + 1e-10 * np.eye(d), grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H + 1e-10 * np.eye(d), grad, rcond=None)[0]
            if not np.all(np.isfinite(step)):
                break
            # step halving: guarantees non-decreasing penalized likelihood
            cur = self._penalized_nll(A, y, w, pen_diag)
            alpha = 1.0
            for _ in range(40):
                w_new = w - alpha * step
                if self._penalized_nll(A, y, w_new, pen_diag) <= cur:
                    break
                alpha /= 2
            if np.max(np.abs(w_new - w)) < tol:
                w = w_new
                self.converged_ = True
                self.n_iter_run_ = it
                break
            w = w_new
        self.w_ = w
        return self

    @staticmethod
    def _penalized_nll(A, y, w, pen_diag) -> float:
        z = A @ w
        # stable log-loss
        ll = np.sum(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z))
        return float(ll / len(y) + (w @ pen_diag @ w) / (2 * len(y)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("call fit() first")
        Z = self._standardize(X, fit=False)
        return self._sigmoid(self._design(Z) @ self.w_)

    def calibration_report(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Sanity check that the map is usable as a probability, not just a ranker."""
        p = self.predict_proba(X)
        y = np.asarray(y, float)
        return {
            "n": int(len(y)),
            "observed_error_rate": float(np.mean(y)),
            "mean_predicted": float(np.mean(p)),
            "median_predicted": float(np.median(p)),
            "min_predicted": float(np.min(p)),
            "max_predicted": float(np.max(p)),
            "bias": float(np.mean(p) - np.mean(y)),
            "converged": bool(self.converged_),
            "n_iter": int(self.n_iter_run_),
            "final_grad_norm": float(self.final_grad_norm_),
        }

    def coefficients(self) -> dict:
        if self.w_ is None:
            return {}
        return {name: float(v) for name, v in zip(["intercept", *self.feature_names_], self.w_)}


# --------------------------------------------------------------------------- #
# cost-sensitive three-way decision policy
# --------------------------------------------------------------------------- #
@dataclass
class CostModel:
    """Declared clinical costs. Set these BEFORE looking at test results.

    Defaults encode a standard screening asymmetry: letting an error through
    (``c_accept_error``) is far worse than deferring a case to a human
    (``c_abstain``), while asking for a re-read costs something but less than a
    miss. The ratio, not the absolute scale, drives the thresholds.
    """
    c_accept_error: float = 10.0   # accepted, wrong
    c_accept_correct: float = 0.0
    c_clarify: float = 1.0         # cost of gather-more-evidence; paid regardless of outcome
    c_abstain: float = 2.0         # defer to human reader
    cost_unit: str = "relative units (ratio is what matters)"

    def describe(self) -> dict:
        d = asdict(self)
        d["accept_to_abstain_ratio"] = self.c_accept_error / self.c_abstain if self.c_abstain else float("inf")
        return d


@dataclass
class ThreeWayPolicy:
    """Thresholds on P(error) splitting cases into accept / clarify / abstain.

    Decision rule, for calibrated error probability ``r = P(error | signals)``:

    * ``r <= tau_accept``  -> ACCEPT  (automatic read)
    * ``r >= tau_abstain`` -> ABSTAIN (human reads the image)
    * otherwise            -> CLARIFY (gather more evidence, then re-decide)

    Fitted on ``source_val`` by minimizing empirical cost under a declared
    :class:`CostModel`; the expected cost of clarifying uses the observed
    error rate of the clarify region itself, so no ground-truth-free assumption
    about what clarification achieves is smuggled in.
    """
    tau_accept: float = float("nan")
    tau_abstain: float = float("nan")
    expected_cost: float = float("nan")
    n_accept: int = 0
    n_clarify: int = 0
    n_abstain: int = 0
    fitted_on: str = ""

    def decide(self, risk: np.ndarray) -> np.ndarray:
        """0 = accept, 1 = clarify, 2 = abstain."""
        r = np.asarray(risk, dtype=float)
        out = np.full(len(r), 1, dtype=int)
        out[r <= self.tau_accept] = 0
        out[r >= self.tau_abstain] = 2
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def _expected_cost(err: np.ndarray, risk: np.ndarray, ta: float, tb: float,
                   cm: CostModel) -> float:
    """Empirical mean cost of a (tau_accept, tau_abstain) pair.

    All three cost terms are **per case**, so each is multiplied by the number
    of cases in its region before summing. (Getting this wrong is easy and
    silent: adding ``c_abstain`` once instead of once per abstained case made
    "abstain on everything" look almost free, which is the opposite of the
    truth and would have quietly inverted the whole policy.)
    """
    accept = risk <= ta
    abstain = risk >= tb
    clarify = ~(accept | abstain)
    n = len(err)
    if n == 0:
        return float("nan")
    total = 0.0
    if accept.any():
        total += float(np.sum(cm.c_accept_error * err[accept]
                              + cm.c_accept_correct * (1 - err[accept])))
    if clarify.any():
        # clarification costs its own price plus the error that survives it;
        # we charge the observed error rate of the clarified region (conservative:
        # assumes that asking for more evidence does not by itself fix anything).
        total += float(np.sum(cm.c_clarify + cm.c_accept_error * err[clarify]))
    if abstain.any():
        total += float(cm.c_abstain) * int(abstain.sum())
    return float(total / n)


def fit_three_way_policy(risk: np.ndarray, err: np.ndarray, cm: CostModel,
                         grid: Optional[np.ndarray] = None,
                         fitted_on: str = "") -> ThreeWayPolicy:
    """Grid-search the cost-minimizing thresholds on (risk, err) -- source_val only."""
    risk = np.asarray(risk, dtype=float)
    err = np.asarray(err, dtype=float)
    ok = np.isfinite(risk) & np.isfinite(err)
    risk, err = risk[ok], err[ok]
    if len(risk) == 0:
        return ThreeWayPolicy(fitted_on=fitted_on)
    if grid is None:
        qs = np.unique(np.round(np.linspace(0, 1, 51), 4))
        grid = np.unique(np.quantile(risk, qs))
        grid = np.unique(np.r_[0.0, grid, 1.0])

    best = (np.inf, 0.0, 1.0)
    for ta in grid:
        for tb in grid:
            if tb < ta:
                continue
            c = _expected_cost(err, risk, ta, tb, cm)
            if np.isfinite(c) and c < best[0]:
                best = (c, float(ta), float(tb))
    _, ta, tb = best
    dec = np.full(len(risk), 1)
    dec[risk <= ta] = 0
    dec[risk >= tb] = 2
    return ThreeWayPolicy(
        tau_accept=ta, tau_abstain=tb, expected_cost=best[0],
        n_accept=int((dec == 0).sum()), n_clarify=int((dec == 1).sum()),
        n_abstain=int((dec == 2).sum()), fitted_on=fitted_on,
    )


# --------------------------------------------------------------------------- #
# policy evaluation (test split, or any labelled split)
# --------------------------------------------------------------------------- #
def evaluate_policy(dec: np.ndarray, err: np.ndarray, cm: CostModel,
                    risk: Optional[np.ndarray] = None) -> dict:
    """Outcome of a three-way policy on labelled data.

    Reports the clinically meaningful quantities: how much work is automated,
    how much is deferred, and -- critically -- the error rate *among the
    accepted cases*, which is what a deployment actually exposes patients to.
    """
    dec = np.asarray(dec, dtype=int)
    err = np.asarray(err, dtype=float)
    n = len(dec)
    if n == 0:
        return {}
    acc, cla, abst = dec == 0, dec == 1, dec == 2
    auto_rate = float(acc.mean())
    accepted_err = float(err[acc].mean()) if acc.any() else float("nan")
    out = {
        "n": int(n),
        "accept_rate": auto_rate,
        "clarify_rate": float(cla.mean()),
        "abstain_rate": float(abst.mean()),
        "accepted_error_rate": accepted_err,
        "clarified_error_rate": float(err[cla].mean()) if cla.any() else float("nan"),
        "overall_error_rate": float(err.mean()),
        # effective error after the policy: accepted errors are exposed,
        # deferred cases are handled by a human at the cost of c_abstain
        "exposed_error_rate": accepted_err * auto_rate,
        "workload_reduction": auto_rate,
        "relative_error_reduction_vs_accept_all": (
            (float(err.mean()) - accepted_err) / float(err.mean())
            if acc.any() and err.mean() > 0 else float("nan")
        ),
    }
    if risk is not None:
        r = np.asarray(risk, dtype=float)
        mask = acc & np.isfinite(r)
        out["mean_risk_accepted"] = float(r[mask].mean()) if mask.any() else float("nan")
    return out


# --------------------------------------------------------------------------- #
# provenance helper
# --------------------------------------------------------------------------- #
def write_provenance(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
