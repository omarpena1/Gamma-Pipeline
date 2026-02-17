"""
mle.py

Step 4: Maximum-likelihood estimation (Poisson) of isotope activities from a calibrated spectrum.

Design goals:
- Generic: works for K=1..K isotopes.
- Extensible: optional background components can be added later without rewriting.
- Pure data out: no plotting here.
- Matches your legacy bridge logic:
    n_events(activity) from efficiency curve (total ROI counts for fixed 60 s convention),
    template normalized to a PDF over ROI, then scaled by n_events to predict counts/bin.

Important conventions (current / "Convention 1"):
- Efficiency model returns TOTAL counts in the ROI for your fixed acquisition (e.g., 60 s).
- Template normalization uses the same ROI used for the fit window unless overridden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List, Union

import numpy as np
from scipy import optimize

from .types import EfficiencyModel, MLEEstimate  # your project types
from .efficiency import evaluate_efficiency_model


# ---------- Units helpers ----------
NCI_TO_BQ = 37.0  # 1 nCi = 37 Bq


def _to_bq(x: float, *, activity_units: str) -> float:
    u = activity_units.lower()
    if u in ("bq", "becquerel", "becquerels"):
        return float(x)
    if u in ("nci", "nano-ci", "nanoci"):
        return float(x) * NCI_TO_BQ
    raise ValueError(f"Unknown activity_units={activity_units!r}. Use 'bq' or 'nci'.")


def _from_bq(x_bq: float, *, activity_units: str) -> float:
    u = activity_units.lower()
    if u in ("bq", "becquerel", "becquerels"):
        return float(x_bq)
    if u in ("nci", "nano-ci", "nanoci"):
        return float(x_bq) / NCI_TO_BQ
    raise ValueError(f"Unknown activity_units={activity_units!r}. Use 'bq' or 'nci'.")


# ---------- Spectrum access ----------
SpectrumLike = Any  # supports dict style or attribute style


def _get_array(obj: SpectrumLike, key: str) -> np.ndarray:
    """
    Supports:
      - dict: obj["bin_centers_kev"], obj["counts_vals"]
      - attribute: obj.bin_centers_kev, obj.counts_vals
    """
    if isinstance(obj, dict):
        if key not in obj:
            raise KeyError(f"Spectrum dict missing key {key!r}. Available keys: {list(obj.keys())}")
        return np.asarray(obj[key])
    if hasattr(obj, key):
        return np.asarray(getattr(obj, key))
    raise TypeError(f"Unsupported spectrum-like object; missing {key!r} as key or attribute.")


def _extract_xy(spec: SpectrumLike) -> Tuple[np.ndarray, np.ndarray]:
    x = _get_array(spec, "bin_centers_kev").astype(float)
    y = _get_array(spec, "counts_vals").astype(float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"Bad spectrum shapes: x={x.shape}, y={y.shape}")
    return x, y


def _roi_mask(x_keV: np.ndarray, emin_keV: float, emax_keV: float) -> np.ndarray:
    return (x_keV > float(emin_keV)) & (x_keV < float(emax_keV))


def _assert_same_grid(x_ref: np.ndarray, x_other: np.ndarray, *, atol: float = 1e-6) -> None:
    if x_ref.shape != x_other.shape:
        raise ValueError(f"Energy axis mismatch: {x_ref.shape} vs {x_other.shape}")
    if not np.allclose(x_ref, x_other, atol=atol, rtol=0.0):
        # You can relax this or implement resampling later if needed.
        raise ValueError("Energy axis values differ; templates and spectrum must share the same bin_centers_kev grid.")


# ---------- Numerical Hessian ----------
def approx_hessian(fun, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Central-difference Hessian approximation.
    fun: callable(x) -> scalar
    x: 1D array
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)

    for i in range(n):
        ei = np.zeros(n); ei[i] = 1.0
        for j in range(i, n):
            ej = np.zeros(n); ej[j] = 1.0
            fpp = fun(x + eps*ei + eps*ej)
            fpm = fun(x + eps*ei - eps*ej)
            fmp = fun(x - eps*ei + eps*ej)
            fmm = fun(x - eps*ei - eps*ej)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4*eps*eps)
            H[j, i] = H[i, j]
    return H


# ---------- Fit result container (pure data) ----------
@dataclass(frozen=True)
class MLEFitResult:
    labels: List[str]                      # isotope parameter order
    x_keV: np.ndarray                      # energies in ROI
    y_obs: np.ndarray                      # observed counts in ROI
    y_hat: np.ndarray                      # predicted mean counts in ROI (sum components)
    activities: Dict[str, float]           # in activity_units (the unit used during optimization)
    sigma_activities: Optional[Dict[str, float]]  # 1σ in activity_units
    cov: Optional[np.ndarray]              # covariance matrix (activity_units^2)
    success: bool
    message: str
    nll2: float                            # -2 log L (Poisson deviance)
    nit: Optional[int]
    diagnostics: Dict[str, Any]
    components_hat: Dict[str, np.ndarray]  # per-label predicted spectrum in ROI


# ---------- Core model building ----------
# def _template_pdf_in_roi(
#     template: SpectrumLike,
#     x_ref: np.ndarray,
#     roi_mask: np.ndarray,
# ) -> np.ndarray:
#     x_t, y_t = _extract_xy(template)
#     _assert_same_grid(x_ref, x_t)

#     y_roi = y_t[roi_mask]
#     denom = float(np.sum(y_roi))
#     if not np.isfinite(denom) or denom <= 0.0:
#         raise ValueError("Template has non-positive integral in ROI; cannot form PDF.")
#     pdf = y_t / denom  # note: outside ROI still defined, but denom is ROI sum
#     return pdf[roi_mask]  # return only ROI slice

def _template_pdf_in_roi(
    template: SpectrumLike,
    x_roi: np.ndarray,
    emin_keV: float,
    emax_keV: float,
) -> np.ndarray:
    """
    Return template PDF evaluated on x_roi, normalized over ROI.
    Allows template and data to have different energy grids.

    Uses linear interpolation; values outside template range are set to 0.
    """
    x_t, y_t = _extract_xy(template)

    # Evaluate template counts on the data ROI grid
    # outside template coverage -> 0 to avoid weird extrapolation
    y_eval = np.interp(x_roi, x_t, y_t, left=0.0, right=0.0)

    # Normalize over ROI (x_roi already corresponds to ROI)
    denom = float(np.sum(y_eval))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Template has non-positive integral on the ROI grid; cannot form PDF.")

    pdf_roi = y_eval / denom
    return pdf_roi


def _predict_component_counts_roi(
    activity: float,
    pdf_roi: np.ndarray,
    eff_model: EfficiencyModel,
) -> np.ndarray:
    # Convention 1: eff_model(activity) returns expected TOTAL counts in ROI for fixed assumed livetime
    n_events = float(np.asarray(evaluate_efficiency_model(eff_model, np.asarray([activity])))[0])
    if not np.isfinite(n_events) or n_events < 0.0:
        # negative can occur if model extrapolates poorly; clamp to 0 for physicality
        n_events = 0.0
    return pdf_roi * n_events


def _poisson_deviance(y_obs: np.ndarray, y_hat: np.ndarray) -> float:
    """
    -2 log L up to constants (Poisson deviance):
      2 * sum( y_hat - y_obs + y_obs * log(y_obs / y_hat) )
    with safe handling for y_obs == 0.
    """
    eps = 1e-12
    mu = np.clip(y_hat, eps, None)
    y = y_obs
    term = mu - y + np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2.0 * np.sum(term))


# ---------- Public API ----------
def fit_mle_poisson(
    spectrum: SpectrumLike,
    *,
    templates: Mapping[str, SpectrumLike],
    eff_models: Mapping[str, EfficiencyModel],
    emin_keV: float = 200.0,
    emax_keV: float = 1250.0,
    activity_units: str = "nci",
    x0: Optional[Mapping[str, float]] = None,
    bounds: Optional[Mapping[str, Tuple[float, Optional[float]]]] = None,
    method: str = "L-BFGS-B",
    hessian_eps: float = 1e-4,
    boundary_eps: float = 1e-12,
) -> MLEFitResult:
    """
    Fit activities for 1..K isotopes.

    Parameters
    ----------
    spectrum:
      Calibrated spectrum (bq_v1 dict or object) with bin_centers_kev and counts_vals.
    templates:
      Dict isotope_label -> template spectrum (same energy grid as spectrum).
    eff_models:
      Dict isotope_label -> EfficiencyModel (same keys as templates).
      Convention 1: evaluate_efficiency_model(model, activity) returns expected TOTAL ROI counts.
    emin_keV/emax_keV:
      ROI used both for (i) fitting and (ii) template normalization and n_events definition.
    activity_units:
      "nci" or "bq" - unit used during optimization. Returned activities/sigmas are in this unit.
    x0:
      Optional initial guess per isotope label.
    bounds:
      Optional bounds per isotope label. Default: (0, None) for each.
    """

    # ---- validate keys ----
    tpl_keys = list(templates.keys())
    if len(tpl_keys) == 0:
        raise ValueError("templates must contain at least 1 isotope (K>=1).")
    if set(tpl_keys) != set(eff_models.keys()):
        raise ValueError(f"templates keys {set(tpl_keys)} must match eff_models keys {set(eff_models.keys())}.")

    # stable parameter order
    labels = sorted(tpl_keys)

    # ---- extract observed spectrum and ROI ----
    x_ref, y_full = _extract_xy(spectrum)
    mask = _roi_mask(x_ref, emin_keV, emax_keV)
    x_roi = x_ref[mask]
    y_obs = y_full[mask]

    # ---- precompute template PDFs in ROI ----
    pdf_roi_by_label: Dict[str, np.ndarray] = {}
    for lab in labels:
        pdf_roi_by_label[lab] = _template_pdf_in_roi(
            templates[lab],
            x_roi=x_roi,
            emin_keV=emin_keV,
            emax_keV=emax_keV,
        )

    # ---- pack optimization vector ----
    K = len(labels)
    if x0 is None:
        x0_vec = np.ones(K, dtype=float)
    else:
        x0_vec = np.array([float(x0.get(lab, 1.0)) for lab in labels], dtype=float)

    if bounds is None:
        bnds = [(0.0, None)] * K
    else:
        bnds = []
        for lab in labels:
            lo, hi = bounds.get(lab, (0.0, None))
            lo_f = float(lo)
            hi_f = None if hi is None else float(hi)
            bnds.append((lo_f, hi_f))

    # ---- objective ----
    def nll2(vec: np.ndarray) -> float:
        y_hat = np.zeros_like(y_obs, dtype=float)
        for i, lab in enumerate(labels):
            comp = _predict_component_counts_roi(vec[i], pdf_roi_by_label[lab], eff_models[lab])
            y_hat += comp
        return _poisson_deviance(y_obs, y_hat)

    # ---- optimize ----
    fit = optimize.minimize(
        nll2,
        x0=x0_vec,
        method=method,
        bounds=bnds,
        options={},
    )

    # ---- reconstruct fitted spectra ----
    vec_hat = np.asarray(fit.x, dtype=float)
    components_hat: Dict[str, np.ndarray] = {}
    y_hat = np.zeros_like(y_obs, dtype=float)
    for i, lab in enumerate(labels):
        comp = _predict_component_counts_roi(vec_hat[i], pdf_roi_by_label[lab], eff_models[lab])
        components_hat[lab] = comp
        y_hat += comp

    # ---- uncertainty via inverse Hessian ----
    cov = None
    sigma_map = None
    try:
        H = approx_hessian(nll2, vec_hat, eps=float(hessian_eps))
        cov = np.linalg.pinv(H)

        sig = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        # boundary handling: if activity essentially 0, curvature-based sigma is unreliable
        for i in range(K):
            if vec_hat[i] <= boundary_eps:
                sig[i] = np.nan

        if np.all(~np.isfinite(sig)):
            cov = None
            sigma_map = None
        else:
            sigma_map = {lab: float(sig[i]) for i, lab in enumerate(labels)}
    except Exception:
        cov = None
        sigma_map = None

    activities_map = {lab: float(vec_hat[i]) for i, lab in enumerate(labels)}

    diagnostics: Dict[str, Any] = {
        "emin_keV": float(emin_keV),
        "emax_keV": float(emax_keV),
        "activity_units": str(activity_units),
        "method": str(method),
        "success": bool(fit.success),
        "status": int(getattr(fit, "status", -999)),
        "fun_nll2": float(fit.fun),
        "nit": int(getattr(fit, "nit", -1)) if getattr(fit, "nit", None) is not None else None,
        "message": str(getattr(fit, "message", "")),
        "x0": [float(v) for v in x0_vec],
        "bounds": [(float(lo), None if hi is None else float(hi)) for (lo, hi) in bnds],
    }

    return MLEFitResult(
        labels=labels,
        x_keV=x_roi,
        y_obs=y_obs,
        y_hat=y_hat,
        activities=activities_map,
        sigma_activities=sigma_map,
        cov=cov,
        success=bool(fit.success),
        message=str(getattr(fit, "message", "")),
        nll2=float(fit.fun),
        nit=int(getattr(fit, "nit", -1)) if getattr(fit, "nit", None) is not None else None,
        diagnostics=diagnostics,
        components_hat=components_hat,
    )

def to_mle_estimate(
    fit: MLEFitResult,
    *,
    experiment_id: str,
    label: str,
    activity_units: str = "nci",
    run_config: Optional[Mapping[str, Any]] = None,
) -> MLEEstimate:
    """
    Convert MLEFitResult into your pipeline's persisted MLEEstimate.

    Convention:
      - persisted activities/uncertainties are stored in Bq
      - diagnostics is namespaced so we can query reliably later

    diagnostics schema (stable):
      diagnostics = {
        "fit": {... optimizer + internal details ...},
        "fit_meta": {... summary fields ...},
        "run_config": {... your mle_config dict ...}   # optional
      }
    """
    activities_bq = {k: _to_bq(v, activity_units=activity_units) for k, v in fit.activities.items()}

    uncertainties_bq: Optional[Dict[str, float]] = None
    if fit.sigma_activities is not None:
        tmp: Dict[str, float] = {}
        for k, s in fit.sigma_activities.items():
            if s is None or not np.isfinite(s):
                continue
            tmp[k] = _to_bq(float(s), activity_units=activity_units)
        uncertainties_bq = tmp or None

    diagnostics: Dict[str, Any] = {
        "fit": dict(fit.diagnostics) if fit.diagnostics is not None else {},
        "fit_meta": {
            "nll2": float(fit.nll2),
            "labels": list(fit.labels),
            "success": bool(fit.success),
            "message": str(fit.message),
            "nit": int(fit.nit) if fit.nit is not None else None,
        },
    }
    if run_config is not None:
        diagnostics["run_config"] = dict(run_config)

    return MLEEstimate(
        experiment_id=str(experiment_id),
        label=str(label),
        activities_bq=activities_bq,
        uncertainties_bq=uncertainties_bq,
        diagnostics=diagnostics,
    )
