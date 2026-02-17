"""
efficiency.py

Step 3: build detector response / "efficiency" models from calibrated standard spectra.

Your intended workflow (matches your monolithic notebook):
- For each standard with known activity x (usually Bq), compute observed y by integrating a
  calibrated spectrum over an ROI.
- Optionally attach y-uncertainty per point (e.g., biological SD from replicate spectra).
- Fit using scipy.optimize.curve_fit with YOUR model forms:
    nonparalyzable_func(x,a,b) = a * (x/(1 + b*x))
    linear_func(x,a,b)        = a*x + b
- Save fitted parameters + diagnostics for downstream MLE.

This module expects calibrated spectra to come from ArtifactStore.load_calibration()
and be either:
- dict schema "bq_v1": {"counts_vals": ..., "bin_centers_kev": ..., "meta": {"schema":"bq_v1", ...}}
- dict schema "dict_v1": {"counts": ..., "energy_axis": optional, "meta": {"schema":"dict_v1", ...}}
- OR a live object with best-effort attributes (during interactive experimentation)

Authoritative axis for becquerel-lightweight format:
- x = obj["bin_centers_kev"] (keV)
- y = obj["counts_vals"] (counts per bin)

Units:
- default_observed_extractor can return counts, counts/s, counts/min via rate_units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import CalibrationResult, EfficiencyModel


# ----------------------------
# Extraction helpers
# ----------------------------

def _get_counts_axis_livetime(obj: Any) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float]]:
    """
    Extract (counts, axis, livetime_s) from:
      - dict schema "bq_v1" or "dict_v1" (recommended),
      - or a live spectrum-like object (best effort).
    """
    # Lightweight dicts returned by ArtifactStore.load_calibration()
    if isinstance(obj, dict):
        meta = obj.get("meta", {}) if isinstance(obj.get("meta", {}), dict) else {}
        schema = meta.get("schema")

        if schema == "bq_v1":
            counts = np.asarray(obj["counts_vals"], dtype=float).reshape(-1)
            axis = np.asarray(obj["bin_centers_kev"], dtype=float).reshape(-1)
            lt = meta.get("livetime_s", None)
            livetime = float(lt) if lt is not None else None
            return counts, axis, livetime

        if schema == "dict_v1":
            counts = np.asarray(obj["counts"], dtype=float).reshape(-1)
            axis = np.asarray(obj["energy_axis"], dtype=float).reshape(-1) if "energy_axis" in obj else None
            lt = meta.get("livetime_s", None)
            livetime = float(lt) if lt is not None else None
            return counts, axis, livetime

        # If someone handed you a dict but didn't include schema, fail loudly.
        raise TypeError(
            "Spectrum dict missing meta['schema']. Expected 'bq_v1' or 'dict_v1'. "
            f"Got meta keys={list(meta.keys())} and dict keys={list(obj.keys())}."
        )

    # Live objects (best effort) — useful during notebook experimentation
    # Prefer becquerel-like attributes if present
    counts_vals = getattr(obj, "counts_vals", None)
    bin_centers_kev = getattr(obj, "bin_centers_kev", None)
    if counts_vals is not None:
        counts = np.asarray(counts_vals, dtype=float).reshape(-1)
        axis = np.asarray(bin_centers_kev, dtype=float).reshape(-1) if bin_centers_kev is not None else None
        lt = getattr(obj, "livetime", None)
        livetime = float(lt) if lt is not None else None
        return counts, axis, livetime

    # Fallback: .counts + any plausible energy axis attributes
    counts_attr = getattr(obj, "counts", None)
    if counts_attr is None:
        raise TypeError("Spectrum object must be a dict schema (bq_v1/dict_v1) or have .counts_vals/.counts")

    counts = np.asarray(counts_attr, dtype=float).reshape(-1)

    axis = None
    for attr in ("energies", "energy", "bin_edges", "edges", "energy_bins"):
        v = getattr(obj, attr, None)
        if v is not None:
            try:
                axis = np.asarray(v, dtype=float).reshape(-1)
                break
            except Exception:
                pass

    # livetime best effort
    livetime = None
    for attr in ("livetime", "live_time", "livetime_s", "real_time", "realtime"):
        v = getattr(obj, attr, None)
        if v is not None:
            try:
                livetime = float(v)
                break
            except Exception:
                pass

    return counts, axis, livetime


def integrate_window(
    counts: np.ndarray,
    axis: Optional[np.ndarray],
    emin_keV: float,
    emax_keV: float,
) -> float:
    """
    Integrate counts in [emin_keV, emax_keV].

    If axis is None, treat emin/emax as index bounds (channel-space).
    """
    counts = np.asarray(counts, dtype=float).reshape(-1)

    if axis is None:
        i0 = int(max(0, np.floor(emin_keV)))
        i1 = int(min(len(counts), np.ceil(emax_keV)))
        return float(np.sum(counts[i0:i1]))

    ax = np.asarray(axis, dtype=float).reshape(-1)

    # axis as bin edges (len = n+1)
    if len(ax) == len(counts) + 1:
        left = ax[:-1]
        right = ax[1:]
        mask = (right > emin_keV) & (left < emax_keV)
        return float(np.sum(counts[mask]))

    # axis as bin centers (len = n)
    if len(ax) == len(counts):
        mask = (ax >= emin_keV) & (ax <= emax_keV)
        return float(np.sum(counts[mask]))

    # unknown axis length; be conservative
    n = min(len(ax), len(counts))
    mask = (ax[:n] >= emin_keV) & (ax[:n] <= emax_keV)
    return float(np.sum(counts[:n][mask]))


def default_observed_extractor(
    spectrum_obj: Any,
    *,
    emin_keV: float,
    emax_keV: float,
    livetime_s: Optional[float] = None,
    rate_units: str = "counts_per_min",
) -> float:
    """
    Turn a calibrated spectrum into a single observed scalar:

      window_counts = sum(counts in ROI)
      return window_counts              if rate_units == "counts"
      return window_counts / lt         if rate_units == "counts_per_s"
      return window_counts / lt * 60    if rate_units == "counts_per_min"

    If lt is missing (and rate_units is per-time), we fall back to returning window_counts.
    """
    counts, axis, lt_obj = _get_counts_axis_livetime(spectrum_obj)
    window_counts = integrate_window(counts, axis, emin_keV, emax_keV)

    if rate_units == "counts":
        return float(window_counts)

    lt = livetime_s if livetime_s is not None else lt_obj
    if lt is None or lt <= 0:
        return float(window_counts)

    if rate_units == "counts_per_s":
        return float(window_counts / lt)

    if rate_units == "counts_per_min":
        return float(window_counts / lt * 60.0)

    raise ValueError(f"Unknown rate_units={rate_units!r}")


# ----------------------------
# Model forms (match your notebook)
# ----------------------------

def nonparalyzable_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return a * (x / (1.0 + (b * x)))


def linear_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return a * x + b


# ----------------------------
# SciPy curve_fit helper (no custom fallbacks)
# ----------------------------

def _require_curve_fit():
    try:
        from scipy.optimize import curve_fit  # type: ignore
    except Exception as e:
        raise ImportError(
            "scipy is required for efficiency fitting (uses scipy.optimize.curve_fit). "
            "Install scipy or run this stage in an environment that has it."
        ) from e
    return curve_fit


def _fit_with_curve_fit(func, x, y, *, sigma: Optional[np.ndarray] = None):
    curve_fit = _require_curve_fit()

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float).reshape(-1)
        popt, pcov = curve_fit(func, x, y, sigma=sigma, absolute_sigma=True, maxfev=50_000)
    else:
        popt, pcov = curve_fit(func, x, y, maxfev=50_000)

    yhat = func(x, *popt)
    nmse = float(np.mean((y - yhat) ** 2) / (np.mean(y ** 2) + 1e-12))

    diag = {
        "nmse": nmse,
        "n_points": int(len(x)),
        "fit": "scipy_curve_fit",
    }
    return popt, pcov, diag


# ----------------------------
# Public fitting API
# ----------------------------

@dataclass(frozen=True)
class EfficiencyFitSpec:
    """
    One response-curve fit configuration.

    label_to_activity:
      x-values. Keep as nCi for consistency.
      If you want nCi, you can do that too—just be consistent downstream.

    label_to_yerr:
      Optional y-error bars (same units as extractor output) keyed by label.
      Example: biological SD of integrated ROI counts across replicate spectra.

    model_type:
      "nonparalyzable" (a * x/(1+b*x)) or "linear" (a*x + b)

    ROI:
      emin_keV / emax_keV (in keV if calibrated axis exists)

    livetime_s:
      Optional override if not stored in spectrum meta.

    rate_units:
      "counts" | "counts_per_s" | "counts_per_min"
    """
    name: str
    label_to_activity: Mapping[str, float]
    model_type: str = "nonparalyzable"
    emin_keV: float = 200.0
    emax_keV: float = 1250.0
    livetime_s: Optional[float] = None
    rate_units: str = "counts_per_min"
    label_to_yerr: Optional[Mapping[str, float]] = None


def fit_efficiency_model(
    cal: CalibrationResult,
    spec: EfficiencyFitSpec,
    *,
    observed_extractor: Optional[Callable[..., float]] = None,
) -> EfficiencyModel:
    """
    Fits one model for one isotope/standard-set.

    Returns an EfficiencyModel with:
      - params: {"a": ..., "b": ...}
      - meta: used points, ROI, units, covariance, diagnostics, etc.
    """
    extractor = observed_extractor or default_observed_extractor

    labels = list(spec.label_to_activity.keys())

    A_list: list[float] = []
    R_list: list[float] = []
    used_labels: list[str] = []

    for lbl in labels:
        if lbl not in cal.calibrated:
            continue

# That “continue silently” can hide mistakes when merging. Consider changing it to:
# - raise if too many labels are missing, or
# - log missing labels.

        obj = cal.get(lbl)
        R = extractor(
            obj,
            emin_keV=spec.emin_keV,
            emax_keV=spec.emax_keV,
            livetime_s=spec.livetime_s,
            rate_units=spec.rate_units,
        )

        A_list.append(float(spec.label_to_activity[lbl]))
        R_list.append(float(R))
        used_labels.append(lbl)

    if len(R_list) < 2:
        raise ValueError(
            f"Not enough spectra to fit response model {spec.name!r}. "
            f"Found {len(R_list)} usable labels out of {len(labels)}."
        )

    A_used = np.asarray(A_list, dtype=float)
    R_used = np.asarray(R_list, dtype=float)

    sigma = None
    if spec.label_to_yerr is not None:
        sigma = np.asarray([float(spec.label_to_yerr[lbl]) for lbl in used_labels], dtype=float)

    mt = spec.model_type.lower()
    if mt == "linear":
        func = linear_func
        popt, pcov, diag = _fit_with_curve_fit(func, A_used, R_used, sigma=sigma)
        params = {"a": float(popt[0]), "b": float(popt[1])}
        model_type = "linear"
    elif mt in ("nonparalyzable", "non_paralyzable", "np"):
        func = nonparalyzable_func
        popt, pcov, diag = _fit_with_curve_fit(func, A_used, R_used, sigma=sigma)
        params = {"a": float(popt[0]), "b": float(popt[1])}
        model_type = "nonparalyzable"
    else:
        raise ValueError(f"Unknown model_type={spec.model_type!r}. Use 'linear' or 'nonparalyzable'.")

    meta = {
        "fit_spec": {
            "name": spec.name,
            "model_type": spec.model_type,
            "emin_keV": float(spec.emin_keV),
            "emax_keV": float(spec.emax_keV),
            "livetime_s": float(spec.livetime_s) if spec.livetime_s is not None else None,
            "rate_units": spec.rate_units,
            "used_sigma": (sigma is not None),
        },
        "used_labels": used_labels,
        "A_used": A_used.tolist(),
        "R_used": R_used.tolist(),
        "R_used_sigma": sigma.tolist() if sigma is not None else None,
        "diagnostics": diag,
        "covariance": pcov.tolist(),
        "experiment_id": cal.experiment_id,
    }

    return EfficiencyModel(
        name=spec.name,
        model_type=model_type,
        params=params,
        meta=meta,
    )


def fit_efficiency_models(
    cal: CalibrationResult,
    fit_specs: Sequence[EfficiencyFitSpec],
    *,
    observed_extractor: Optional[Callable[..., float]] = None,
) -> Dict[str, EfficiencyModel]:
    """
    Fit multiple models (e.g. one per isotope) and return dict[name -> EfficiencyModel].
    """
    out: Dict[str, EfficiencyModel] = {}
    for fs in fit_specs:
        model = fit_efficiency_model(cal, fs, observed_extractor=observed_extractor)
        out[model.name] = model
    return out


# ----------------------------
# Evaluating models later (for MLE)
# ----------------------------

def evaluate_efficiency_model(model: EfficiencyModel, x_activity: np.ndarray) -> np.ndarray:
    """
    Given an EfficiencyModel and activities, return predicted observed y (same units as fit).

    model.params uses {"a","b"} for BOTH models.
    """
    x = np.asarray(x_activity, dtype=float)

    if model.model_type == "linear":
        a = float(model.params["a"])
        b = float(model.params["b"])
        return linear_func(x, a, b)

    if model.model_type == "nonparalyzable":
        a = float(model.params["a"])
        b = float(model.params["b"])
        return nonparalyzable_func(x, a, b)

    raise ValueError(f"Unknown model.model_type={model.model_type!r}")

# # How you’ll use this in 03_efficiency.ipynb (concretely)
# # import sys
# # from pathlib import Path
# # sys.path.insert(0, str(Path.cwd().parent))

# # from gamma.artifacts import ArtifactStore
# # from gamma.efficiency import EfficiencyFitSpec, fit_efficiency_models

# # store = ArtifactStore("../artifacts")

# # cal = store.load_calibration("ANIL_AC", "peakfit_v2_emin120_emax2000")

# # # You provide the known activity per label (from your standards)
# # ac_fit = EfficiencyFitSpec(
# #     name="Ac225_det_v1",
# #     label_to_activity_bq={
# #         "125nCi": 125 * 37_000,   # example only; use YOUR mapping
# #         "62.5nCi": 62.5 * 37_000,
# #         # ...
# #     },
# #     model_type="nonparalyzable",
# #     emin_keV=300,
# #     emax_keV=1700,
# #     livetime_s=60.0,
# # )

# # models = fit_efficiency_models(cal, [ac_fit])

# # store.save_efficiency_model(models["Ac225_det_v1"])



# """
# efficiency.py

# Step 3: build detector response / efficiency models from calibrated standard spectra.

# Big picture:
# - For each standard sample with known activity A (Bq), compute an observed count rate R
#   from the calibrated spectrum (e.g., integrate counts in an energy window and divide by livetime).
# - Fit a model R = f(A; params).
# - Return an EfficiencyModel you can save and later use during MLE.

# This module intentionally does NOT assume your exact old notebook logic.
# Instead, it gives you a clean, explicit interface that you can adapt gradually.

# Supported spectrum inputs (from CalibrationResult.calibrated[label]):
# - lightweight dicts produced by ArtifactStore.load_calibration() for NPZ entries:
#     {"counts": array, "axis": array (optional), "meta": {...}}
# - objects with attribute `.counts` (best-effort), e.g. becquerel.Spectrum

# The "axis" is assumed to be energy in keV when present.
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

# import numpy as np

# from .types import CalibrationResult, EfficiencyModel


# # ----------------------------
# # Extraction helpers
# # ----------------------------

# def _get_counts_axis_livetime(obj: Any) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float]]:
#     """
#     Extract counts (1D), optional axis (1D), and optional livetime (seconds).

#     - If axis exists, we assume it is energy in keV or bin edges (best-effort).
#     - Livetime is best-effort: you may store it in obj["meta"]["livetime_s"] or similar later.
#     """
#     # lightweight dict
#     if isinstance(obj, dict):
#         meta = obj.get("meta", {}) if isinstance(obj.get("meta", {}), dict) else {}
#         schema = meta.get("schema")

#         if schema == "bq_v1":
#             counts = np.asarray(obj["counts_vals"], dtype=float).reshape(-1)
#             axis = np.asarray(obj["bin_centers_kev"], dtype=float).reshape(-1)
#             livetime = meta.get("livetime_s", None)
#             livetime = float(livetime) if livetime is not None else None
#             return counts, axis, livetime

#         if schema == "dict_v1":
#             counts = np.asarray(obj["counts"], dtype=float).reshape(-1)
#             axis = np.asarray(obj["energy_axis"], dtype=float).reshape(-1) if "energy_axis" in obj else None
#             livetime = meta.get("livetime_s", None)
#             livetime = float(livetime) if livetime is not None else None
#             return counts, axis, livetime


#     # object with .counts
#     counts_attr = getattr(obj, "counts", None)
#     if counts_attr is None:
#         raise TypeError("Spectrum object must be a dict with 'counts' or have attribute .counts")

#     counts = np.asarray(counts_attr, dtype=float).reshape(-1)

#     axis = None
#     for attr in ("energies", "energy", "bin_edges", "edges", "energy_bins"):
#         v = getattr(obj, attr, None)
#         if v is not None:
#             try:
#                 axis = np.asarray(v, dtype=float).reshape(-1)
#                 break
#             except Exception:
#                 pass

#     # livetime best-effort
#     livetime = None
#     for attr in ("livetime", "live_time", "livetime_s", "real_time", "realtime"):
#         v = getattr(obj, attr, None)
#         if v is not None:
#             try:
#                 livetime = float(v)
#                 break
#             except Exception:
#                 pass

#     return counts, axis, livetime


# def integrate_window(
#     counts: np.ndarray,
#     axis: Optional[np.ndarray],
#     emin_keV: float,
#     emax_keV: float,
# ) -> float:
#     """
#     Integrate counts in [emin_keV, emax_keV].

#     If axis is None, we treat indices as "channels" and integrate by index range
#     (NOT physically meaningful unless your calibration uses channel as axis).
#     """
#     counts = np.asarray(counts, dtype=float).reshape(-1)

#     if axis is None:
#         # assume axis is channel index
#         i0 = int(max(0, np.floor(emin_keV)))
#         i1 = int(min(len(counts), np.ceil(emax_keV)))
#         return float(np.sum(counts[i0:i1]))

#     ax = np.asarray(axis, dtype=float).reshape(-1)

#     # Two common cases:
#     # - axis same length as counts: axis is bin centers
#     # - axis length is counts+1: axis is bin edges
#     if len(ax) == len(counts) + 1:
#         # bin edges: choose bins whose edges overlap the interval
#         # bins i span [ax[i], ax[i+1]]
#         left = ax[:-1]
#         right = ax[1:]
#         mask = (right > emin_keV) & (left < emax_keV)
#         return float(np.sum(counts[mask]))
#     elif len(ax) == len(counts):
#         # bin centers
#         mask = (ax >= emin_keV) & (ax <= emax_keV)
#         return float(np.sum(counts[mask]))
#     else:
#         # unknown axis length; fall back to min length
#         n = min(len(ax), len(counts))
#         mask = (ax[:n] >= emin_keV) & (ax[:n] <= emax_keV)
#         return float(np.sum(counts[:n][mask]))


# def default_observed_rate_extractor(
#     spectrum_obj: Any,
#     *,
#     emin_keV: float,
#     emax_keV: float,
#     livetime_s: Optional[float] = None,
#     rate_units: str = "counts_per_min", 
# ) -> float:
#     """
#     Default way to turn a calibrated spectrum into a single observed count RATE:

#         R = counts_in_window / livetime

#     If livetime is missing, returns "counts in window" (rate-like but not a true rate).
#     """
#     counts, axis, lt_obj = _get_counts_axis_livetime(spectrum_obj)
#     window_counts = integrate_window(counts, axis, emin_keV, emax_keV)

#     lt = livetime_s if livetime_s is not None else lt_obj
#     if lt is None or lt <= 0:
#         return float(window_counts)

#     if rate_units == "counts":
#         return float(window_counts)

#     if rate_units == "counts_per_s":
#         return float(window_counts / lt)

#     if rate_units == "counts_per_min":
#         return float(window_counts / lt * 60.0)

#     raise ValueError(f"Unknown rate_units={rate_units!r}")


# # ----------------------------
# # Model forms
# # ----------------------------

# def linear_model(A: np.ndarray, k: float, b: float) -> np.ndarray:
#     """R = k*A + b"""
#     return k * A + b


# def nonparalyzable_model(A: np.ndarray, k: float, tau: float, b: float) -> np.ndarray:
#     """
#     A simple nonparalyzable-like response model (rate saturation):

#         R = (k*A) / (1 + k*tau*A) + b

#     k: proportionality from activity to true rate
#     tau: dead-time constant (s)
#     b: background rate offset (counts/s)

#     This isn’t the only possible form, but it’s a good general-purpose one.
#     """
#     A = np.asarray(A, dtype=float)
#     return (k * A) / (1.0 + (k * tau * A)) + b


# # ----------------------------
# # Fit utilities
# # ----------------------------

# def _try_curve_fit(func, x, y, p0, bounds):
#     try:
#         from scipy.optimize import curve_fit  # type: ignore
#     except Exception:
#         return None

#     popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=50_000)
#     return popt, pcov


# def _fit_linear(A: np.ndarray, R: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Any]]:
#     # Fit R = k*A + b via least squares
#     A = np.asarray(A, dtype=float).reshape(-1)
#     R = np.asarray(R, dtype=float).reshape(-1)

#     X = np.column_stack([A, np.ones_like(A)])
#     beta, *_ = np.linalg.lstsq(X, R, rcond=None)
#     k, b = float(beta[0]), float(beta[1])

#     Rhat = linear_model(A, k, b)
#     diag = {
#         "nmse": float(np.mean((R - Rhat) ** 2) / (np.mean(R ** 2) + 1e-12)),
#         "n_points": int(len(A)),
#         "model": "linear",
#     }
#     return {"k": k, "b": b}, diag


# def _fit_nonparalyzable(A: np.ndarray, R: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Any]]:
#     """
#     Try scipy curve_fit; if unavailable, do a small coarse search on tau + least squares on k,b.
#     """
#     A = np.asarray(A, dtype=float).reshape(-1)
#     R = np.asarray(R, dtype=float).reshape(-1)

#     # Reasonable initial guesses
#     k0 = float((R[-1] - R[0]) / (A[-1] - A[0] + 1e-12)) if len(A) >= 2 else 1.0
#     k0 = max(k0, 1e-12)
#     tau0 = 1e-6
#     b0 = float(np.min(R)) if len(R) else 0.0

#     # Try scipy first
#     out = _try_curve_fit(
#         nonparalyzable_model,
#         A,
#         R,
#         p0=[k0, tau0, b0],
#         bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
#     )
#     if out is not None:
#         popt, pcov = out
#         k, tau, b = map(float, popt.tolist())
#         Rhat = nonparalyzable_model(A, k, tau, b)
#         diag = {
#             "nmse": float(np.mean((R - Rhat) ** 2) / (np.mean(R ** 2) + 1e-12)),
#             "n_points": int(len(A)),
#             "model": "nonparalyzable",
#             "fit": "scipy_curve_fit",
#         }
#         return {"k": k, "tau": tau, "b": b}, diag

#     # Fallback: coarse tau grid search
#     tau_grid = np.logspace(-9, -2, 120)  # adjust later if needed
#     best = None

#     for tau in tau_grid:
#         # For fixed tau, model is: R ≈ (kA)/(1 + k tau A) + b
#         # Not linear in k, but we can still do a small search on k as well.
#         # We'll do a coarse search on k too.
#         k_grid = np.logspace(-12, 2, 120)
#         for k in k_grid:
#             # Solve best b for this (k,tau) by least squares offset:
#             # R ≈ g(A; k,tau) + b  =>  b = mean(R - g)
#             g = (k * A) / (1.0 + (k * tau * A))
#             b = float(np.mean(R - g))
#             Rhat = g + b
#             mse = float(np.mean((R - Rhat) ** 2))
#             if best is None or mse < best[0]:
#                 best = (mse, float(k), float(tau), float(b))

#     assert best is not None
#     mse, k, tau, b = best
#     Rhat = nonparalyzable_model(A, k, tau, b)
#     diag = {
#         "nmse": float(mse / (np.mean(R ** 2) + 1e-12)),
#         "n_points": int(len(A)),
#         "model": "nonparalyzable",
#         "fit": "grid_search_fallback",
#         "tau_grid": [float(tau_grid[0]), float(tau_grid[-1]), int(len(tau_grid))],
#     }
#     return {"k": k, "tau": tau, "b": b}, diag


# # ----------------------------
# # Public fitting API
# # ----------------------------

# @dataclass(frozen=True)
# class EfficiencyFitSpec:
#     """
#     Tells efficiency.py how to build one EfficiencyModel for one isotope.

#     label_to_activity_bq:
#         Known activities for each standard label you want to use in the fit.
#     model_type:
#         "linear" or "nonparalyzable"
#     energy window:
#         The ROI used to compute observed counts/rate from the spectrum.
#     livetime:
#         Optional override if your spectra objects don't carry it.
#     """
#     name: str
#     label_to_activity_bq: Mapping[str, float]
#     model_type: str = "nonparalyzable"
#     emin_keV: float = 200.0
#     emax_keV: float = 2000.0
#     livetime_s: Optional[float] = None


# def fit_efficiency_model(
#     cal: CalibrationResult,
#     spec: EfficiencyFitSpec,
#     *,
#     observed_rate_extractor: Optional[Callable[..., float]] = None,
# ) -> EfficiencyModel:
#     """
#     Fit one efficiency/response model for one isotope using calibration results.

#     Returns EfficiencyModel with fitted params.
#     """
#     extractor = observed_rate_extractor or default_observed_rate_extractor

#     labels = list(spec.label_to_activity_bq.keys())
#     A = np.array([float(spec.label_to_activity_bq[lbl]) for lbl in labels], dtype=float)

#     R_list = []
#     used_labels = []
#     for lbl in labels:
#         if lbl not in cal.calibrated:
#             continue
#         obj = cal.get(lbl)
#         R = extractor(obj, emin_keV=spec.emin_keV, emax_keV=spec.emax_keV, livetime_s=spec.livetime_s)
#         R_list.append(float(R))
#         used_labels.append(lbl)

#     if len(R_list) < 2:
#         raise ValueError(
#             f"Not enough spectra to fit efficiency model {spec.name!r}. "
#             f"Found {len(R_list)} usable labels out of {len(labels)}."
#         )

#     # Filter A to only used labels (in same order)
#     A_used = np.array([float(spec.label_to_activity_bq[lbl]) for lbl in used_labels], dtype=float)
#     R_used = np.array(R_list, dtype=float)

#     if spec.model_type.lower() == "linear":
#         params, diag = _fit_linear(A_used, R_used)
#         model_type = "linear"
#     elif spec.model_type.lower() in ("nonparalyzable", "non_paralyzable", "np"):
#         params, diag = _fit_nonparalyzable(A_used, R_used)
#         model_type = "nonparalyzable"
#     else:
#         raise ValueError(f"Unknown model_type={spec.model_type!r}. Use 'linear' or 'nonparalyzable'.")

#     meta = {
#         "fit_spec": {
#             "name": spec.name,
#             "model_type": spec.model_type,
#             "emin_keV": spec.emin_keV,
#             "emax_keV": spec.emax_keV,
#             "livetime_s": spec.livetime_s,
#         },
#         "used_labels": used_labels,
#         "A_used_bq": A_used.tolist(),
#         "R_used": R_used.tolist(),
#         "diagnostics": diag,
#         "experiment_id": cal.experiment_id,
#     }

#     return EfficiencyModel(
#         name=spec.name,
#         model_type=model_type,
#         params=params,
#         meta=meta,
#     )


# def fit_efficiency_models(
#     cal: CalibrationResult,
#     fit_specs: Sequence[EfficiencyFitSpec],
#     *,
#     observed_rate_extractor: Optional[Callable[..., float]] = None,
# ) -> Dict[str, EfficiencyModel]:
#     """
#     Fit multiple models (e.g., one per isotope) and return dict[name -> EfficiencyModel].
#     """
#     out: Dict[str, EfficiencyModel] = {}
#     for fs in fit_specs:
#         model = fit_efficiency_model(cal, fs, observed_rate_extractor=observed_rate_extractor)
#         out[model.name] = model
#     return out


# # ----------------------------
# # Evaluating models later (used by MLE)
# # ----------------------------

# def evaluate_efficiency_model(model: EfficiencyModel, A_bq: np.ndarray) -> np.ndarray:
#     """
#     Given an EfficiencyModel and activities (Bq), return predicted observed rate R.

#     This is what MLE should call to map activities -> predicted observed rates (or counts).
#     """
#     A = np.asarray(A_bq, dtype=float)

#     if model.model_type == "linear":
#         k = float(model.params["k"])
#         b = float(model.params.get("b", 0.0))
#         return linear_model(A, k, b)

#     if model.model_type == "nonparalyzable":
#         k = float(model.params["k"])
#         tau = float(model.params["tau"])
#         b = float(model.params.get("b", 0.0))
#         return nonparalyzable_model(A, k, tau, b)

#     raise ValueError(f"Unknown model.model_type={model.model_type!r}")
