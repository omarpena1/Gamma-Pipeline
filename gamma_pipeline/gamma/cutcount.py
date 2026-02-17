"""
cutcount.py

Static-window cut-and-count activity estimation consistent with the pipeline.

Key choices:
- Uses EfficiencyModel (efficiency.py) as the activity -> expected window-counts model.
- Works in *counts* only (NOT count-rate). Assumes livetime_s = 60.0 everywhere.
- Supports both ROI paradigms:
    (A) peak-centered via (center_keV, resolution_pct_fwhm, nsigma)
    (B) fixed bounds via (emin_keV, emax_keV)

Includes a Ce-134 / Ac-225 mixture-specific method:
- Estimate Ce from 1022 keV window using ce_eff_model
- Scale/subtract Ce template from mixture by matching 1022-window counts
- Estimate Ac from residual using ac_eff_model (218 or 440 keV window)

The CutCountEstimate schema mirrors MLEEstimate for notebook comparability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .types import EfficiencyModel
from .efficiency import evaluate_efficiency_model

LIVETIME_S: float = 60.0


# -------------------------
# ROI window specification
# -------------------------

@dataclass(frozen=True)
class EnergyWindowSpec:
    """
    Defines an integration ROI.

    Specify either:
      - center_keV + resolution_pct_fwhm + nsigma (peak-centered), OR
      - emin_keV + emax_keV (fixed bounds)
    """
    # peak-centered
    center_keV: Optional[float] = None
    resolution_pct_fwhm: Optional[float] = None
    nsigma: float = 2.0

    # fixed bounds
    emin_keV: Optional[float] = None
    emax_keV: Optional[float] = None

    def bounds(self) -> tuple[float, float]:
        # fixed bounds path
        if self.emin_keV is not None or self.emax_keV is not None:
            if self.emin_keV is None or self.emax_keV is None:
                raise ValueError("If using fixed bounds, both emin_keV and emax_keV must be set.")
            lo = float(self.emin_keV)
            hi = float(self.emax_keV)
            if not (hi > lo):
                raise ValueError(f"Fixed ROI bounds must satisfy emax>emin. Got ({lo}, {hi}).")
            return (lo, hi)

        # peak-centered path
        if self.center_keV is None or self.resolution_pct_fwhm is None:
            raise ValueError(
                "EnergyWindowSpec must be specified either by (emin_keV, emax_keV) "
                "or by (center_keV, resolution_pct_fwhm, nsigma)."
            )
        c = float(self.center_keV)
        hw = half_width_from_resolution_pct(c, float(self.resolution_pct_fwhm), nsigma=float(self.nsigma))
        return (c - hw, c + hw)

    def describe(self) -> dict[str, Any]:
        lo, hi = self.bounds()
        return {
            "center_keV": self.center_keV,
            "resolution_pct_fwhm": self.resolution_pct_fwhm,
            "nsigma": float(self.nsigma),
            "emin_keV": self.emin_keV,
            "emax_keV": self.emax_keV,
            "bounds_keV": [float(lo), float(hi)],
        }


# -------------------------
# Estimate container (MLE-like)
# -------------------------

@dataclass
class CutCountEstimate:
    experiment_id: str
    label: str
    activities_bq: dict[str, float]
    uncertainties_bq: Optional[dict[str, float]]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "label": self.label,
            "activities_bq": dict(self.activities_bq),
            "uncertainties_bq": dict(self.uncertainties_bq) if self.uncertainties_bq is not None else None,
            "diagnostics": dict(self.diagnostics),
        }


# -------------------------
# Window utilities
# -------------------------

def half_width_from_resolution_pct(E_keV: float, resolution_pct_fwhm: float, nsigma: float = 2.0) -> float:
    fwhm = (float(resolution_pct_fwhm) / 100.0) * float(E_keV)
    sigma = fwhm / 2.355
    return float(nsigma) * float(sigma)


def integrate_counts_in_bounds(x_keV: np.ndarray, y_counts: np.ndarray, bounds_keV: tuple[float, float]) -> float:
    x = np.asarray(x_keV, dtype=float).reshape(-1)
    y = np.asarray(y_counts, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x_keV and y_counts must have same shape. Got {x.shape} vs {y.shape}.")
    lo, hi = float(bounds_keV[0]), float(bounds_keV[1])
    mask = (x >= lo) & (x <= hi)
    return float(np.sum(y[mask]))


def subtract_scaled_template(y_mix: np.ndarray, y_template: np.ndarray, scale: float) -> np.ndarray:
    y_mix = np.asarray(y_mix, dtype=float).reshape(-1)
    y_template = np.asarray(y_template, dtype=float).reshape(-1)
    if y_mix.shape != y_template.shape:
        raise ValueError(f"Mixture and template must have same shape. Got {y_mix.shape} vs {y_template.shape}.")
    return np.clip(y_mix - float(scale) * y_template, 0.0, None)


# -------------------------
# Efficiency inversion (counts -> activity)
# -------------------------

def _assert_effmodel_counts60(model: EfficiencyModel) -> None:
    fs = (model.meta or {}).get("fit_spec", {}) if isinstance(model.meta, dict) else {}
    ru = fs.get("rate_units", None)
    lt = fs.get("livetime_s", None)

    if ru is not None and str(ru).lower() != "counts":
        raise ValueError(
            f"Cutcount expects EfficiencyModel fit with rate_units='counts'. Got rate_units={ru!r} for model={model.name!r}."
        )
    if lt is not None and abs(float(lt) - LIVETIME_S) > 1e-6:
        raise ValueError(
            f"Cutcount expects EfficiencyModel fit with livetime_s={LIVETIME_S}. Got livetime_s={lt!r} for model={model.name!r}."
        )


def activity_from_window_counts(model: EfficiencyModel, counts: float) -> float:
    """
    Analytic inversion for the models used in efficiency.py.

    efficiency.py parameterization (see evaluate_efficiency_model):
      - linear:           y = a*x + b
      - nonparalyzable:   y = (a*x) / (1 + b*x)
    """
    _assert_effmodel_counts60(model)

    y = float(counts)
    a = float(model.params["a"])
    b = float(model.params["b"])

    if model.model_type == "linear":
        if abs(a) < 1e-18:
            raise ZeroDivisionError(f"Linear model has near-zero slope a={a}. Cannot invert.")
        return (y - b) / a

    if model.model_type == "nonparalyzable":
        denom = a - b * y
        if denom <= 0:
            raise ValueError(
                f"Cannot invert nonparalyzable model for counts={y:.3g}: a - b*y = {denom:.3g} <= 0 "
                f"(model={model.name!r}, a={a:.6g}, b={b:.6g})."
            )
        return y / denom

    raise ValueError(f"Unknown model_type={model.model_type!r} for model={model.name!r}.")


def sigma_activity_from_window_counts(model: EfficiencyModel, counts: float, sigma_counts: float) -> float:
    _assert_effmodel_counts60(model)

    y = float(counts)
    sigy = float(sigma_counts)
    a = float(model.params["a"])
    b = float(model.params["b"])

    if model.model_type == "linear":
        if abs(a) < 1e-18:
            raise ZeroDivisionError(f"Linear model has near-zero slope a={a}. Cannot compute uncertainty.")
        return abs(1.0 / a) * sigy

    if model.model_type == "nonparalyzable":
        denom = a - b * y
        if denom <= 0:
            raise ValueError(f"Cannot compute nonparalyzable uncertainty: a - b*y = {denom} <= 0.")
        dA_dy = a / (denom * denom)   # derivative of y/(a - b y)
        return abs(dA_dy) * sigy

    raise ValueError(f"Unknown model_type={model.model_type!r} for model={model.name!r}.")


# -------------------------
# Primary estimators
# -------------------------

def estimate_single_isotope_cutcount(
    *,
    experiment_id: str,
    label: str,
    iso: str,
    x_keV: np.ndarray,
    y_counts: np.ndarray,
    window: EnergyWindowSpec,
    eff_model: EfficiencyModel,
    y_template: Optional[np.ndarray] = None,
    template_label: Optional[str] = None,
) -> tuple[CutCountEstimate, dict[str, Any]]:
    bounds = window.bounds()
    C = integrate_counts_in_bounds(x_keV, y_counts, bounds)
    sigma_C = float(np.sqrt(max(C, 0.0)))  # Poisson

    A = activity_from_window_counts(eff_model, C)
    sigma_A = sigma_activity_from_window_counts(eff_model, C, sigma_C)

    extra: dict[str, Any] = {
        "window": window.describe(),
        "counts_window": float(C),
        "sigma_counts_window": float(sigma_C),
        "eff_model_name": eff_model.name,
        "eff_model_type": eff_model.model_type,
        "eff_model_params": dict(eff_model.params),
    }

    if y_template is not None:
        C_tpl = integrate_counts_in_bounds(x_keV, y_template, bounds)
        scale = (C / C_tpl) if C_tpl > 0 else 0.0
        extra.update({
            "template_label": template_label,
            "counts_template_window": float(C_tpl),
            "scale_template_to_sample_in_window": float(scale),
        })

    est = CutCountEstimate(
        experiment_id=str(experiment_id),
        label=str(label),
        activities_bq={str(iso): float(A)},
        uncertainties_bq={str(iso): float(sigma_A)},
        diagnostics={
            "method": "cutcount_single_v1",
            "livetime_s": LIVETIME_S,
            "run_config": {
                "method": "cutcount_single_v1",
                "iso": str(iso),
                "window": window.describe(),
                "eff_model": {"name": eff_model.name, "type": eff_model.model_type},
            },
            "extra": extra,
        },
    )
    return est, extra


@dataclass(frozen=True)
class CeAcCutCountConfig:
    method_tag: str = "cutcount_ceac_v2"
    roi_keV: tuple[float, float] = (200.0, 1250.0)

    ce_window: EnergyWindowSpec = EnergyWindowSpec(center_keV=1022.0, resolution_pct_fwhm=8.0, nsigma=2.0)
    ac_window: EnergyWindowSpec = EnergyWindowSpec(center_keV=218.0, resolution_pct_fwhm=8.0, nsigma=2.0)

    ce_iso: str = "ce134"
    ac_iso: str = "ac225"


def estimate_ceac_mixture_cutcount(
    *,
    experiment_id: str,
    label: str,
    x_keV: np.ndarray,
    y_mix: np.ndarray,
    y_ce_template: np.ndarray,
    ce_eff_model: EfficiencyModel,
    ac_eff_model: EfficiencyModel,
    config: CeAcCutCountConfig = CeAcCutCountConfig(),
    ce_template_label: Optional[str] = None,
) -> tuple[CutCountEstimate, dict[str, Any], dict[str, np.ndarray]]:
    x = np.asarray(x_keV, dtype=float).reshape(-1)
    y_mix = np.asarray(y_mix, dtype=float).reshape(-1)
    y_tpl = np.asarray(y_ce_template, dtype=float).reshape(-1)
    if not (x.shape == y_mix.shape == y_tpl.shape):
        raise ValueError(f"x, y_mix, y_ce_template must have same shape. Got {x.shape}, {y_mix.shape}, {y_tpl.shape}.")

    # Ce
    ce_bounds = config.ce_window.bounds()
    C_ce = integrate_counts_in_bounds(x, y_mix, ce_bounds)
    sigma_C_ce = float(np.sqrt(max(C_ce, 0.0)))
    A_ce = activity_from_window_counts(ce_eff_model, C_ce)
    sigma_A_ce = sigma_activity_from_window_counts(ce_eff_model, C_ce, sigma_C_ce)

    C_ce_tpl = integrate_counts_in_bounds(x, y_tpl, ce_bounds)
    scale_ce = (C_ce / C_ce_tpl) if C_ce_tpl > 0 else 0.0
    y_ce_scaled = scale_ce * y_tpl

    # residual
    y_res = subtract_scaled_template(y_mix, y_tpl, scale=scale_ce)

    # Ac
    ac_bounds = config.ac_window.bounds()
    C_ac = integrate_counts_in_bounds(x, y_res, ac_bounds)
    sigma_C_ac = float(np.sqrt(max(C_ac, 0.0)))
    A_ac = activity_from_window_counts(ac_eff_model, C_ac)
    sigma_A_ac = sigma_activity_from_window_counts(ac_eff_model, C_ac, sigma_C_ac)

    extra: dict[str, Any] = {
        "ce_window": config.ce_window.describe(),
        "ac_window": config.ac_window.describe(),
        "counts_ce_window_mix": float(C_ce),
        "sigma_counts_ce_window_mix": float(sigma_C_ce),
        "counts_ce_window_template": float(C_ce_tpl),
        "scale_ce_template": float(scale_ce),
        "counts_ac_window_residual": float(C_ac),
        "sigma_counts_ac_window_residual": float(sigma_C_ac),
        "ce_eff_model": {"name": ce_eff_model.name, "type": ce_eff_model.model_type, "params": dict(ce_eff_model.params)},
        "ac_eff_model": {"name": ac_eff_model.name, "type": ac_eff_model.model_type, "params": dict(ac_eff_model.params)},
        "ce_template_label": ce_template_label,
    }

    run_config = {
        "method": config.method_tag,
        "livetime_s": LIVETIME_S,
        "roi_keV": list(map(float, config.roi_keV)),
        "windows": {"ce": config.ce_window.describe(), "ac": config.ac_window.describe()},
        "isos": {"ce": config.ce_iso, "ac": config.ac_iso},
        "eff_models": {
            "ce": {"name": ce_eff_model.name, "type": ce_eff_model.model_type},
            "ac": {"name": ac_eff_model.name, "type": ac_eff_model.model_type},
        },
        "templates": {"ce": {"label": ce_template_label}},
    }

    est = CutCountEstimate(
        experiment_id=str(experiment_id),
        label=str(label),
        activities_bq={config.ce_iso: float(A_ce), config.ac_iso: float(A_ac)},
        uncertainties_bq={config.ce_iso: float(sigma_A_ce), config.ac_iso: float(sigma_A_ac)},
        diagnostics={"method": config.method_tag, "livetime_s": LIVETIME_S, "run_config": run_config, "extra": extra},
    )

    fit_arrays = {"x_keV": x, "y_mix": y_mix, "y_ce_scaled": y_ce_scaled, "y_residual": y_res}

    return est, extra, fit_arrays
