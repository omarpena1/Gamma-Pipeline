# How this connects to the rest of my modules:

# In 02_calibrate.ipynb you’ll create a spec like:
# from gamma.types import CalibrationSpec
# from gamma.calibration import calibrate_experiment

# spec = CalibrationSpec(
#     method="peakfit_v2",
#     params={
#         "emin": 120,
#         "emax": 2000,
#         # You can store numpy arrays too, but for saving later prefer plain lists.
#         "es": np.linspace(120, 2000, 2000),
#         # plus whatever your current calibrate_spectrum_ expects
#     }
# )

# cal_res = calibrate_experiment(exp, spec)

# In mle.py your run_mle(...) can return an MLEEstimate.
# In efficiency.py your fit function returns an EfficiencyModel.


"""
types.py

Core domain types for the gamma pipeline.

These objects are meant to be:
- simple (dataclasses)
- explicit (no hidden global state)
- serializable (at least via .to_dict() for metadata + parameters)

We avoid putting "big logic" here. Logic lives in calibration.py, efficiency.py, mle.py, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


# ----------------------------
# Utilities
# ----------------------------

def _as_float_array(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be array-like; got scalar {x!r}")
    return arr


# ----------------------------
# Step 1: Ingested experiment
# ----------------------------

@dataclass
class Experiment:
    """
    Standardized representation of an experiment after Excel ingest + mapping.

    counts:
        2D array shaped (n_samples, n_channels)
    labels:
        list of sample identifiers (length n_samples)
    weights:
        optional 1D array length n_samples (e.g., tissue masses)
    meta:
        arbitrary metadata (file path, detector id, live time, etc.)
    """
    experiment_id: str
    counts: np.ndarray
    labels: list[str]
    weights: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.counts = _as_float_array(self.counts, name="counts")
        if self.counts.ndim != 2:
            raise ValueError(f"counts must be 2D (n_samples, n_channels); got shape {self.counts.shape}")

        if len(self.labels) != self.counts.shape[0]:
            raise ValueError(
                f"len(labels)={len(self.labels)} must match n_samples={self.counts.shape[0]}"
            )

        if self.weights is not None:
            self.weights = _as_float_array(self.weights, name="weights").reshape(-1)
            if len(self.weights) != self.counts.shape[0]:
                raise ValueError(
                    f"len(weights)={len(self.weights)} must match n_samples={self.counts.shape[0]}"
                )

    @property
    def n_samples(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.counts.shape[1])

    def index_of(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError as e:
            raise KeyError(f"Label {label!r} not found in experiment {self.experiment_id}") from e

    def counts_for(self, label: str, *, as_list: bool = False) -> np.ndarray | list[float]:
        """
        Return the 1D counts vector for a given spectrum label.
        """
        i = self.index_of(label)
        y = np.asarray(self.counts[i, :], dtype=float).reshape(-1)
        return y.tolist() if as_list else y

    def spectrum(self, label: str) -> "SpectrumRecord":
        """
        Return a SpectrumRecord for one spectrum (counts + label + provenance).
        """
        i = self.index_of(label)
        w = None if self.weights is None else float(self.weights[i])
        return SpectrumRecord(
            experiment_id=self.experiment_id,
            label=str(label),
            counts=np.asarray(self.counts[i, :], dtype=float).reshape(-1),
            weight=w,
            meta=dict(self.meta or {}),
        )


# Optional but handy if you ever want a "single spectrum record"
@dataclass(frozen=True)
class SpectrumRecord:
    """
    A single raw spectrum pulled from an Experiment, with identifying info.

    counts is 1D length n_channels.
    """
    experiment_id: str
    label: str
    counts: np.ndarray
    weight: Optional[float] = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", _as_float_array(self.counts, name="counts").reshape(-1))


# ----------------------------
# Step 2: Calibration
# ----------------------------

@dataclass(frozen=True)
class CalibrationSpec:
    """
    Specification for calibration.

    method:
        short string identifying which calibration procedure to use
        (e.g., "peakfit_v2", "fallback_linear", "identity")
    params:
        method-specific parameters (emin/emax, peak list, fallback coefficients, etc.)
    """
    method: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # Shallow copy is usually enough; keep it simple
        return {"method": self.method, "params": dict(self.params)}


@dataclass
class CalibrationResult:
    """
    Result of calibrating an Experiment.

    calibrated:
        dict[label -> calibrated spectrum object]

        During migration you can store actual `becquerel.Spectrum` objects here.
        Later, if you want artifact-friendly storage, you can instead store a lightweight
        calibrated representation (counts + energy bins + metadata).
    meta:
        global + per-sample calibration details (fit coefficients, fallback flags, etc.)
    """
    experiment_id: str
    calibrated: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)

    def get(self, label: str) -> Any:
        try:
            return self.calibrated[label]
        except KeyError as e:
            raise KeyError(f"No calibrated spectrum for label {label!r} in {self.experiment_id}") from e

    @property
    def labels(self) -> list[str]:
        return list(self.calibrated.keys())


# ----------------------------
# Step 3: Efficiency / response model
# ----------------------------

@dataclass(frozen=True)
class EfficiencyModel:
    """
    Represents a detector response / efficiency mapping used by MLE.

    model_type:
        e.g. "nonparalyzable" or "linear" or "interp"
    params:
        fitted parameters for the model
    meta:
        additional info: energy window, which spectra used, fit quality, etc.
    """
    name: str
    model_type: str
    params: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "params": dict(self.params),
            "meta": dict(self.meta),
        }


# ----------------------------
# Step 4: MLE results
# ----------------------------

@dataclass
class MLEEstimate:
    """
    Output of an MLE run.

    activities_bq:
        dict of isotope -> estimated activity (Bq)
    uncertainties_bq:
        optional dict isotope -> uncertainty (Bq), if you compute it
    diagnostics:
        fit stats, NMSE, likelihood value, convergence flags, etc.
    """
    experiment_id: str
    label: str
    activities_bq: Dict[str, float]
    uncertainties_bq: Optional[Dict[str, float]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "label": self.label,
            "activities_bq": dict(self.activities_bq),
            "uncertainties_bq": dict(self.uncertainties_bq) if self.uncertainties_bq else None,
            "diagnostics": dict(self.diagnostics),
        }
