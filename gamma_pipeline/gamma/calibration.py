"""
calibration.py

Calibration stage (Step 2).

Responsibilities:
- Convert raw spectra (counts vs channel) into calibrated spectra (counts vs energy)
  using a chosen calibration method.
- Return calibrated objects + metadata, WITHOUT doing efficiency fitting or MLE.

Non-responsibilities:
- No Excel reading (io_excel.py)
- No sheet mapping / labeling (io_mappings.py)
- No efficiency curve fitting (efficiency.py)
- No MLE (mle.py)
- No persistence (artifacts.py), though we return metadata needed for saving.
"""

from __future__ import annotations
import becquerel as bq

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import Experiment, CalibrationResult, CalibrationSpec


# ----------------------------
# Public API
# ----------------------------

def calibrate_experiment(
    exp: Experiment,
    spec: CalibrationSpec,
    *,
    progress: bool = True,
) -> CalibrationResult:
    """
    Calibrate every sample spectrum in an Experiment.

    Parameters
    ----------
    exp:
        Experiment with exp.counts shape (n_samples, n_channels) and exp.labels.
    spec:
        CalibrationSpec with method + parameters.
    progress:
        If True, prints periodic progress messages (keeps it notebook-friendly).

    Returns
    -------
    CalibrationResult
        Contains:
          - experiment_id
          - calibrated: dict[label -> calibrated spectrum object]
          - meta: dict with calibration spec and any per-spectrum notes (fallbacks, fit quality)
    """
    counts = exp.counts
    labels = exp.labels

    if counts.ndim != 2:
        raise ValueError(f"exp.counts must be 2D; got shape={counts.shape}")
    if len(labels) != counts.shape[0]:
        raise ValueError(
            f"len(exp.labels)={len(labels)} must match n_samples={counts.shape[0]}"
        )

    calibrated: Dict[str, Any] = {}
    per_sample_meta: Dict[str, Dict[str, Any]] = {}

    n = counts.shape[0]
    for i, label in enumerate(labels):
        if progress and (i % max(1, n // 10) == 0):
            print(f"[calibrate_experiment] {exp.experiment_id}: {i}/{n} ...")

        raw_counts = counts[i, :]

        cal_obj, meta_i = calibrate_spectrum(
            raw_counts,
            spec,
            label=label,
            experiment_id=exp.experiment_id,
        )

        calibrated[label] = cal_obj
        per_sample_meta[label] = meta_i

    meta = {
        "experiment_id": exp.experiment_id,
        "calibration_spec": spec.to_dict() if hasattr(spec, "to_dict") else asdict(spec),
        "per_sample": per_sample_meta,
        "source_meta": exp.meta,
    }

    return CalibrationResult(
        experiment_id=exp.experiment_id,
        calibrated=calibrated,
        meta=meta,
    )


def calibrate_spectrum(
    counts: np.ndarray,
    spec: CalibrationSpec,
    *,
    label: str,
    experiment_id: str,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Calibrate a single spectrum.

    Parameters
    ----------
    counts:
        1D array of counts indexed by channel.
    spec:
        CalibrationSpec controlling method/parameters.
    label:
        Sample label for metadata.
    experiment_id:
        For error messages / provenance.

    Returns
    -------
    (calibrated_object, meta)
        - calibrated_object: usually a becquerel.Spectrum (or your Spectrum wrapper)
        - meta: dict with fit coefficients, fallback flags, goodness-of-fit, etc.

    Notes
    -----
    This function is deliberately a thin wrapper over your existing implementation.
    You will paste your current "calibrate_spectrum_" logic into _calibrate_impl(...) below.
    """
    if counts.ndim != 1:
        raise ValueError(
            f"counts must be 1D for {experiment_id}:{label}; got shape={counts.shape}"
        )

    try:
        cal_obj, meta = _calibrate_impl(counts, spec)
    except Exception as e:
        raise RuntimeError(
            f"Calibration failed for {experiment_id}:{label} using method={spec.method!r}: {e}"
        ) from e

    # Ensure meta always has identifying info
    meta = dict(meta) if meta is not None else {}
    meta.setdefault("experiment_id", experiment_id)
    meta.setdefault("label", label)
    meta.setdefault("method", spec.method)

    return cal_obj, meta


# ----------------------------
# CalibrationSpec + CalibrationResult live in types.py
# but if you haven't implemented them yet, here is the target shape:
# ----------------------------

# Your types.py should define something like:
#
# @dataclass(frozen=True)
# class CalibrationSpec:
#     method: str  # e.g. "peakfit", "fallback_linear", ...
#     params: dict[str, Any]  # method-specific parameters
#
# @dataclass
# class CalibrationResult:
#     experiment_id: str
#     calibrated: dict[str, Any]   # label -> calibrated spectrum (becquerel Spectrum)
#     meta: dict[str, Any]
#
# For now, calibration.py assumes these exist and imports them from .types.





# ----------------------------
# Implementation hook (PASTE YOUR EXISTING CODE HERE)
# ----------------------------

def calibrate_spectrum_legacy(
    spectrum,
    expected_energies,
    kernel=(500, 50),
    min_snr=1.2,
    xmin=200,
    livetime=60.0,
    tolerance=75,
    method="fit",  # "fit" or "interpolation"
    fit_order=1,   # 1 = linear, 2 = quadratic, ...
    fallback_calibration=None,
    force_calibration=None,
    debug = False,
):
    """
    General spectrum calibration routine.
    
    Parameters
    ----------
    spectrum : array
        Raw counts per channel.
    expected_energies : list of float
        Known calibration gamma energies [keV].
    kernel : tuple
        Parameters for GaussianPeakFilter.
    min_snr : float
        Minimum SNR for peak finding.
    xmin : int
        Minimum channel to consider for peak finding.
    livetime : float
        Spectrum live time (s).
    tolerance : int or dict
        Allowed deviation between found peaks and expected energies.
        If int, same window is used for all peaks.
        If dict, must map energy -> (low_tolerance, high_tolerance).
    method : str
        "fit" uses Calibration.from_points with polynomial regression,
        "interpolation" uses Calibration.from_interpolation.
    fit_order : int
        Polynomial order for "fit" mode.
    """
    
    # Build spectrum
    spec = bq.Spectrum(counts=spectrum, livetime=livetime)

    # Peak finding
    finder = bq.PeakFinder(spec, bq.GaussianPeakFilter(*kernel, fwhm_at_0=10))
    finder.find_peaks(min_snr=min_snr, xmin=xmin)
    found_peaks = finder.centroids

    
    # Match peaks
    matched_channels = []
    matched_energies = []

    for E in expected_energies:
        if isinstance(tolerance, dict):
            low_tol, high_tol = tolerance.get(E, (50, 50))
        else:
            low_tol, high_tol = tolerance, tolerance

        diffs = np.array(found_peaks) - E
        if diffs.size == 0:
            continue
        # Check nearest peak within tolerance
        nearest_idx = np.argmin(np.abs(diffs))
        if diffs[nearest_idx] < 0 and abs(diffs[nearest_idx]) < low_tol:
            matched_channels.append(found_peaks[nearest_idx])
            matched_energies.append(E)
        elif diffs[nearest_idx] >= 0 and abs(diffs[nearest_idx]) < high_tol:
            matched_channels.append(found_peaks[nearest_idx])
            matched_energies.append(E)

    # Require at least 2 matched peaks
    # print(matched_channels)
    # print(matched_energies)
    
    # if len(matched_channels) < 2:
    #     raise ValueError("Not enough calibration peaks found.")

    if debug:
        print(f"These are the matched channels: {matched_channels}")
        print(f"These are the matched energies: {matched_energies}")

    if force_calibration:
        spec.apply_calibration(force_calibration)
        return spec, force_calibration, (found_peaks, matched_channels, matched_energies, 'Forced calibration')

    if len(matched_channels) < 2:
        if fallback_calibration:
            spec.apply_calibration(fallback_calibration)
            return spec, fallback_calibration, (found_peaks, matched_channels, matched_energies, 'Fallback calibration')
        else:
            raise ValueError("Not enough calibration peaks found and no fallback calibration provided.")

    if len(matched_energies) >= 3:
        fit_order = 2

    # Build calibration
    if method == "fit":
        poly_str = " + ".join([f"p[{i}]*x**{i}" for i in range(fit_order+1)])
        cal = bq.Calibration.from_points(
            poly_str, matched_channels, matched_energies
        )
    elif method == "interpolation":
        cal = bq.Calibration.from_interpolation(matched_channels, matched_energies)
    else:
        raise ValueError("Unknown method: choose 'fit' or 'interpolation'")

    try:
        spec.apply_calibration(cal)
        # Apply calibration
        # print(cal)
        return spec, cal, (found_peaks, matched_channels, matched_energies, 'Normal calibration')

    except Exception as E:
        print(E)
        if fallback_calibration:
            spec.apply_calibration(fallback_calibration)
            return spec, fallback_calibration, (found_peaks, matched_channels, matched_energies, 'Fallback calibration')
        else:
            raise ValueError(f"Calibration failed because {E}. No fallback calibration provided.")


def _calibrate_impl(counts: np.ndarray, spec: CalibrationSpec) -> Tuple[Any, Dict[str, Any]]:
    """
    Internal implementation for calibration.

    Replace this stub with your real calibration function(s).

    What you should paste here:
    - Your current calibrate_spectrum_(...) logic from the notebook,
      but adapted to accept (counts, spec) rather than a pile of globals.

    Suggested contract:
      returns (calibrated_spectrum_object, meta_dict)

    Where:
      calibrated_spectrum_object: likely a becquerel.Spectrum with proper energy calibration
      meta_dict: fit coefficients, identified peaks, fallback used, chi2, etc.
    """


    p = spec.params  # shorthand

    # Required parameter for your calibration
    expected_energies = p["expected_energies"]

    # Optional parameters with defaults (matching your legacy function defaults)
    kernel = p.get("kernel", (500, 50))
    min_snr = p.get("min_snr", 1.2)
    xmin = p.get("xmin", 200)
    livetime = p.get("livetime", 60.0)
    tolerance = p.get("tolerance", 75)
    method = p.get("method", "fit")
    fit_order = p.get("fit_order", 1)
    fallback_calibration = p.get("fallback_calibration", None)
    force_calibration = p.get("force_calibration", None)
    debug = p.get("debug", False)

    # Call your existing calibration code
    cal_spec, cal, meta_tuple = calibrate_spectrum_legacy(
        counts,
        expected_energies=expected_energies,
        kernel=kernel,
        min_snr=min_snr,
        xmin=xmin,
        livetime=livetime,
        tolerance=tolerance,
        method=method,
        fit_order=fit_order,
        fallback_calibration=fallback_calibration,
        force_calibration=force_calibration,
        debug=debug,
    )

    # Ensure we always return (cal_obj, meta_dict)
    # if meta is None:
    meta = {}
    found_peaks, matched_channels, matched_energies, cal_method = meta_tuple
    # meta = dict(meta)
    meta.update({
        "kernel": kernel,
        "min_snr": min_snr,
        "xmin": xmin,
        "found_peaks" : list(map(float, np.atleast_1d(found_peaks))),
        "matched_channels": list(map(float, matched_channels)),
        "matched_energies": list(map(float, matched_energies)),
        "livetime": livetime,
        "tolerance": tolerance,
        "cal_method": cal_method,
        "fit_order": fit_order,
    })
    return cal_spec, meta

    # method = spec.method

    # # --- Example placeholder behavior ---
    # # This is NOT real calibration; it just passes through counts.
    # # Replace with your actual logic.
    # if method == "identity":
    #     # Treat channel index as "energy" (not physically meaningful, just a stub)
    #     meta = {"note": "identity calibration (placeholder)"}
    #     return {"counts": counts.copy(), "energy_axis": np.arange(len(counts))}, meta

    # raise NotImplementedError(
    #     f"_calibrate_impl is a stub. Implement calibration for method={method!r} "
    #     "by pasting your notebook calibration code here."
    # )
