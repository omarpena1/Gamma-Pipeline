# gamma/preprocess.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Union, Literal, Dict, Any
import re
import numpy as np
from .types import Experiment

ArrayLike = Union[np.ndarray, Sequence[float]]

SuppressBackground = Union[
    bool,
    Literal["auto"],
    Sequence[int],
    np.ndarray,
]

def _resolve_bkg_rows(
    exp: Experiment,
    suppress_background: SuppressBackground,
    *,
    bkg_label_pattern: str = r"^(0|0\.0)(?:_rep\d+)?$",
    # If you literally always use 0.0_repN, use: r"^0\.0_rep\d+$"
) -> np.ndarray:
    """
    Returns background row indices (possibly empty).
    """
    if suppress_background is False or suppress_background is None:
        return np.asarray([], dtype=int)

    if suppress_background == "auto":
        rx = re.compile(bkg_label_pattern)
        idx = [i for i, lab in enumerate(exp.labels) if rx.match(str(lab))]
        return np.asarray(idx, dtype=int)

    # Explicit indices
    idx = np.asarray(suppress_background, dtype=int)
    if idx.ndim != 1:
        raise ValueError("suppress_background indices must be a 1D sequence.")
    if idx.size == 0:
        return idx
    if idx.min(initial=0) < 0 or idx.max(initial=-1) >= len(exp.labels):
        raise ValueError("suppress_background contains indices out of range.")
    return idx


def _drop_rows(exp: Experiment, drop_idx: np.ndarray, *, id_suffix: Optional[str] = None) -> Experiment:
    if drop_idx.size == 0:
        return exp
    keep = np.ones(len(exp.labels), dtype=bool)
    keep[drop_idx] = False
    new_counts = exp.counts[keep, :]
    new_labels = [lab for i, lab in enumerate(exp.labels) if keep[i]]

    new_weights = None
    if exp.weights is not None:
        new_weights = exp.weights[keep].copy()

    new_id = exp.experiment_id if id_suffix is None else (exp.experiment_id + id_suffix)

    return Experiment(
        experiment_id=new_id,
        counts=new_counts,
        labels=new_labels,
        weights=new_weights,
        meta=dict(exp.meta or {}),
    )

def _build_bkg_from_rows(
    exp: Experiment,
    bkg_idx: np.ndarray,
    *,
    reduce: Literal["sum", "mean", "median"] = "sum",
) -> np.ndarray:
    """
    Reduce one or more background rows into a single background spectrum (n_channels,).
    """
    if bkg_idx.size == 0:
        raise ValueError("No background rows available to build background spectrum.")
    bkg_2d = np.asarray(exp.counts[bkg_idx, :], dtype=float)
    if reduce == "sum":
        return bkg_2d.sum(axis=0)
    if reduce == "mean":
        return bkg_2d.mean(axis=0)
    if reduce == "median":
        return np.median(bkg_2d, axis=0)
    raise ValueError(f"Unknown reduce='{reduce}'")


def subtract_background(
    exp: Experiment,
    bkg_counts: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
    *,
    # --- Optional "bank + assignment" mode ---
    bkg_bank: Optional[np.ndarray] = None,
    bkg_assign: Optional[Sequence[int]] = None,

    # --- Livetime scaling ---
    exp_livetime_s: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    bkg_livetime_s: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    clamp_nonnegative: bool = True,

    # --- New: optionally detect/drop embedded background rows ---
    suppress_background: SuppressBackground = False,
    bkg_label_pattern: str = r"^(0|0\.0)(?:_rep\d+)?$",
    embedded_bkg_reduce: Literal["sum", "mean", "median"] = "sum",
    # Policy: if suppress_background is active and user did NOT supply bkg_bank,
    # do we use embedded background rows to define bkg_counts automatically?
    use_embedded_background_if_available: bool = True,
) -> Experiment:
    """
    Subtract background(s) from sample spectra.

    If suppress_background is enabled, background rows can be identified by
    label pattern or explicit indices and removed from the returned Experiment.
    Optionally (default), those embedded background rows can be used to define
    the background spectrum (single) when bkg_bank is not provided.
    """
    n_spectra, n_channels = exp.counts.shape

    # 0) resolve embedded background rows (optional)
    bkg_rows = _resolve_bkg_rows(
        exp,
        suppress_background,
        bkg_label_pattern=bkg_label_pattern,
    )

    # 1) If requested and no bank provided, optionally build bkg_counts from embedded rows
    if (bkg_rows.size > 0) and (bkg_bank is None) and use_embedded_background_if_available:
        # Build a single background spectrum from the embedded rows
        bkg_counts = _build_bkg_from_rows(exp, bkg_rows, reduce=embedded_bkg_reduce)

    # --------------------------
    # Choose background per sample: bkg_per_sample -> (n_spectra, n_channels)
    # --------------------------
    mode = None

    if bkg_bank is not None or bkg_assign is not None:
        if bkg_bank is None or bkg_assign is None:
            raise ValueError("Provide both bkg_bank and bkg_assign for bank+assignment mode.")
        bkg_bank = np.asarray(bkg_bank, dtype=float)
        if bkg_bank.ndim != 2 or bkg_bank.shape[1] != n_channels:
            raise ValueError(f"bkg_bank must have shape (K, {n_channels}). Got {bkg_bank.shape}.")
        bkg_assign = np.asarray(bkg_assign, dtype=int)
        if bkg_assign.shape != (n_spectra,):
            raise ValueError(f"bkg_assign must have length {n_spectra}. Got {bkg_assign.shape}.")
        if bkg_assign.min(initial=0) < 0 or bkg_assign.max(initial=-1) >= bkg_bank.shape[0]:
            raise ValueError("bkg_assign contains indices outside bkg_bank range.")
        bkg_per_sample = bkg_bank[bkg_assign, :]
        mode = "bank_assignment"
    else:
        bkg_arr = np.asarray(bkg_counts, dtype=float)
        if bkg_arr.ndim == 1:
            if bkg_arr.shape[0] != n_channels:
                raise ValueError(f"bkg_counts has {bkg_arr.shape[0]} channels but exp has {n_channels}")
            bkg_per_sample = np.broadcast_to(bkg_arr.reshape(1, -1), (n_spectra, n_channels))
            mode = "single"
        elif bkg_arr.ndim == 2:
            if bkg_arr.shape != (n_spectra, n_channels):
                raise ValueError(
                    f"Per-sample bkg must have shape ({n_spectra}, {n_channels}). Got {bkg_arr.shape}."
                )
            bkg_per_sample = bkg_arr
            mode = "per_sample"
        else:
            raise ValueError("bkg_counts must be 1D (single) or 2D (per-sample).")

    # --------------------------
    # Compute per-sample scale -> (n_spectra, 1)
    # --------------------------
    scale_vec = np.ones((n_spectra, 1), dtype=float)

    if (exp_livetime_s is not None) and (bkg_livetime_s is not None):
        exp_lt = np.asarray(exp_livetime_s, dtype=float)
        bkg_lt = np.asarray(bkg_livetime_s, dtype=float)

        if exp_lt.ndim == 0:
            exp_lt_vec = np.full((n_spectra,), float(exp_lt))
        elif exp_lt.shape == (n_spectra,):
            exp_lt_vec = exp_lt
        else:
            raise ValueError("exp_livetime_s must be a scalar or length n_spectra.")

        if bkg_lt.ndim == 0:
            bkg_lt_vec = np.full((n_spectra,), float(bkg_lt))
        elif bkg_lt.shape == (n_spectra,):
            bkg_lt_vec = bkg_lt
        elif (mode == "bank_assignment") and (bkg_lt.ndim == 1) and (bkg_lt.shape[0] == bkg_bank.shape[0]):
            bkg_lt_vec = bkg_lt[np.asarray(bkg_assign, dtype=int)]
        else:
            raise ValueError(
                "bkg_livetime_s must be a scalar, length n_spectra, "
                "or (K,) matching bkg_bank in bank+assignment mode."
            )

        if np.any(bkg_lt_vec <= 0):
            raise ValueError("All background livetimes must be > 0.")
        scale_vec = (exp_lt_vec / bkg_lt_vec).reshape(-1, 1)

    # --------------------------
    # Subtract
    # --------------------------
    new_counts = exp.counts.astype(float) - scale_vec * bkg_per_sample
    if clamp_nonnegative:
        new_counts = np.maximum(new_counts, 0.0)

    # --------------------------
    # Metadata
    # --------------------------
    new_meta = dict(exp.meta or {})
    new_meta["background_subtraction"] = {
        "method": "subtract_scaled_background",
        "mode": mode,
        "exp_livetime_s": None if exp_livetime_s is None else "provided",
        "bkg_livetime_s": None if bkg_livetime_s is None else "provided",
        "clamp_nonnegative": clamp_nonnegative,
        "bkg_bank_size": None if bkg_bank is None else int(bkg_bank.shape[0]),
    }
    if bkg_rows.size > 0:
        new_meta["background_subtraction"]["embedded_background"] = {
            "suppress_background": suppress_background if suppress_background != "auto" else "auto",
            "bkg_label_pattern": bkg_label_pattern,
            "bkg_rows": bkg_rows.tolist(),
            "reduce": embedded_bkg_reduce,
            "used_to_define_background": bool((bkg_bank is None) and use_embedded_background_if_available),
        }

    exp_out = Experiment(
        experiment_id=exp.experiment_id + "__bkgsub",
        counts=new_counts,
        labels=list(exp.labels),
        weights=exp.weights.copy() if exp.weights is not None else None,
        meta=new_meta,
    )

    # 2) optionally drop the background rows from the returned object
    if bkg_rows.size > 0:
        # exp_out = _drop_rows(exp_out, bkg_rows)
        exp_out = _drop_rows(exp_out, bkg_rows, id_suffix="__nobkgrows")


    return exp_out


# How you’d call it (your intended workflow)

# Case 1: Embedded background rows exist; auto-detect by label; subtract and remove them

# exp_clean = subtract_background(
#     exp,
#     bkg_counts=[0]*exp.n_channels,   # ignored because we use embedded background
#     suppress_background="auto",      # finds labels like "0.0" or "0.0_repN"
#     embedded_bkg_reduce="sum",       # combine multiple 0.0_repN into one background
# )


# Case 2: Explicit background row indices (more “surgical”)

# exp_clean = subtract_background(
#     exp,
#     bkg_counts=[0]*exp.n_channels,
#     suppress_background=[10, 11, 12],   # these rows are background
# )


# Case 3: You’re using an external background bank (your aggregated-spectra case)

# exp_clean = subtract_background(
#     exp,
#     bkg_counts=[0]*exp.n_channels,  # ignored in bank mode
#     bkg_bank=bkg_bank,
#     bkg_assign=bkg_assign,
#     suppress_background="auto",     # still allowed: drop embedded bkg rows if present
#     use_embedded_background_if_available=False,  # make sure we DON’T override bank mode intent
# )


@dataclass(frozen=True)
class BackgroundStripResult:
    exp_clean: Experiment
    bkg_rows: np.ndarray                 # (M,)
    bkg_spectrum: Optional[np.ndarray]   # (n_channels,) if built, else None


def strip_embedded_background(
    exp: Experiment,
    *,
    suppress_background: SuppressBackground = "auto",
    bkg_label_pattern: str = r"^(0|0\.0)(?:_rep\d+)?$",
    embedded_bkg_reduce: Literal["sum", "mean", "median"] = "sum",
    build_bkg_spectrum: bool = True,
    keep_id_suffix: str = "__nobkgrows",
) -> BackgroundStripResult:
    """
    Remove embedded background rows from an Experiment *without* doing subtraction.

    Intended use:
      - produce an Experiment safe for energy calibration / peak finding
      - optionally construct a single background spectrum from the dropped rows
        (useful later for subtraction, QC, plotting, etc.)

    Returns a BackgroundStripResult with:
      - exp_clean: Experiment without background rows
      - bkg_rows: indices of rows removed (possibly empty)
      - bkg_spectrum: reduced background spectrum if built, else None
    """
    bkg_rows = _resolve_bkg_rows(exp, suppress_background, bkg_label_pattern=bkg_label_pattern)

    bkg_spec = None
    if build_bkg_spectrum and (bkg_rows.size > 0):
        bkg_spec = _build_bkg_from_rows(exp, bkg_rows, reduce=embedded_bkg_reduce)

    exp_clean = exp if bkg_rows.size == 0 else _drop_rows(exp, bkg_rows, id_suffix=keep_id_suffix)

    # Make the provenance explicit
    new_meta = dict(exp_clean.meta or {})
    new_meta["embedded_background_stripping"] = {
        "suppress_background": suppress_background if suppress_background != "auto" else "auto",
        "bkg_label_pattern": bkg_label_pattern,
        "bkg_rows": bkg_rows.tolist(),
        "reduce": embedded_bkg_reduce,
        "built_bkg_spectrum": bool(build_bkg_spectrum and (bkg_rows.size > 0)),
    }

    # Ensure experiment_id suffix is consistent even if _drop_rows already applied one
    # (Your _drop_rows currently appends "__nobkgrows". This just normalizes behavior.)
    exp_id = exp_clean.experiment_id
    # if (bkg_rows.size > 0) and (keep_id_suffix not in exp_id):
    #     exp_id = exp.experiment_id + keep_id_suffix

    exp_clean2 = Experiment(
        experiment_id=exp_id,
        counts=exp_clean.counts,
        labels=list(exp_clean.labels),
        weights=exp_clean.weights.copy() if exp_clean.weights is not None else None,
        meta=new_meta,
    )

    return BackgroundStripResult(exp_clean=exp_clean2, bkg_rows=bkg_rows, bkg_spectrum=bkg_spec)