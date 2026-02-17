"""
io_mappings.py

Mapping layer: interpret raw Excel blocks (io_excel.ExcelRaw) into domain objects (types.Experiment).

This module answers:
- Where in the sheet are the spectra counts?
- Are spectra stored as rows=samples, cols=channels (or vice versa)?
- How do we derive sample labels / IDs?
- Where are optional weights/masses (biodistribution)?
- How do we assign mouse/tissue ordering when sheets are block-structured?

Design rules:
- NO calibration here.
- NO fitting / MLE here.
- This is *schema interpretation* and labeling.
"""

#Run this as a Sanity Check, for experimental data
# from gamma.io_excel import load_experiment_xls
# from gamma.io_mappings import ExperimentMapping, apply_mapping

# raw = load_experiment_xls("../data/ANIL_AC.xlsx", sheet_name=0)

# mapping = ExperimentMapping(
#     experiment_id="ANIL_AC",
#     samples_are_rows=True,
#     # if you need to cut off non-spectral columns:
#     channel_col_slice=slice(0, 4096),
#     labels=[...],  # or label_fn=...
# )

# exp = apply_mapping(raw, mapping)
# exp.counts.shape, exp.labels[:5]

#Run this as a Sanity Check, for biodistribution data
# from gamma.io_excel import load_biodistribution_xls
# from gamma.io_mappings import ExperimentMapping, PatternSpec, label_fn_from_pattern, apply_mapping

# raw = load_biodistribution_xls("../data/WENJO_BIOD.xlsx", sheet_name=0)

# pattern = PatternSpec(
#     mice=[f"mouse{i}" for i in range(1, 7)],
#     tissues=["blood","liver","kidney"],
#     order="mouse_major"
# )

# mapping = ExperimentMapping(
#     experiment_id="WENJO_BIOD",
#     samples_are_rows=True,
#     label_fn=label_fn_from_pattern(pattern),
#     weights_block="weights",
#     weights_col=0,
# )

# exp = apply_mapping(raw, mapping)


from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .io_excel import ExcelRaw, df_to_numpy
from .types import Experiment


# ----------------------------
# Core mapping specs
# ----------------------------

@dataclass(frozen=True)
class ExperimentMapping:
    """
    How to map one ExcelRaw into one Experiment.

    Required concept:
    - which block contains the spectra
    - orientation: whether samples are rows or columns
    - where the channel axis is
    - how to label samples (explicit list or a function)

    Optional:
    - weights/masses block (biodistribution)
    - sample weights transform (e.g., grams -> mg, or to normalize)
    """
    experiment_id: str

    spectra_block: str = "spectra"

    # If True: each ROW is a sample spectrum, columns are channels.
    # If False: each COLUMN is a sample spectrum, rows are channels.
    samples_are_rows: bool = True #always True

    # If the spectra block contains non-spectral columns (labels, timestamps), specify slice.
    # Example: channels are columns 1..4096 -> channel_col_slice=slice(1, None)
    channel_col_slice: slice | None = None
    channel_row_slice: slice | None = None

    # Optional: if there is a header row/col in the numeric block you want dropped.
    drop_first_row: bool = False
    drop_first_col: bool = False

    # Labels: either provide explicit labels, or a function that computes them from the raw DataFrame.
    labels: Sequence[str] | None = None
    label_fn: Callable[[pd.DataFrame], list[str]] | None = None

    # Optional weights block (e.g. biodistribution masses)
    weights_block: str | None = "weights"
    weights_are_rows: bool = True
    weights_col: int | None = None  # if weights are in one column
    weights_row_slice: slice | None = None
    weights_col_slice: slice | None = None

    # Optional: postprocess weights (e.g., convert grams->g, or compute mg)
    weights_transform: Callable[[np.ndarray], np.ndarray] | None = None

    # Optional metadata to carry around
    meta: Mapping[str, object] | None = None


def _get_block(raw: ExcelRaw, name: str) -> pd.DataFrame:
    for blk in raw.blocks:
        if blk.name == name:
            return blk.df
    avail = [b.name for b in raw.blocks]
    raise KeyError(f"ExcelRaw has no block named {name!r}. Available blocks: {avail}")


def _apply_slices(
    arr: np.ndarray,
    *,
    row_slice: slice | None,
    col_slice: slice | None,
) -> np.ndarray:
    if row_slice is not None:
        arr = arr[row_slice, :]
    if col_slice is not None:
        arr = arr[:, col_slice]
    return arr


def _drop_edges(arr: np.ndarray, *, drop_first_row: bool, drop_first_col: bool) -> np.ndarray:
    if drop_first_row and arr.shape[0] > 0:
        arr = arr[1:, :]
    if drop_first_col and arr.shape[1] > 0:
        arr = arr[:, 1:]
    return arr


def _make_labels(mapping: ExperimentMapping, spectra_df: pd.DataFrame, n_samples: int) -> list[str]:
    if mapping.labels is not None:
        labels = list(mapping.labels)
        if len(labels) != n_samples:
            raise ValueError(
                f"Provided labels length {len(labels)} does not match n_samples {n_samples} "
                f"for experiment_id={mapping.experiment_id}"
            )
        return labels

    if mapping.label_fn is not None:
        labels = mapping.label_fn(spectra_df)
        if len(labels) != n_samples:
            raise ValueError(
                f"label_fn produced {len(labels)} labels but expected {n_samples} "
                f"for experiment_id={mapping.experiment_id}"
            )
        return labels

    # Default fallback: generic numeric labels
    return [f"sample_{i:03d}" for i in range(n_samples)]


# ----------------------------
# Public API: raw -> Experiment
# ----------------------------

def apply_mapping(raw: ExcelRaw, mapping: ExperimentMapping) -> Experiment:
    """
    Convert ExcelRaw to an Experiment using ExperimentMapping.

    Returns:
        Experiment with:
          - counts: shape (n_samples, n_channels)
          - labels: length n_samples
          - weights: optional length n_samples (or None)
          - meta: includes raw path, mapping meta, etc.
    """
    spectra_df = _get_block(raw, mapping.spectra_block)

    # Convert to numeric matrix
    mat = df_to_numpy(spectra_df, coerce_numeric=True)
    mat = _drop_edges(mat, drop_first_row=mapping.drop_first_row, drop_first_col=mapping.drop_first_col)

    # Slice down to channels if specified
    mat = _apply_slices(
        mat,
        row_slice=mapping.channel_row_slice,
        col_slice=mapping.channel_col_slice,
    )

    # Orient into (n_samples, n_channels)
    if mapping.samples_are_rows:
        counts = mat
    else:
        counts = mat.T

    if counts.ndim != 2:
        raise ValueError(f"Spectra matrix must be 2D; got shape {counts.shape} for {mapping.experiment_id}")

    n_samples = counts.shape[0]
    labels = _make_labels(mapping, spectra_df, n_samples)

    # Optional weights
    weights = None
    if mapping.weights_block is not None:
        try:
            wdf = _get_block(raw, mapping.weights_block)
        except KeyError:
            wdf = None

        if wdf is not None:
            warr = df_to_numpy(wdf, coerce_numeric=True)
            warr = _drop_edges(warr, drop_first_row=mapping.drop_first_row, drop_first_col=mapping.drop_first_col)

            warr = _apply_slices(
                warr,
                row_slice=mapping.weights_row_slice,
                col_slice=mapping.weights_col_slice,
            )

            if mapping.weights_col is not None:
                # Take a single column of weights
                if warr.ndim != 2 or mapping.weights_col >= warr.shape[1]:
                    raise ValueError(
                        f"weights_col={mapping.weights_col} invalid for weights array shape {warr.shape}"
                    )
                w = warr[:, mapping.weights_col]
            else:
                # If it’s a 1D-ish block, flatten
                w = warr.reshape(-1)

            if mapping.weights_are_rows is False:
                # if weights are actually stored in a row, transpose logic
                w = w.T.reshape(-1)

            if mapping.weights_transform is not None:
                w = mapping.weights_transform(w)

            # Safety: if mismatch, store but warn by raising (better early fail)
            if len(w) != n_samples:
                raise ValueError(
                    f"Parsed weights length {len(w)} does not match number of samples {n_samples} "
                    f"for experiment_id={mapping.experiment_id}"
                )

            weights = w.astype(float)

    meta = {
        "source_path": str(raw.path),
        "mapping_meta": dict(mapping.meta) if mapping.meta else {},
        "raw_meta": dict(raw.meta) if raw.meta else {},
    }

    return Experiment(
        experiment_id=mapping.experiment_id,
        counts=counts.astype(float),
        labels=labels,
        weights=weights,
        meta=meta,
    )


# ----------------------------
# Biodistribution / mice pattern utilities
# ----------------------------

@dataclass(frozen=True)
class PatternSpec:
    """
    Describe repeating label patterns for block-structured sheets.

    Example:
      tissues = ("blood", "kidney", "liver", ...)
      mice = ("mouse1", "mouse2", ...)
      order says how to interleave, e.g. tissue-major or mouse-major.

    This is intentionally generic: you can implement your exact notebook logic here.
    """
    mice: Sequence[str]
    tissues: Sequence[str]
    # "mouse_major": mouse1 tissue1..tissueN, then mouse2...
    # "tissue_major": tissue1 mouse1..mouseM, then tissue2...
    order: str = "mouse_major"

    def generate_labels(self) -> list[str]:
        labels: list[str] = []
        if self.order == "mouse_major":
            for m in self.mice:
                for t in self.tissues:
                    labels.append(f"{m}:{t}")
        elif self.order == "tissue_major":
            for t in self.tissues:
                for m in self.mice:
                    labels.append(f"{m}:{t}")
        else:
            raise ValueError(f"Unknown order={self.order!r}")
        return labels


def label_fn_from_pattern(pattern: PatternSpec) -> Callable[[pd.DataFrame], list[str]]:
    """
    Returns a label_fn compatible with ExperimentMapping.label_fn, using a PatternSpec.
    """
    def _fn(_df: pd.DataFrame) -> list[str]:
        return pattern.generate_labels()
    return _fn
