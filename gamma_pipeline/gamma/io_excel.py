"""
io_excel.py

Excel ingestion utilities.

Design rules:
- This module ONLY reads data from Excel into pandas/numpy structures.
- It does NOT decide "what is what" (labeling/mapping belongs in io_mappings.py).
- It does NOT calibrate anything (calibration.py).
- It does NOT run MLE (mle.py).

Typical flow:
    raw = read_excel_blocks(path, sheet="Sheet1", ...)
    exp = mapping.apply(raw)  # io_mappings.py turns raw blocks into an Experiment
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

#Run this as a Sanity Check
# from gamma.io_excel import load_experiment_xls
# from gamma.io_mappings import ExperimentMapping, apply_mapping_to_raw

# raw = load_experiment_xls("../data/ANIL_AC.xlsx", sheet_name=0)
# mapping = ExperimentMapping(...)  # we’ll define this next
# exp = apply_mapping_to_raw(raw, mapping)


# ----------------------------
# Types
# ----------------------------

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ExcelBlock:
    """
    Represents a rectangular block extracted from an Excel sheet.
    """
    name: str
    df: pd.DataFrame
    sheet: str


@dataclass(frozen=True)
class ExcelRaw:
    """
    Container for raw Excel content needed to build an Experiment.
    """
    path: Path
    blocks: tuple[ExcelBlock, ...]
    # Optional: raw metadata that might be useful later
    meta: Mapping[str, Any] | None = None


# ----------------------------
# Core helpers
# ----------------------------

def _coerce_path(path: PathLike) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Excel file not found: {p}")
    return p


def read_excel_sheet(
    path: PathLike,
    sheet_name: str | int = 0,
    *,
    header: int | None = None,
    usecols: str | Sequence[int] | None = None,
    skiprows: int | Sequence[int] | None = None,
    nrows: int | None = None,
    engine: str | None = None,
    dtype: Any | None = None,
) -> pd.DataFrame:
    """
    Read an Excel sheet into a DataFrame with controlled defaults.

    We default to header=None because many spectroscopy spreadsheets are "data blocks"
    rather than nicely labeled tables.
    """
    p = _coerce_path(path)

    try:
        df = pd.read_excel(
            p,
            sheet_name=sheet_name,
            header=header,
            usecols=usecols,
            skiprows=skiprows,
            nrows=nrows,
            engine=engine,   # let pandas pick if None
            dtype=dtype,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to read Excel file {p} sheet={sheet_name!r}: {e}"
        ) from e

    return df


def read_excel_block(
    path: PathLike,
    *,
    sheet_name: str | int = 0,
    name: str = "block",
    header: int | None = None,
    usecols: str | Sequence[int] | None = None,
    skiprows: int | Sequence[int] | None = None,
    nrows: int | None = None,
    engine: str | None = None,
    dtype: Any | None = None,
) -> ExcelBlock:
    """
    Read a single rectangular block from a sheet.
    """
    df = read_excel_sheet(
        path,
        sheet_name=sheet_name,
        header=header,
        usecols=usecols,
        skiprows=skiprows,
        nrows=nrows,
        engine=engine,
        dtype=dtype,
    )
    sheet_str = str(sheet_name)
    return ExcelBlock(name=name, df=df, sheet=sheet_str)


def read_excel_blocks(
    path: PathLike,
    blocks: Sequence[Mapping[str, Any]],
    *,
    meta: Mapping[str, Any] | None = None,
) -> ExcelRaw:
    """
    Read multiple blocks from an Excel file.

    Parameters
    ----------
    path:
        Excel file path.
    blocks:
        A sequence of dicts, each specifying a block to read. Each dict may contain:
            - name (str)
            - sheet_name (str|int)
            - header (int|None)
            - usecols (str|list[int]|None)
            - skiprows (int|list[int]|None)
            - nrows (int|None)
            - dtype (optional)
    meta:
        Optional metadata (detector id, experiment id, etc.) stored with the result.

    Returns
    -------
    ExcelRaw containing a tuple of ExcelBlock entries.
    """
    p = _coerce_path(path)
    out: list[ExcelBlock] = []
    for spec in blocks:
        if "sheet_name" not in spec:
            raise ValueError(f"Block spec missing 'sheet_name': {spec}")

        blk = read_excel_block(
            p,
            sheet_name=spec.get("sheet_name", 0),
            name=spec.get("name", "block"),
            header=spec.get("header", None),
            usecols=spec.get("usecols", None),
            skiprows=spec.get("skiprows", None),
            nrows=spec.get("nrows", None),
            engine=spec.get("engine", None),
            dtype=spec.get("dtype", None),
        )
        out.append(blk)

    return ExcelRaw(path=p, blocks=tuple(out), meta=meta)


# ----------------------------
# Convenience loaders
# ----------------------------

def load_experiment_xls(
    path: PathLike,
    *,
    sheet_name: str | int = 0,
    data_usecols: str | Sequence[int] | None = None,
    data_skiprows: int | Sequence[int] | None = None,
    data_nrows: int | None = None,
    header: int | None = None,
    engine: str | None = None,
) -> ExcelRaw:
    """
    Convenience loader for the most common case where your spectra live in one big block.

    Returns an ExcelRaw with a single block named 'spectra'.
    Interpretation (how to split samples vs channels, where livetime is, etc.)
    is handled by io_mappings.py.
    """
    raw = read_excel_blocks(
        path,
        blocks=[
            dict(
                name="spectra",
                sheet_name=sheet_name,
                header=header,
                usecols=data_usecols,
                skiprows=data_skiprows,
                nrows=data_nrows,
                engine=engine,
            )
        ],
        meta={"kind": "experiment"},
    )
    return raw


def load_biodistribution_xls(
    path: PathLike,
    *,
    sheet_name: str | int = 0,
    spectra_usecols: str | Sequence[int] | None = None,
    spectra_skiprows: int | Sequence[int] | None = None,
    spectra_nrows: int | None = None,
    weights_usecols: str | Sequence[int] | None = None,
    weights_skiprows: int | Sequence[int] | None = None,
    weights_nrows: int | None = None,
    header: int | None = None,
    engine: str | None = None,
) -> ExcelRaw:
    """
    Convenience loader for biodistribution spreadsheets, where you may have:
      - spectra block
      - optional weights/masses block

    You can pass None for weights_* args if the sheet doesn't contain weights,
    or if you want to map weights in a different way.

    Returns ExcelRaw with blocks:
        - 'spectra'
        - 'weights' (if weights_* provided)
    """
    blocks = [
        dict(
            name="spectra",
            sheet_name=sheet_name,
            header=header,
            usecols=spectra_usecols,
            skiprows=spectra_skiprows,
            nrows=spectra_nrows,
            engine=engine,
        )
    ]

    if any(v is not None for v in (weights_usecols, weights_skiprows, weights_nrows)):
        blocks.append(
            dict(
                name="weights",
                sheet_name=sheet_name,
                header=header,
                usecols=weights_usecols,
                skiprows=weights_skiprows,
                nrows=weights_nrows,
                engine=engine,
            )
        )

    raw = read_excel_blocks(path, blocks=blocks, meta={"kind": "biodistribution"})
    return raw


# ----------------------------
# Utilities (optional)
# ----------------------------

def df_to_numpy(df: pd.DataFrame, *, coerce_numeric: bool = True) -> np.ndarray:
    """
    Convert a DataFrame to a numpy array, optionally coercing values to numeric.

    Useful because Excel blocks may contain mixed types (empty cells, strings).
    """
    if not coerce_numeric:
        return df.to_numpy()

    # Coerce each column to numeric where possible; non-numeric -> NaN
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.to_numpy()
