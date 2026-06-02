"""
artifacts.py

Persistent artifact store for intermediate pipeline products.

Goals:
- Make notebooks fast by reusing expensive intermediate results.
- Make your pipeline reproducible by saving specs + metadata alongside arrays.
- Work during migration even if some objects (e.g., becquerel.Spectrum) are hard to serialize.

Storage approach (simple + robust):
- Arrays/data -> .npz
- Metadata/specs -> .json
- Hard-to-serialize python objects -> .pkl (fallback)

Directory layout:
artifacts/
  experiments/<experiment_id>/
    experiment.npz
    meta.json
  calibrations/<experiment_id>/<calibration_tag>/
    meta.json
    calibrated_index.json
    calibrated/<label>.npz      (preferred lightweight)
    calibrated/<label>.pkl      (fallback)
  efficiency/<name>/
    model.json
  mle/<experiment_id>/
    <label>.json
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import hashlib
import json
import pandas as pd
from typing import Any, Mapping

import numpy as np

from .types import Experiment, CalibrationResult, EfficiencyModel, MLEEstimate, CalibrationSpec
from .cutcount import CutCountEstimate

# try:
#     from uncertainties import unumpy as unp
# except Exception:
#     unp = None


PathLike = Union[str, Path]


# ----------------------------
# Small JSON helpers
# ----------------------------

def _to_jsonable(x: Any) -> Any:
    """
    Convert common python/numpy objects into JSON-serializable forms.
    """
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        # Arrays in meta/specs: store as lists (ok for small arrays)
        return x.tolist()
    if is_dataclass(x):
        return _to_jsonable(asdict(x))
    # fallback: string repr
    return repr(x)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2, sort_keys=True)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


# ----------------------------
# Lightweight spectrum serialization (migration-friendly)
# ----------------------------

def _try_serialize_spectrum_light(obj: Any) -> Optional[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    """
    Best-effort conversion of a "calibrated spectrum object" into:
      - meta dict
      - arrays dict (to store in NPZ)

    This is intentionally conservative: if we can't confidently extract arrays,
    we return None and caller can pickle instead.

    If you're using becquerel.Spectrum:
      - It typically has attributes like .counts and bin edges / energies.
      - But API can vary; during migration we won't assume too much.
    """
    # Case 1: our placeholder dict form (from calibration stub)
    if isinstance(obj, dict) and "counts" in obj:
        counts = np.asarray(obj["counts"], dtype=float).reshape(-1)
        energy_axis = None
        if "energy_axis" in obj:
            energy_axis = np.asarray(obj["energy_axis"], dtype=float).reshape(-1)

        arrays = {"counts": counts}
        meta = {"schema": "dict_v1", "kind": "dict_spectrum"}
        if energy_axis is not None:
            arrays["energy_axis"] = energy_axis
            meta["axis"] = "energy_axis"
        else:
            meta["axis"] = "channel"

        return meta, arrays
        # raise TypeError("Dict-spectrum artifacts are not supported; expected becquerel Spectrum-like object.")

    # Case 2: attempt becquerel-like extraction (best effort, no hard dependency)
    counts_vals = getattr(obj, "counts_vals", None)
    bin_centers_kev = getattr(obj, "bin_centers_kev", None)

    if counts_vals is not None:
        # counts_vals should be numeric, but may be uncertainties objects in rare cases
        try:
            y = np.asarray(counts_vals, dtype=float).reshape(-1)
            arrays: Dict[str, np.ndarray] = {"counts_vals": y}
            meta = {"schema": "bq_v1", "kind": type(obj).__name__, "y_attr": "counts_vals"}
        except TypeError:
            # uncertainties support (optional)
            try:
                from uncertainties import unumpy as unp
            except Exception:
                return None  # fall back to pickle
            y_nom = np.asarray(unp.nominal_values(counts_vals), dtype=float).reshape(-1)
            y_std = np.asarray(unp.std_devs(counts_vals), dtype=float).reshape(-1)
            arrays = {"counts_vals": y_nom, "counts_std": y_std}
            meta = {"schema": "bq_v1", "kind": type(obj).__name__, "y_attr": "counts_vals"}

        # x-axis (bin centers in keV)
        if bin_centers_kev is not None:
            x = np.asarray(bin_centers_kev, dtype=float).reshape(-1)
            arrays["bin_centers_kev"] = x
            meta["x_attr"] = "bin_centers_kev"
            meta["x_units"] = "keV"
        else:
            # fall back: if no bin_centers_kev, keep channel index
            meta["x_attr"] = "channel_index"

        # optional: livetime if present
        lt = getattr(obj, "livetime", None)
        if lt is not None:
            try:
                meta["livetime"] = float(lt)
            except Exception:
                pass

        return meta, arrays

    # counts = getattr(obj, "counts", None)
    # if counts is not None:
    #     # First try the normal float conversion
    #     try:
    #         counts_arr = np.asarray(counts, dtype=float).reshape(-1)
    #         arrays: Dict[str, np.ndarray] = {"counts": counts_arr}
    #         meta: Dict[str, Any] = {"kind": type(obj).__name__}
    #     except TypeError:
    #         # Likely an uncertainties array (ufloat/Variable objects)
    #         if unp is None:
    #             return None  # fall back to pickling
    #         counts_nom = np.asarray(unp.nominal_values(counts), dtype=float).reshape(-1)
    #         counts_std = np.asarray(unp.std_devs(counts), dtype=float).reshape(-1)

    #         arrays = {"counts": counts_nom, "counts_std": counts_std}
    #         meta = {"kind": type(obj).__name__, "counts_has_uncertainty": True}

    #     # Try to find an axis as before
    #     for attr in ("energies", "energy", "bin_edges", "edges", "energy_bins"):
    #         axis = getattr(obj, attr, None)
    #         if axis is not None:
    #             try:
    #                 axis_arr = np.asarray(axis, dtype=float).reshape(-1)
    #                 arrays["axis"] = axis_arr
    #                 meta["axis_attr"] = attr
    #                 break
    #             except Exception:
    #                 pass

    #     return meta, arrays


    # Unknown object type
    return None


# ----------------------------
# ArtifactStore
# ----------------------------

class ArtifactStore:
    def __init__(self, root: PathLike = "artifacts") -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- paths ----

    def _exp_dir(self, experiment_id: str) -> Path:
        return self.root / "experiments" / experiment_id

    def _cal_dir(self, experiment_id: str, calibration_tag: str) -> Path:
        return self.root / "calibrations" / experiment_id / calibration_tag

    def _eff_dir(self, name: str) -> Path:
        return self.root / "efficiency" / name

    def _mle_dir(self, experiment_id: str) -> Path:
        return self.root / "mle" / experiment_id

    # ---- Experiment ----

    def save_experiment(self, exp: Experiment) -> None:
        d = self._exp_dir(exp.experiment_id)
        d.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            d / "experiment.npz",
            counts=np.asarray(exp.counts, dtype=float),
            labels=np.asarray(exp.labels, dtype=object),
            weights=np.asarray(exp.weights, dtype=float) if exp.weights is not None else None,
        )
        _write_json(d / "meta.json", {"experiment_id": exp.experiment_id, "meta": exp.meta})

    def load_experiment(self, experiment_id: str) -> Experiment:
        d = self._exp_dir(experiment_id)
        npz = np.load(d / "experiment.npz", allow_pickle=True)
        meta = _read_json(d / "meta.json")

        counts = npz["counts"]
        labels = [str(x) for x in npz["labels"].tolist()]
        weights = npz["weights"]
        if weights is None or (isinstance(weights, np.ndarray) and weights.shape == () and weights.item() is None):
            weights_arr = None
        else:
            weights_arr = np.asarray(weights, dtype=float).reshape(-1)

        return Experiment(
            experiment_id=experiment_id,
            counts=np.asarray(counts, dtype=float),
            labels=labels,
            weights=weights_arr,
            meta=dict(meta.get("meta", {})),
        )

    # added in the creation of aggregation.py

    def save_experiment_array(self, experiment_id: str, name: str, array: np.ndarray) -> None:
        d = self._exp_dir(experiment_id)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{name}.npz", data=np.asarray(array))

    def load_experiment_array(self, experiment_id: str, name: str) -> np.ndarray:
        d = self._exp_dir(experiment_id)
        npz = np.load(d / f"{name}.npz", allow_pickle=True)
        return np.asarray(npz["data"])

    # ---- CalibrationResult ----

    def save_calibration(
        self,
        cal: CalibrationResult,
        *,
        calibration_spec: CalibrationSpec,
        calibration_tag: str,
        prefer_lightweight: bool = True,
    ) -> None:
        """
        Save a CalibrationResult.

        calibration_tag:
            A short stable name for this calibration run, e.g.:
              - "peakfit_v2_emin120_emax2000"
              - "fallback_linear_v1"
            (You choose this; it becomes part of the path.)

        prefer_lightweight:
            If True, attempt to store each calibrated spectrum as NPZ arrays + JSON meta.
            If it can't, store as pickle.
        """
        d = self._cal_dir(cal.experiment_id, calibration_tag)
        (d / "calibrated").mkdir(parents=True, exist_ok=True)

        # Save global meta
        _write_json(
            d / "meta.json",
            {
                "experiment_id": cal.experiment_id,
                "calibration_tag": calibration_tag,
                "calibration_spec": calibration_spec.to_dict()
                if hasattr(calibration_spec, "to_dict")
                else asdict(calibration_spec),
                "meta": cal.meta,
            },
        )

        # Save each spectrum
        index: Dict[str, Dict[str, Any]] = {}
        for label, obj in cal.calibrated.items():
            entry: Dict[str, Any] = {"label": label}

            if prefer_lightweight:
                lite = _try_serialize_spectrum_light(obj)
            else:
                lite = None

            if lite is not None:
                meta_i, arrays_i = lite
                npz_path = d / "calibrated" / f"{_safe_filename(label)}.npz"
                json_path = d / "calibrated" / f"{_safe_filename(label)}.json"

                np.savez_compressed(npz_path, **arrays_i)
                _write_json(json_path, {"label": label, "spectrum_meta": meta_i})

                entry.update({"format": "npz", "npz": str(npz_path.name), "json": str(json_path.name)})
            else:
                pkl_path = d / "calibrated" / f"{_safe_filename(label)}.pkl"
                _write_pickle(pkl_path, obj)
                entry.update({"format": "pkl", "pkl": str(pkl_path.name)})

            index[label] = entry

        _write_json(d / "calibrated_index.json", {"index": index})

    def load_calibration(
        self,
        experiment_id: str,
        calibration_tag: str,
    ) -> CalibrationResult:
        d = self._cal_dir(experiment_id, calibration_tag)
        meta = _read_json(d / "meta.json")
        idx = _read_json(d / "calibrated_index.json")["index"]

        calibrated: Dict[str, Any] = {}
        for label, entry in idx.items():
            fmt = entry["format"]
            if fmt == "pkl":
                obj = _read_pickle(d / "calibrated" / entry["pkl"])
            # elif fmt == "npz":
            #     # Load lightweight arrays + meta; return as a simple dict for now
            #     npz = np.load(d / "calibrated" / entry["npz"], allow_pickle=True)
            #     j = _read_json(d / "calibrated" / entry["json"])
            #     obj = {
            #         "counts": np.asarray(npz["counts"], dtype=float),
            #         # optional arrays:
            #         **({k: np.asarray(npz[k], dtype=float) for k in npz.files if k != "counts"}),
            #         "meta": j.get("spectrum_meta", {}),
            #     }
            elif fmt == "npz":
                npz = np.load(d / "calibrated" / entry["npz"], allow_pickle=True)
                j = _read_json(d / "calibrated" / entry["json"])
                smeta = j.get("spectrum_meta", {}) or {}
                schema = smeta.get("schema")  # "bq_v1" or "dict_v1" ideally
                files = set(npz.files)

                if schema == "bq_v1":
                    required = {"counts_vals", "bin_centers_kev"}
                    missing = required - files
                    if missing:
                        raise ValueError(
                            f"Calibration artifact {experiment_id}:{label} expected schema=bq_v1 "
                            f"but missing keys {sorted(missing)}; found {sorted(files)}"
                        )
                    obj = {
                        "counts_vals": np.asarray(npz["counts_vals"], dtype=float),
                        "bin_centers_kev": np.asarray(npz["bin_centers_kev"], dtype=float),
                        **({"counts_std": np.asarray(npz["counts_std"], dtype=float)} if "counts_std" in files else {}),
                        "meta": smeta,
                    }

                elif schema == "dict_v1":
                    required = {"counts"}
                    missing = required - files
                    if missing:
                        raise ValueError(
                            f"Calibration artifact {experiment_id}:{label} expected schema=dict_v1 "
                            f"but missing keys {sorted(missing)}; found {sorted(files)}"
                        )
                    obj = {
                        "counts": np.asarray(npz["counts"], dtype=float),
                        **({"energy_axis": np.asarray(npz["energy_axis"], dtype=float)} if "energy_axis" in files else {}),
                        "meta": smeta,
                    }

                else:
                    # If schema wasn't written (older artifacts), you can either error
                    # or infer based on keys. If you want strict, error:
                    raise ValueError(
                        f"Calibration artifact {experiment_id}:{label} has unknown or missing schema in spectrum_meta. "
                        f"spectrum_meta keys={list(smeta.keys())}; npz keys={sorted(files)}"
                    )

            else:
                raise ValueError(f"Unknown calibration entry format: {fmt!r}")

            calibrated[str(label)] = obj

        return CalibrationResult(
            experiment_id=experiment_id,
            calibrated=calibrated,
            meta=dict(meta.get("meta", {})),
        )

    # ---- EfficiencyModel ----

    def save_efficiency_model(self, model: EfficiencyModel) -> None:
        d = self._eff_dir(model.name)
        d.mkdir(parents=True, exist_ok=True)
        _write_json(d / "model.json", model.to_dict())

    def load_efficiency_model(self, name: str) -> EfficiencyModel:
        d = self._eff_dir(name)
        data = _read_json(d / "model.json")
        return EfficiencyModel(
            name=str(data["name"]),
            model_type=str(data["model_type"]),
            params=dict(data["params"]),
            meta=dict(data.get("meta", {})),
        )

    # ---- MLEEstimate ----

    def _mle_sample_dir(self, experiment_id: str, label: str) -> Path:
        # artifacts/mle/<experiment_id>/<label>/
        return self._mle_dir(experiment_id) / _safe_filename(label)

    def _mle_manifest_path(self, experiment_id: str) -> Path:
        # artifacts/mle/<experiment_id>/manifest.jsonl
        return self._mle_dir(experiment_id) / "manifest.jsonl"

    def _mle_npz_path(self, experiment_id: str, label: str, run_id: str) -> Path:
        # artifacts/mle/<experiment_id>/<label>/<run_id>.npz
        d = self._mle_sample_dir(experiment_id, label)
        rid = _safe_filename(run_id)
        return d / f"{rid}.npz"

    def save_mle_estimate(
        self,
        est: MLEEstimate,
        *,
        run_id: str | None = None,
        fit_arrays: Any | None = None,
    ) -> None:
        """
        Layout:
          artifacts/mle/<experiment_id>/<label>/<run_id>.json
          artifacts/mle/<experiment_id>/<label>/<run_id>.npz   (optional)
          artifacts/mle/<experiment_id>/manifest.jsonl         (appended)

        fit_arrays is expected to be an MLEFitResult (or a duck-typed object) with:
          x_keV, y_obs, y_hat, cov, components_hat
        """
        d = self._mle_sample_dir(est.experiment_id, est.label)
        d.mkdir(parents=True, exist_ok=True)

        rid = "default" if run_id is None else _safe_filename(run_id)

        # --- write estimate json ---
        payload = est.to_dict()

        # add a save timestamp (helps debugging / sorting)
        # keep it inside diagnostics so it travels with the artifact
        diag = dict(payload.get("diagnostics", {}) or {})
        if "saved_at" not in diag:
            from datetime import datetime, timezone
            diag["saved_at"] = datetime.now(timezone.utc).isoformat()
        payload["diagnostics"] = diag

        _write_json(d / f"{rid}.json", payload)

        # --- optional: write arrays NPZ ---
        if fit_arrays is not None:
            npz_path = d / f"{rid}.npz"
            arrays: Dict[str, Any] = {
                "x_keV": np.asarray(getattr(fit_arrays, "x_keV")),
                "y_obs": np.asarray(getattr(fit_arrays, "y_obs")),
                "y_hat": np.asarray(getattr(fit_arrays, "y_hat")),
            }
            cov = getattr(fit_arrays, "cov", None)
            if cov is not None:
                arrays["cov"] = np.asarray(cov)

            comps = getattr(fit_arrays, "components_hat", None)
            if isinstance(comps, dict):
                for k, v in comps.items():
                    arrays[f"comp__{str(k)}"] = np.asarray(v)

            np.savez_compressed(npz_path, **arrays)

        # --- append manifest record (for fast querying) ---
        # We try to extract stable fields from diagnostics["run_config"] if present.
        diag2 = payload.get("diagnostics", {}) or {}
        run_cfg = {}
        if isinstance(diag2, dict):
            run_cfg = diag2.get("run_config", {}) if isinstance(diag2.get("run_config", {}), dict) else {}

        record = {
            "experiment_id": est.experiment_id,
            "label": est.label,
            "run_id": rid,
            "saved_at": (diag2.get("saved_at") if isinstance(diag2, dict) else None),
            "method_tag": run_cfg.get("method", None),  # e.g. "poisson_mle_v1"
            "roi_keV": run_cfg.get("roi_keV", None),
            "activity_units": run_cfg.get("activity_units", None),
            "templates": run_cfg.get("templates", None),
            "eff_models": run_cfg.get("eff_models", None),
            # lightweight summary for quick filtering:
            "fit_success": (diag2.get("fit_meta", {}).get("success") if isinstance(diag2.get("fit_meta", {}), dict) else None),
            "nll2": (diag2.get("fit_meta", {}).get("nll2") if isinstance(diag2.get("fit_meta", {}), dict) else None),
            "npz_exists": bool((d / f"{rid}.npz").exists()),
        }

        manifest = self._mle_manifest_path(est.experiment_id)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_to_jsonable(record), sort_keys=True) + "\n")


    def load_mle_estimate(self, experiment_id: str, label: str, *, run_id: str | None = None) -> MLEEstimate:
        """
        Looks for:
        artifacts/mle/<experiment_id>/<label>/<run_id>.json   (new)
        Falls back to:
        artifacts/mle/<experiment_id>/<label>.json            (legacy)
        """
        # --- new layout first ---
        d_new = self._mle_sample_dir(experiment_id, label)
        rid = "default" if run_id is None else _safe_filename(run_id)
        p_new = d_new / f"{rid}.json"
        if p_new.exists():
            data = _read_json(p_new)
        else:
            # --- legacy fallback ---
            d_old = self._mle_dir(experiment_id)
            p_old = d_old / f"{_safe_filename(label)}.json"
            data = _read_json(p_old)

        return MLEEstimate(
            experiment_id=str(data["experiment_id"]),
            label=str(data["label"]),
            activities_bq={k: float(v) for k, v in data["activities_bq"].items()},
            uncertainties_bq={k: float(v) for k, v in data["uncertainties_bq"].items()}
            if data.get("uncertainties_bq") else None,
            diagnostics=dict(data.get("diagnostics", {})),
        )

    def load_mle_fit_arrays(
        self,
        experiment_id: str,
        label: str,
        *,
        run_id: str | None = None,
    ) -> Dict[str, np.ndarray]:
        """
        Load arrays saved alongside an MLE estimate.
        Returns dict keys like: x_keV, y_obs, y_hat, cov, comp__ce134, ...
        """
        rid = "default" if run_id is None else _safe_filename(run_id)
        d = self._mle_sample_dir(experiment_id, label)
        p = d / f"{rid}.npz"
        if not p.exists():
            raise FileNotFoundError(f"No MLE NPZ arrays found: {p}")

        out: Dict[str, np.ndarray] = {}
        with np.load(p, allow_pickle=False) as z:
            for k in z.files:
                out[k] = np.asarray(z[k])
        return out

    def has_calibration(self, experiment_id: str, calibration_tag: str) -> bool:
            d = self._cal_dir(experiment_id, calibration_tag)
            return (d / "meta.json").exists() and (d / "calibrated_index.json").exists()

    def list_mle_runs(self, experiment_id: str, label: str) -> list[str]:
        """
        Returns list of run_ids available for a given (experiment_id, label).
        Only checks the new layout.
        """
        d = self._mle_sample_dir(experiment_id, label)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.json"))

    def make_run_id(config: Mapping[str, Any], *, prefix: str = "mle_poisson_v1", n: int = 10) -> str:
        """
        Deterministic short ID from a config dict.
        Same config -> same run_id.

        prefix: human-readable tag
        n: number of hex chars to keep
        """
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n]
        return f"{prefix}__{h}"

    def scan_mle_manifest(self, experiment_id: str) -> list[dict]:
        """
        Fast scan: reads artifacts/mle/<experiment_id>/manifest.jsonl if present.
        Returns list of dict records.
        """
        p = self._mle_manifest_path(experiment_id)
        if not p.exists():
            return []
        records: list[dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        return records

    def iter_mle_estimates(
        self,
        experiment_id: str,
        *,
        label: str | None = None,
    ):
        """
        Robust iterator that walks new-layout directories and yields:
          (label, run_id, MLEEstimate)

        If label is None: iterates all labels under experiment_id.
        """
        base = self._mle_dir(experiment_id)
        if not base.exists():
            return
        label_dirs = [base / _safe_filename(label)] if label is not None else sorted([p for p in base.iterdir() if p.is_dir()])

        for d in label_dirs:
            lab = d.name
            for p in sorted(d.glob("*.json")):
                run_id = p.stem
                est = self.load_mle_estimate(experiment_id, lab, run_id=run_id)
                yield (lab, run_id, est)

    def scan_mle_estimates(self, experiment_id: str) -> list[dict]:
        """
        Slow-but-complete scan: loads each JSON estimate and extracts a record.
        Useful if manifest doesn't exist yet.
        """
        rows: list[dict] = []
        for lab, run_id, est in self.iter_mle_estimates(experiment_id):
            diag = est.diagnostics or {}
            run_cfg = diag.get("run_config", {}) if isinstance(diag.get("run_config", {}), dict) else {}
            fit_meta = diag.get("fit_meta", {}) if isinstance(diag.get("fit_meta", {}), dict) else {}

            rows.append({
                "experiment_id": experiment_id,
                "label": lab,
                "run_id": run_id,
                "saved_at": diag.get("saved_at", None),
                "method_tag": run_cfg.get("method", None),
                "roi_keV": run_cfg.get("roi_keV", None),
                "activity_units": run_cfg.get("activity_units", None),
                "templates": run_cfg.get("templates", None),
                "eff_models": run_cfg.get("eff_models", None),
                "fit_success": fit_meta.get("success", None),
                "nll2": fit_meta.get("nll2", None),
            })
        return rows

    def mle_runs_table(self, experiment_id: str) -> pd.DataFrame:
        """
        Return a pandas DataFrame describing all MLE runs for an experiment.

        Priority:
          1) manifest.jsonl (fast)
          2) JSON scan (slow)
        """
        records = self.scan_mle_manifest(experiment_id)
        if not records:
            records = self.scan_mle_estimates(experiment_id)

        df = pd.DataFrame.from_records(records)

        # Ensure common columns exist
        for col in [
            "experiment_id", "label", "run_id", "saved_at",
            "method_tag", "roi_keV", "activity_units",
            "templates", "eff_models", "fit_success", "nll2", "npz_exists",
        ]:
            if col not in df.columns:
                df[col] = None

        return df

### Ground truth tracking (for calculating percent deviation)

    def _truth_dir(self) -> Path:
        return self.root / "truth"

    def _truth_path(self, experiment_id: str) -> Path:
        return self._truth_dir() / f"{_safe_filename(experiment_id)}.json"

    def save_ground_truth(self, experiment_id: str, truth_payload: dict) -> None:
        """
        truth_payload should follow the JSON schema described in the notebook docs.
        """
        p = self._truth_path(experiment_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_json(p, truth_payload)

    def load_ground_truth(self, experiment_id: str) -> dict:
        p = self._truth_path(experiment_id)
        if not p.exists():
            raise FileNotFoundError(f"No ground truth file found at {p}")
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ---- CutCountEstimate ----

    def _cutcount_dir(self, experiment_id: str) -> Path:
        return self.root / "cutcount" / _safe_filename(experiment_id)

    def _cutcount_sample_dir(self, experiment_id: str, label: str) -> Path:
        return self._cutcount_dir(experiment_id) / _safe_filename(label)

    def _cutcount_manifest_path(self, experiment_id: str) -> Path:
        # artifacts/cutcount/<experiment_id>/manifest.jsonl
        return self._cutcount_dir(experiment_id) / "manifest.jsonl"

    def _cutcount_npz_path(self, experiment_id: str, label: str, run_id: str) -> Path:
        d = self._cutcount_sample_dir(experiment_id, label)
        rid = _safe_filename(run_id)
        return d / f"{rid}.npz"

    def save_cutcount_estimate(
        self,
        est: CutCountEstimate,
        *,
        run_id: str | None = None,
        fit_arrays: dict[str, Any] | None = None,
    ) -> None:
        """
        Layout:
          artifacts/cutcount/<experiment_id>/<label>/<run_id>.json
          artifacts/cutcount/<experiment_id>/<label>/<run_id>.npz   (optional)
          artifacts/cutcount/<experiment_id>/manifest.jsonl         (appended)

        fit_arrays (optional) is a dict of arrays to save for plotting/debug:
          recommended keys:
            - x_keV
            - y_mix
            - y_template_ce
            - y_residual
            - y_scaled_template_ce
        """
        d = self._cutcount_sample_dir(est.experiment_id, est.label)
        d.mkdir(parents=True, exist_ok=True)

        rid = "default" if run_id is None else _safe_filename(run_id)

        # --- write estimate json ---
        payload = est.to_dict()

        # add a save timestamp (helps debugging / sorting)
        # keep it inside diagnostics so it travels with the artifact
        diag = dict(payload.get("diagnostics", {}) or {})
        if "saved_at" not in diag:
            from datetime import datetime, timezone
            diag["saved_at"] = datetime.now(timezone.utc).isoformat()
        payload["diagnostics"] = diag

        _write_json(d / f"{rid}.json", payload)

        # --- optional: write arrays NPZ ---
        if fit_arrays is not None:
            npz_path = d / f"{rid}.npz"
            arrays_out: Dict[str, Any] = {}
            for k, v in fit_arrays.items():
                if v is None:
                    continue
                arrays_out[str(k)] = np.asarray(v)
            np.savez_compressed(npz_path, **arrays_out)

        # --- append manifest record (for fast querying) ---
        diag2 = payload.get("diagnostics", {}) or {}
        run_cfg = {}
        if isinstance(diag2, dict):
            run_cfg = diag2.get("run_config", {}) if isinstance(diag2.get("run_config", {}), dict) else {}

        record = {
            "experiment_id": est.experiment_id,
            "label": est.label,
            "run_id": rid,
            "saved_at": (diag2.get("saved_at") if isinstance(diag2, dict) else None),
            "method_tag": run_cfg.get("method", None),     # e.g. "cutcount_ceac_v2"
            "roi_keV": run_cfg.get("roi_keV", None),       # if you store it
            "activity_units": run_cfg.get("activity_units", "Bq"),
            "templates": run_cfg.get("templates", None),   # ce template provenance, etc.
            "eff_models": run_cfg.get("eff_models", None), # names/ids of response models used
            "npz_exists": bool((d / f"{rid}.npz").exists()),
        }

        manifest = self._cutcount_manifest_path(est.experiment_id)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_to_jsonable(record), sort_keys=True) + "\n")

    def load_cutcount_estimate(
        self,
        experiment_id: str,
        label: str,
        *,
        run_id: str | None = None,
    ) -> CutCountEstimate:
        d = self._cutcount_sample_dir(experiment_id, label)
        rid = "default" if run_id is None else _safe_filename(run_id)
        p = d / f"{rid}.json"
        data = _read_json(p)
        return CutCountEstimate(
            experiment_id=str(data["experiment_id"]),
            label=str(data["label"]),
            activities_bq={k: float(v) for k, v in data["activities_bq"].items()},
            uncertainties_bq={k: float(v) for k, v in data["uncertainties_bq"].items()}
            if data.get("uncertainties_bq") else None,
            diagnostics=dict(data.get("diagnostics", {})),
        )

    def load_cutcount_fit_arrays(
        self,
        experiment_id: str,
        label: str,
        *,
        run_id: str | None = None,
    ) -> Dict[str, np.ndarray]:
        """
        Load arrays saved alongside a CutCount estimate.
        Returns dict keys like: x_keV, y_mix, y_residual, ...
        """
        rid = "default" if run_id is None else _safe_filename(run_id)
        d = self._cutcount_sample_dir(experiment_id, label)
        p = d / f"{rid}.npz"
        if not p.exists():
            raise FileNotFoundError(f"No CutCount NPZ arrays found: {p}")

        out: Dict[str, np.ndarray] = {}
        with np.load(p, allow_pickle=False) as z:
            for k in z.files:
                out[k] = np.asarray(z[k])
        # print(type(out))
        # print(out)
        return out

    def list_cutcount_runs(self, experiment_id: str, label: str) -> list[str]:
        d = self._cutcount_sample_dir(experiment_id, label)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.json"))

    def scan_cutcount_manifest(self, experiment_id: str) -> list[dict]:
        """
        Fast scan: reads artifacts/cutcount/<experiment_id>/manifest.jsonl if present.
        """
        p = self._cutcount_manifest_path(experiment_id)
        if not p.exists():
            return []
        records: list[dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        return records

    def iter_cutcount_estimates(
        self,
        experiment_id: str,
        *,
        label: str | None = None,
    ):
        """
        Iterator that walks cutcount directories and yields:
          (label, run_id, CutCountEstimate)
        """
        base = self._cutcount_dir(experiment_id)
        if not base.exists():
            return
        label_dirs = [base / _safe_filename(label)] if label is not None else sorted([p for p in base.iterdir() if p.is_dir()])

        for d in label_dirs:
            lab = d.name
            for p in sorted(d.glob("*.json")):
                run_id = p.stem
                est = self.load_cutcount_estimate(experiment_id, lab, run_id=run_id)
                yield (lab, run_id, est)

    def scan_cutcount_estimates(self, experiment_id: str) -> list[dict]:
        """
        Slow-but-complete scan: loads each JSON estimate and extracts a record.
        Useful if manifest doesn't exist yet.
        """
        rows: list[dict] = []
        for lab, run_id, est in self.iter_cutcount_estimates(experiment_id):
            diag = est.diagnostics or {}
            run_cfg = diag.get("run_config", {}) if isinstance(diag.get("run_config", {}), dict) else {}

            rows.append({
                "experiment_id": experiment_id,
                "label": lab,
                "run_id": run_id,
                "saved_at": diag.get("saved_at", None),
                "method_tag": run_cfg.get("method", None),
                "roi_keV": run_cfg.get("roi_keV", None),
                "activity_units": run_cfg.get("activity_units", "Bq"),
                "templates": run_cfg.get("templates", None),
                "eff_models": run_cfg.get("eff_models", None),
                "npz_exists": bool((self._cutcount_npz_path(experiment_id, lab, run_id)).exists()),
            })
        return rows

    def cutcount_runs_table(self, experiment_id: str) -> pd.DataFrame:
        """
        Return a pandas DataFrame describing all CutCount runs for an experiment.

        Priority:
          1) manifest.jsonl (fast)
          2) JSON scan (slow)
        """
        records = self.scan_cutcount_manifest(experiment_id)
        if not records:
            records = self.scan_cutcount_estimates(experiment_id)

        df = pd.DataFrame.from_records(records)

        for col in [
            "experiment_id", "label", "run_id", "saved_at",
            "method_tag", "roi_keV", "activity_units",
            "templates", "eff_models", "npz_exists",
        ]:
            if col not in df.columns:
                df[col] = None

        return df

    # def _cutcount_dir(self, experiment_id: str) -> Path:
    #     return self.root / "cutcount" / _safe_filename(experiment_id)

    # def _cutcount_sample_dir(self, experiment_id: str, label: str) -> Path:
    #     return self._cutcount_dir(experiment_id) / _safe_filename(label)

    # def save_cutcount_estimate(self, est: CutCountEstimate, *, run_id: str | None = None) -> None:
    #     """
    #     Layout:
    #     artifacts/cutcount/<experiment_id>/<label>/<run_id>.json
    #     """
    #     d = self._cutcount_sample_dir(est.experiment_id, est.label)
    #     d.mkdir(parents=True, exist_ok=True)
    #     rid = "default" if run_id is None else _safe_filename(run_id)
    #     _write_json(d / f"{rid}.json", est.to_dict())

    # def load_cutcount_estimate(self, experiment_id: str, label: str, *, run_id: str | None = None) -> CutCountEstimate:
    #     d = self._cutcount_sample_dir(experiment_id, label)
    #     rid = "default" if run_id is None else _safe_filename(run_id)
    #     p = d / f"{rid}.json"
    #     data = _read_json(p)
    #     return CutCountEstimate(
    #         experiment_id=str(data["experiment_id"]),
    #         label=str(data["label"]),
    #         activities_bq={k: float(v) for k, v in data["activities_bq"].items()},
    #         uncertainties_bq={k: float(v) for k, v in data["uncertainties_bq"].items()}
    #         if data.get("uncertainties_bq") else None,
    #         diagnostics=dict(data.get("diagnostics", {})),
    #     )

    # def list_cutcount_runs(self, experiment_id: str, label: str) -> list[str]:
    #     d = self._cutcount_sample_dir(experiment_id, label)
    #     if not d.exists():
    #         return []
    #     return sorted(p.stem for p in d.glob("*.json"))

# ----------------------------
# Filename safety
# ----------------------------

def _safe_filename(s: str) -> str:
    """
    Make a reasonably safe filename from an arbitrary label.
    """
    s = str(s)
    # Replace common problematic characters
    for ch in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|", "\n", "\r", "\t"]:
        s = s.replace(ch, "_")
    s = s.strip()
    return s if s else "empty_label"
