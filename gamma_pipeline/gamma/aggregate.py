# Note for if dozens of data sets are used in the future to generate activity-to-detector-response curves:
# 4) Recommended evolution: keep your merge, but add a “points collector”
# This is the most idiomatic path that preserves your current code and adds scalability without rewriting everything.
# Add a helper (new function, probably in efficiency.py or aggregate.py)

# Something like:
# collect_efficiency_points(cals, key_to_activity, spec, extractor) -> arrays + provenance
# optionally: aggregate_points_by_activity(points, method="mean", yerr="sd|sem")
# Then you can decide (per notebook):
# 	· fit using all points (maximum info), or
# 	· fit using activity-aggregated points (balanced, cleaner).
# This makes the workflow robust when you start pooling dozens of experiments.


from __future__ import annotations

import re
import numpy as np

from gamma.types import Experiment
from gamma.types import CalibrationResult

from typing import Dict, Iterable, List, Literal, Tuple
from collections import defaultdict

# -------------------------------------------------------------------
# Label parsing (generalized)
#
# OLD accepted:  "ACT_repN"                (single component)
# NEW accepts:  "A1:A2:...:AK_repN"        (K>=1 components)
#
# Behavior is identical for single-component labels; the only change
# is that we now allow ':'-separated mixtures and aggregate by the
# full mixture key rather than a single float.
# -------------------------------------------------------------------

_NUM_RE = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$")
_LABEL_RE = re.compile(r"^(?P<acts>.+)_rep(?P<rep>\d+)$")


def parse_activity_key(label: str) -> Tuple[float, ...]:
    """
    Parse label into an activity key.

    Accepts:
      - "A_repN"
      - "A1:A2_repN"
      - "A1:A2:...:AK_repN" for any K>=1

    Returns:
      tuple of floats (length K)
    """
    m = _LABEL_RE.match(label)
    if not m:
        raise ValueError(f"Label {label!r} doesn't match 'A(:A:...)?_repN'")

    acts_str = m.group("acts").strip()
    parts = [p.strip() for p in acts_str.split(":") if p.strip() != ""]
    if not parts:
        raise ValueError(f"Label {label!r} has empty activity list before _repN")

    # Validate numeric tokens (helps catch accidental formatting issues early)
    for p in parts:
        if not _NUM_RE.match(p):
            raise ValueError(f"Label {label!r} has non-numeric activity token {p!r}")

    return tuple(float(p) for p in parts)


def format_activity_key(key: Tuple[float, ...]) -> str:
    """Canonical label for aggregated groups (drops the _rep suffix)."""
    return ":".join(f"{x:g}" for x in key)


def aggregate_replicates(
    exp: Experiment,
    *,
    ddof: int = 1,
    use_sem: bool = False,
    store_poisson: bool = True,
):
    """
    Same semantics as the original aggregate_replicates, except:
    - Grouping key is the full activity mixture tuple, not a single float.
    - Output group labels are 'A' or 'A1:A2:...:AK' (no _rep), matching the mixture.
    """
    keys = [parse_activity_key(lab) for lab in exp.labels]

    # Build groups efficiently (avoids repeated np.where scans and supports tuple keys)
    groups: Dict[Tuple[float, ...], List[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)

    uniq_keys = sorted(groups.keys())

    mean_list, sd_list, n_list, group_labels = [], [], [], []

    for key in uniq_keys:
        idx = groups[key]
        x = exp.counts[idx, :]
        mu = x.mean(axis=0)
        sd_between = x.std(axis=0, ddof=ddof) if len(idx) > 1 else np.zeros_like(mu)

        mean_list.append(mu)
        sd_list.append(sd_between)
        n_list.append(len(idx))
        group_labels.append(format_activity_key(key))

    counts_mean = np.stack(mean_list, axis=0)
    counts_sd_between = np.stack(sd_list, axis=0)
    n_reps = np.array(n_list, dtype=int)

    counts_sem = counts_sd_between / np.sqrt(np.maximum(n_reps[:, None], 1))
    unc = counts_sem if use_sem else counts_sd_between

    new_meta = dict(exp.meta or {})
    new_meta["aggregation"] = {
        "group_key": "activity",  # keep same key name for compatibility
        "n_reps": n_reps.tolist(),
        "ddof": ddof,
        "uncertainty_kind": "sem" if use_sem else "sd_between",
        "label_grammar": "A(:A:...)*_repN",
    }

    exp_agg = Experiment(
        experiment_id=exp.experiment_id + "__agg",
        counts=counts_mean,
        labels=group_labels,
        weights=None,
        meta=new_meta,
    )

    if store_poisson:
        sd_poisson = np.sqrt(np.maximum(counts_mean, 0.0))
        return exp_agg, unc, sd_poisson

    return exp_agg, unc


def merge_calibrations_for_efficiency(
    cals: Iterable[CalibrationResult],
    *,
    key_style: Literal["prefix"] = "prefix",
    sep: str = "::",
) -> Tuple[CalibrationResult, Dict[str, str]]:
    """
    Merge multiple CalibrationResult objects into one CalibrationResult suitable for
    efficiency fitting, without losing provenance or risking label collisions.

    Returns
    -------
    merged_cal : CalibrationResult
        merged_cal.calibrated has UNIQUE keys.
    key_to_activity_label : Dict[str, str]
        maps merged keys -> original activity label (the per-file label string).

    Notes
    -----
    - We do NOT attempt to interpret numeric activities here.
    - We do NOT change the underlying spectrum objects.
    """
    cals = list(cals)
    if len(cals) == 0:
        raise ValueError("merge_calibrations_for_efficiency: no calibrations provided.")

    merged: Dict[str, object] = {}
    key_to_activity_label: Dict[str, str] = {}
    source_expids: List[str] = []

    for cal in cals:
        source_expids.append(cal.experiment_id)

        for lab, obj in cal.calibrated.items():
            lab_str = str(lab)

            if key_style == "prefix":
                k = f"{cal.experiment_id}{sep}{lab_str}"
            else:
                raise ValueError(f"Unknown key_style={key_style!r}")

            if k in merged:
                # collision should basically never happen with prefix scheme
                raise ValueError(f"Collision after prefixing: {k!r}")

            merged[k] = obj
            key_to_activity_label[k] = lab_str

    # Merge meta (keep it lightweight)
    merged_meta = {
        "merge": {
            "kind": "merge_calibrations_for_efficiency",
            "n_sources": len(cals),
            "source_experiment_ids": source_expids,
            "key_style": key_style,
            "sep": sep,
        }
    }

    merged_cal = CalibrationResult(
        experiment_id="__".join(source_expids) + "__merged",
        calibrated=merged,
        meta=merged_meta,
    )
    return merged_cal, key_to_activity_label