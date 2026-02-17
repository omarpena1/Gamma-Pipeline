"""
gamma: a small local package for the gamma-spectroscopy pipeline.

This __init__.py defines the *public API* (what you import from `gamma`),
and keeps imports light so notebooks start quickly.

Usage in notebooks (examples):
    from gamma import ArtifactStore, Experiment, CalibratedSpectrum
    from gamma import load_experiment_xls, calibrate_experiment
    from gamma import fit_efficiency_models, run_mle
"""

# Notes (important)
# This assumes you will define the following in those modules later:
# types.py: Experiment, SpectrumRecord, CalibrationSpec, CalibrationResult, EfficiencyModel, MLEEstimate
# io_excel.py: load_experiment_xls, load_biodistribution_xls
# io_mappings.py: ExperimentMapping, PatternSpec
# calibration.py: calibrate_experiment
# efficiency.py: fit_efficiency_models
# mle.py: run_mle
# artifacts.py: ArtifactStore

#Run this as a Sanity Check
# import sys
# from pathlib import Path

# # Ensure repo root is on sys.path (so `import gamma` works)
# repo_root = Path.cwd().parent
# if str(repo_root) not in sys.path:
#     sys.path.insert(0, str(repo_root))

# import gamma
# print(gamma.__version__)

from __future__ import annotations

# Versioning is optional but helpful for debugging
__all__ = [
    "__version__",
    # Types / domain objects
    "Experiment",
    "SpectrumRecord",
    "CalibrationSpec",
    "CalibrationResult",
    "EfficiencyModel",
    "MLEEstimate",
    # IO
    "load_experiment_xls",
    "load_biodistribution_xls",
    # Mapping helpers (spec objects)
    "ExperimentMapping",
    "PatternSpec",
    # Processing
    "calibrate_experiment",
    "fit_efficiency_models",
    # "run_mle",
    "fit_efficiency_models",
    "fit_mle_poisson",
    "to_mle_estimate",
    # Persistence
    "ArtifactStore",
]

__version__ = "0.1.0"

# --- Domain types (lightweight) ---
from .types import (
    Experiment,
    SpectrumRecord,
    CalibrationSpec,
    CalibrationResult,
    EfficiencyModel,
    MLEEstimate,
)

# --- IO + mappings ---
from .io_excel import load_experiment_xls, load_biodistribution_xls
from .io_mappings import ExperimentMapping, PatternSpec

# --- Processing ---
from .calibration import calibrate_experiment
from .efficiency import fit_efficiency_models
from .mle import fit_mle_poisson, to_mle_estimate

# --- Artifacts ---
from .artifacts import ArtifactStore
