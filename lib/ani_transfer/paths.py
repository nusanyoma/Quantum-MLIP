"""Centralized filesystem paths for the reproduction code.

Every script under code/code/ imports DATA_DIR / MODEL_DIR / RESULTS_DIR from
here instead of hardcoding relative paths. This makes the scripts work
regardless of how deep they sit under code/code/, since the paths are always
resolved relative to this file's own location
(code/code/lib/ani_transfer/paths.py), not the caller's working directory.

code/code/ is self-contained: DATA_DIR/MODEL_DIR/RESULTS_DIR point at
code/code/data, code/code/model, code/code/results (not the original
project's top-level data/model/results directories), which are pre-seeded
with exactly the files this reproduction pipeline needs. Running the
scripts adds to model/ and results/ in place.
"""
import pathlib

# code/code/lib/ani_transfer/paths.py -> parents[0]=ani_transfer, [1]=lib,
# [2]=code/code (this project's own root)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"
