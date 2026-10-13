"""Filesystem paths shared by all scripts: DATA_DIR, MODEL_DIR, RESULTS_DIR.

Resolved relative to the repository root (two levels above this file), so
scripts work regardless of the current working directory.
"""
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"
