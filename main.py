#!/usr/bin/env python3
"""
main.py – orchestration driver for the video‑processing pipeline
================================================================
Runs **all four stages** in sequence:

1. Stabilisation      (stabilize.py →  stabilised_*.avi)
2. Background removal (subtract.py  →  extracted_*.avi & binary_*.avi)
3. Matting            (matting.py   →  matted_*.avi   & alpha_*.avi)
4. Tracking           (tracking.py  →  OUTPUT_*.avi)

For every output created we also record the running time (in seconds from
script start) into *Outputs/timing.json* so that it complies with the
project’s automatic tester.  All paths are **relative** to the project’s
root folder and **no user interaction** is required.

Usage
-----
    $ python Code/main.py  [--id1 123456] [--id2 987654]

The two optional command–line flags let you pass your student IDs without
hard‑coding them in the file.

Directory layout assumed (case‑sensitive!)
------------------------------------------
Final_Project_ID1_ID2/
├── Code/
│   ├── main.py            ← *this* script
│   ├── stabilize.py
│   ├── subtract.py
│   ├── matting.py
│   └── tracking.py
├── Inputs/
│   ├── INPUT.avi
│   └── background.jpg
└── Outputs/               ← will be created if missing

If you renamed the folders you only need to adjust the *DIR_* constants
below – nothing else.
"""

from __future__ import annotations
import argparse
import json
import runpy
import sys
import time
from pathlib import Path

# ───────────────────────────── constants ──────────────────────────────
DIR_CODE    = Path(__file__).resolve().parent           # …/Code
DIR_ROOT    = DIR_CODE.parent                           # project root
DIR_INPUTS  = DIR_ROOT / "Inputs"
DIR_OUTPUTS = DIR_ROOT / "Outputs"
DIR_OUTPUTS.mkdir(exist_ok=True)

# default dummy IDs – override with CLI flags!
DEFAULT_ID1 = "325106854"
DEFAULT_ID2 = "207234550"

# helper – pretty printing in one place
def log(msg: str) -> None:
    print(msg, flush=True)

# ─────────────────────────── argument parsing ─────────────────────────
cli = argparse.ArgumentParser(prog="main.py",
                              description="Run the full project pipeline.")
cli.add_argument("--id1", default=DEFAULT_ID1, help="first student ID")
cli.add_argument("--id2", default=DEFAULT_ID2, help="second student ID")
IDS = cli.parse_args()
ID_TAG = f"{IDS.id1}_{IDS.id2}"

# ─────────────────────────────  helpers  ──────────────────────────────
TIMING: dict[str, float] = {}
START_TIME = time.perf_counter()

def timestamp(filename: str) -> None:
    """Store time (s) elapsed since script launch for *filename*."""
    TIMING[filename] = round(time.perf_counter() - START_TIME, 3)

# ---------------------------------------------------------------------
# 1)  stabilisation  –  call function from *stabilize.py*
# ---------------------------------------------------------------------
log("[1/4]  Stabilising video…")
from stabilize import video_stabilization  # local import keeps startup fast

IN_VIDEO   = DIR_INPUTS  / "INPUT.avi"
STAB_VIDEO = DIR_OUTPUTS / f"stabilized_{ID_TAG}.avi"
video_stabilization(str(IN_VIDEO), str(STAB_VIDEO))
timestamp(STAB_VIDEO.name)

# ---------------------------------------------------------------------
# 2)  background subtraction  –  call function from *subtract.py*
# ---------------------------------------------------------------------
log("[2/4]  Removing background…")
from subtract import extract_person  # imported only when needed

EXTRACTED_VIDEO = DIR_OUTPUTS / f"extracted_{ID_TAG}.avi"
BINARY_VIDEO    = DIR_OUTPUTS / f"binary_{ID_TAG}.avi"
extract_person(
    video_path     = STAB_VIDEO,
    out_color_path = EXTRACTED_VIDEO,
    out_bin_path   = BINARY_VIDEO,
)
# both videos finish at the same moment → identical time‑stamp
for fname in (EXTRACTED_VIDEO.name, BINARY_VIDEO.name):
    timestamp(fname)

# ---------------------------------------------------------------------
# 3)  matting  –  run *matting.py* as a script after patching its paths
# ---------------------------------------------------------------------
log("[3/4]  Compositing on new background…")
import importlib
import matting  # this loads the *module* without executing its __main__

# overwrite its hard‑coded constants so no edit inside the file is needed
matting.ROOT     = DIR_ROOT
matting.FG_PATH  = EXTRACTED_VIDEO
matting.MASK_PATH = BINARY_VIDEO
matting.BG_PATH  = DIR_INPUTS / "background.jpg"
matting.OUT_DIR  = DIR_OUTPUTS
matting.PREFIX   = ID_TAG

# now execute the script portion inside a fresh namespace
runpy.run_module(matting.__name__, run_name="__main__")

ALPHA_VIDEO  = DIR_OUTPUTS / f"alpha_{ID_TAG}.avi"
MATTED_VIDEO = DIR_OUTPUTS / f"matted_{ID_TAG}.avi"
for fname in (ALPHA_VIDEO.name, MATTED_VIDEO.name):
    timestamp(fname)

# ---------------------------------------------------------------------
# 4)  tracking  –  run *tracking.py* as a script with patched paths
# ---------------------------------------------------------------------
log("[4/4]  Tracking subject…")
import tracking  # again: import first, then patch, then run
tracking.PROJECT     = DIR_ROOT
tracking.COLOR_PATH  = MATTED_VIDEO
tracking.ALPHA_PATH  = ALPHA_VIDEO
TRACKED_VIDEO        = DIR_OUTPUTS / f"OUTPUT_{ID_TAG}.avi"
tracking.OUT_PATH    = TRACKED_VIDEO

runpy.run_module(tracking.__name__, run_name="__main__")

timestamp(TRACKED_VIDEO.name)

# ---------------------------------------------------------------------
# 5)  write timing.json   (tracking.json is produced by *tracking.py*)
# ---------------------------------------------------------------------
with (DIR_OUTPUTS / "timing.json").open("w", encoding="utf-8") as fp:
    json.dump(TIMING, fp, indent=2, ensure_ascii=False)

log("\n✓  All done – results are in the ‘Outputs’ folder.")
