# Code/background_and_foreground.py
# --------------------------------------------------------------------
# Build a median background from a stabilised video, then extract the
# moving person using that background.  Outputs:
#   • MEDIAN_BG.png
#   • extracted_ID1_ID2.avi   (colour, person only)
#   • binary_ID1_ID2.avi      (mask, 0/255)
# --------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, Dict, Any

import os
import cv2
import numpy as np


# ────────────────────────── helper: logging ────────────────────────────
def _log(msg: str, gui_cb: Optional[Callable[[str], None]]) -> None:
    if gui_cb:
        gui_cb(msg)
    else:
        print(msg)


# ────────────────────────── main worker ────────────────────────────────
def extract_person(
    video_path: str | Path,
    out_color_path: str | Path,
    out_bin_path: str | Path,
    params: Dict[str, Any] | None = None,
    textGUI: Optional[Callable[[str], None]] = None,
) -> None:
    p = {
        "thrsh_const": 35,
        "morph_kernel": 5,
        "min_area_frac": 0.02,
        "median_stride": 1,    # take every N-th frame for median
    }
    if params:
        p.update(params)

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    # ── 0. Inspect once to get video meta ──────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    nfrm   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # ── 1. Build median background ─────────────────────────────────────
    _log("Step 1/3: computing median background …", textGUI)
    stride = max(1, int(p["median_stride"]))
    sample_idx = np.arange(0, nfrm, stride, dtype=int)

    bg_accum = np.empty((len(sample_idx), H, W, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(str(video_path))
    for i, fidx in enumerate(sample_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frm = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {fidx}")
        bg_accum[i] = frm
    cap.release()

    median_bg = np.median(bg_accum.astype(np.float32), axis=0).astype(np.uint8)

    # save background for debugging
    bg_path = Path(out_color_path).with_name("MEDIAN_BG.png")
    cv2.imwrite(str(bg_path), median_bg)
    _log(f"Background saved → {bg_path}", textGUI)

    # ── 2. Prepare writers and common objects ──────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw_color = cv2.VideoWriter(str(out_color_path), fourcc, fps, (W, H))
    vw_bin = cv2.VideoWriter(
        str(out_bin_path), fourcc, fps, (W, H), isColor=False
    )

    kernel = np.ones((p["morph_kernel"], p["morph_kernel"]), np.uint8)
    min_area = p["min_area_frac"] * W * H
    thrsh_const = int(p["thrsh_const"])

    # ── 3. Second pass: foreground extraction ──────────────────────────
    _log("Step 2/3: extracting foreground …", textGUI)
    cap = cv2.VideoCapture(str(video_path))
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # diff in HSV
        diff = cv2.absdiff(
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
            cv2.cvtColor(median_bg, cv2.COLOR_BGR2HSV),
        )
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # threshold
        if thrsh_const < 0:
            _, mask = cv2.threshold(
                diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, mask = cv2.threshold(diff_gray, thrsh_const, 255, cv2.THRESH_BINARY)

        # open + close with square kernel (NO edge-rounding)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # largest blob only
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= min_area:
                mask[:] = 0
                cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
            else:
                mask[:] = 0

        extracted = cv2.bitwise_and(frame, frame, mask=mask)

        vw_color.write(extracted)
        vw_bin.write(mask)

        f += 1
        if f % 50 == 0:
            _log(f"  processed {f}/{nfrm} frames", textGUI)

    cap.release()
    vw_color.release()
    vw_bin.release()
    _log("Step 3/3: done!", textGUI)


# ───────────────────────────── CLI runner ──────────────────────────────
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]  # …/Final_Project
    OUTPUT = ROOT / "Outputs"

    extract_person(
        video_path=OUTPUT / "STAB.avi",
        out_color_path=OUTPUT / "extracted_ID1_ID2.avi",
        out_bin_path=OUTPUT / "binary_ID1_ID2.avi",
        params=dict(
            thrsh_const=30,   # tweak if limbs clip or background shows
            morph_kernel=3,
            min_area_frac=0.01,
            median_stride=2
        ),
    )
