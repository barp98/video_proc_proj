#!/usr/bin/env python3
"""
matting.py – Stand‑alone soft‑matting & compositing script (hard‑coded paths)
============================================================================

This version keeps *exactly the same* matting logic as before but removes
all command‑line arguments.  Fill in **your own absolute paths** below once
and just run:

    python matting.py

Nothing else is required – the script will read the fixed files, create the
result directory if needed, and write the alpha‑matte & composited video
there.
"""
from pathlib import Path
import cv2
import numpy as np
import os

# ───────────────────────────────────────
# USER‑SPECIFIC CONFIG  – ***EDIT HERE***
# ───────────────────────────────────────
ROOT = Path(r"C:\Users\Bar\Desktop\University\semester_8\video_processing\final_project")

FG_PATH   = ROOT / "Outputs" / "extracted_ID1_ID2.avi"      # person on black
MASK_PATH = ROOT / "Outputs" / "binary_ID1_ID2.avi"         # 0/255 mask video
BG_PATH   = ROOT / "Inputs"  / "background.jpg"     # new background image

OUT_DIR   = ROOT / "Outputs"                         # results folder
PREFIX    = "123456_987654"                          # file‑name tag
# ───────────────────────────────────────

# Derived output filenames
ALPHA_PATH  = OUT_DIR / f"alpha_{PREFIX}.avi"
MATTED_PATH = OUT_DIR / f"matted_{PREFIX}.avi"

# Create output directory if it doesn't exist
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────
# Helper functions
# ───────────────────────────────────────

def make_alpha(mask: np.ndarray, blur_sigma: int = 7, band: int = 5) -> np.ndarray:
    """Return a float32 alpha matte ∈ [0,1] from a uint8{0,1} mask."""
    dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)  # background side
    dist_in  = cv2.distanceTransform(mask,      cv2.DIST_L2, 3)  # foreground side
    signed   = dist_out - dist_in                               # + outside, ‑ inside
    soft     = 0.5 * (1 + np.tanh(-signed / band))              # smooth step
    return cv2.GaussianBlur(soft, (0, 0), blur_sigma)

def composite(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Alpha‑blend foreground & background, returning uint8 BGR."""
    alpha3 = alpha[..., None]  # H×W×1  → broadcast across RGB
    return (alpha3 * fg + (1 - alpha3) * bg).astype(np.uint8)

# ───────────────────────────────────────
# Main processing
# ───────────────────────────────────────

if __name__ == "__main__":
    # Open sources
    fg_cap   = cv2.VideoCapture(str(FG_PATH))
    mask_cap = cv2.VideoCapture(str(MASK_PATH))

    if not fg_cap.isOpened():
        raise FileNotFoundError(f"Cannot open foreground video: {FG_PATH}")
    if not mask_cap.isOpened():
        raise FileNotFoundError(f"Cannot open mask video: {MASK_PATH}")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps    = fg_cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(fg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(fg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg_img = cv2.imread(str(BG_PATH))
    if bg_img is None:
        raise FileNotFoundError(f"Cannot open background image: {BG_PATH}")
    bg_img = cv2.resize(bg_img, (W, H))

    alpha_vw  = cv2.VideoWriter(str(ALPHA_PATH),  fourcc, fps, (W, H), False)
    matted_vw = cv2.VideoWriter(str(MATTED_PATH), fourcc, fps, (W, H), True)

    frame_idx = 0
    while True:
        ret_fg,  fg_frame   = fg_cap.read()
        ret_mk,  mask_frame = mask_cap.read()
        if not (ret_fg and ret_mk):
            break  # finished all frames

        # binary mask (0 or 1)
        mask_gray  = (cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

        # soft α-matte (kept for deliverables)
        alpha      = make_alpha(mask_gray)                 # float32 [0,1]

        # ── HARD compositing: use the binary mask, not the α-matte ────────────
        hard_alpha = mask_gray.astype(np.float32)          # 0.0 or 1.0
        matted     = composite(fg_frame, bg_img, hard_alpha)


        alpha_vw.write((alpha * 255).astype(np.uint8))  # save alpha matte
        matted_vw.write(matted)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames…")

    # Cleanup
    fg_cap.release()
    mask_cap.release()
    alpha_vw.release()
    matted_vw.release()

    print("Done! →", ALPHA_PATH.name, "&", MATTED_PATH.name)
