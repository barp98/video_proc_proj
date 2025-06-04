#!/usr/bin/env python3
"""
tracking.py – single‑input † matted video + alpha‑matte helper
=============================================================

We now lock the bounding box from **frame 0** by taking advantage of the
alpha‑matte video as a ready‑made foreground mask.  No warm‑up, no MOG2
background model.

Inputs (hard‑coded ‑ edit once)
------------------------------
* **Colour / composited clip** …  `Outputs/matted_123456_987654.avi`
* **Alpha‑matte clip** …………...  `Outputs/alpha_123456_987654.avi`

Output
------
* `Outputs/tracking_123456_987654.avi` – original colour frames with a
  green rectangle following the largest contour (person) every frame.

If your IDs or folder names differ, tweak the strings under *PATHS* below.
The remainder of the script is unchanged.
"""
from pathlib import Path
import cv2
import numpy as np

# ─────────────────────── PATHS (edit to match your setup) ──────────────
PROJECT = Path(__file__).resolve().parents[1]   # …/final_project
COLOR_PATH = PROJECT / "Outputs" / "matted_123456_987654.avi"
ALPHA_PATH = PROJECT / "Outputs" / "alpha_123456_987654.avi"
OUT_PATH   = PROJECT / "Outputs" / "tracking_123456_987654.avi"
# ───────────────────────────────────────────────────────────────────────

# open videos -----------------------------------------------------------
cap_col   = cv2.VideoCapture(str(COLOR_PATH))
cap_alpha = cv2.VideoCapture(str(ALPHA_PATH))
assert cap_col.isOpened(),   f"Could not open {COLOR_PATH}"
assert cap_alpha.isOpened(), f"Could not open {ALPHA_PATH}"

fps = cap_col.get(cv2.CAP_PROP_FPS)
W   = int(cap_col.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap_col.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_vw = cv2.VideoWriter(str(OUT_PATH), fourcc, fps, (W, H), True)

# bbox smoothing --------------------------------------------------------
EMA_WEIGHT = 0.25  # lower = smoother, higher = quicker
bbox_ema   = None  # (cx, cy, w, h)

def smooth_bbox(new_bbox):
    global bbox_ema
    if bbox_ema is None:
        bbox_ema = np.array(new_bbox, dtype=float)
    else:
        bbox_ema = EMA_WEIGHT * np.array(new_bbox) + (1 - EMA_WEIGHT) * bbox_ema
    return bbox_ema.astype(int)

# processing loop -------------------------------------------------------
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
frame_idx = 0

while True:
    ret_c, frame_c = cap_col.read()
    ret_a, frame_a = cap_alpha.read()
    if not (ret_c and ret_a):
        break  # reached EOF on one of the streams

    # grab binary mask from alpha video
    gray = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # tidy edges a bit
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 200:  # small area threshold to ignore noise
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy     = x + w // 2, y + h // 2
            scx, scy, sw, sh = smooth_bbox((cx, cy, w, h))
            tl = (int(scx - sw/2), int(scy - sh/2))
            br = (int(scx + sw/2), int(scy + sh/2))
            cv2.rectangle(frame_c, tl, br, (0, 255, 0), 2)

    cv2.putText(frame_c, f"Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 255, 20), 2, cv2.LINE_AA)

    out_vw.write(frame_c)
    frame_idx += 1

# cleanup --------------------------------------------------------------
cap_col.release()
cap_alpha.release()
out_vw.release()

print(f"Tracking finished → {OUT_PATH}")
