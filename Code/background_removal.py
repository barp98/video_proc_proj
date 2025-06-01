"""
background_removal_nofeather.py
===============================
* 1-pass colour-median background
* Colour-distance threshold  (ΔRGB L2)
* Mask clean-up: erode → dilate-small → dilate-big
* Keep largest blob, **NO feather**
* Writes:
      Outputs/EXTRACTED.avi   – subject on hard black
      Outputs/BINARY.avi      – 0/255 mask
      Outputs/MEDIAN_BG.png   – the median background image

Requires: opencv-python, numpy
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Callable, Optional


def video_background_removal(in_path: str,
                             out_path_extracted: str,
                             out_path_binary: str,
                             params: Dict[str, Any],
                             textGUI: Optional[Callable[[str], None]] = None
                             ) -> None:
    """Foreground extraction with hard edges + saves the background image."""
    def log(msg: str) -> None:
        if textGUI:
            textGUI(msg)

    # ---------- open input --------------------------------------------------
    if not os.path.isfile(in_path):
        raise FileNotFoundError(in_path)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {in_path}")

    nfrm   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # ---------- build colour-median background ------------------------------
    log("Pre-processing: caching frames for median background …")
    all_frames = np.empty((nfrm, height, width, 3), dtype=np.uint8)

    for i in range(nfrm):
        ok, frm = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {i}")
        all_frames[i] = frm

    median_bg = np.median(all_frames.astype(np.float32), axis=0).astype(np.uint8)
    cap.release()

    # ---------- NEW: save the background image ------------------------------
    bg_path = os.path.join(os.path.dirname(out_path_extracted), "MEDIAN_BG.png")
    cv2.imwrite(bg_path, median_bg)
    log(f"Median background saved to: {bg_path}")

    # ---------- parameters & structuring elements ---------------------------
    thresh      = float(params.get("thresh", 36))
    er_k        = int(params.get("Erode",   5))
    dil1_k      = int(params.get("Dilate1", 3))
    dil2_k      = int(params.get("Dilate2", 6))
    gauss_sigma = float(params.get("GaussianFiltSigma", 3.0))

    se_erode  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er_k,  er_k))
    se_dil1   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil1_k, dil1_k))
    se_dil2   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil2_k, dil2_k))

    # ---------- prepare output writers --------------------------------------
    cap = cv2.VideoCapture(in_path)
    out_extr = cv2.VideoWriter(out_path_extracted,
                               cv2.VideoWriter_fourcc(*'XVID'),
                               fps, (width, height))
    out_bin  = cv2.VideoWriter(out_path_binary,
                               cv2.VideoWriter_fourcc(*'XVID'),
                               fps, (width, height), isColor=False)

    border = 5  # ignore narrow rim to avoid codec debris

    # ---------- main loop ----------------------------------------------------
    for idx in range(nfrm):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        log(f"Processing frame {idx+1}/{nfrm}")

        # (1) colour-distance mask -------------------------------------------
        diff = np.linalg.norm(frame_bgr.astype(np.int16) -
                              median_bg.astype(np.int16), axis=2)
        fg = (diff > thresh * np.sqrt(3)).astype(np.uint8)

        # (2) pin-hole fill ---------------------------------------------------
        fg = cv2.GaussianBlur(fg, (5, 5), 0)
        _, fg = cv2.threshold(fg, 0.2, 1, cv2.THRESH_BINARY)

        # (3) morphology ------------------------------------------------------
        fg = cv2.erode(fg, se_erode, 1)
        fg = cv2.dilate(fg, se_dil1, 1)
        fg = cv2.dilate(fg, se_dil2, 1)
        fg[:border, :] = fg[-border:, :] = fg[:, :border] = fg[:, -border:] = 0

        # keep largest blob
        n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
        if n_lbl > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            fg = (lbl_map == largest).astype(np.uint8)

        # (4) write outputs ---------------------------------------------------
        mask255 = fg.astype(np.uint8) * 255
        out_bin.write(mask255)

        extracted = (frame_bgr * fg[:, :, None]).astype(np.uint8)
        out_extr.write(extracted)

    cap.release()
    out_extr.release()
    out_bin.release()
    log("Background subtraction finished (no feather).")


# ---------------------------------------------------------------------------#
#  CLI driver                                                                #
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    BASE_DIR = r"C:\Users\golan\Desktop\Contrib-Video"   # adjust if needed

    in_path            = os.path.join(BASE_DIR, "Outputs", "STAB.avi")
    out_path_extracted = os.path.join(BASE_DIR, "Outputs", "EXTRACTED.avi")
    out_path_binary    = os.path.join(BASE_DIR, "Outputs", "BINARY.avi")

    params = {
        "thresh":          36,
        "Erode":            5,
        "Dilate1":          3,
        "Dilate2":          6,
        "GaussianFiltSigma":3.0
    }

    print("=== Hard-edge Background Removal ===")
    video_background_removal(
        in_path, out_path_extracted, out_path_binary,
        params,
        textGUI=lambda m: print(m, end="\r", flush=True)
    )
    print("\nDone!  Outputs saved in:", os.path.dirname(out_path_extracted))
