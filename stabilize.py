"""
stabilize_and_score.py  –  Stabilise a video and report MSE improvement
======================================================================

Steps
-----
1.  Run the cumulative-similarity stabiliser (identical to your MATLAB flow).
2.  Re-open *both* the raw and the stabilised AVI files.
3.  For every consecutive pair of frames in each stream, compute
       MSE = mean( (frame[k] – frame[k–1])² )
    on 8-bit grayscale images.
4.  Print the **average** MSE for the original and the stabilised video,
   plus the improvement factor.

Dependencies:  opencv-python ≥ 4.5,  numpy
"""
from __future__ import annotations
import cv2
import numpy as np
import os
from typing import Callable, Dict, Optional, Any, Tuple

# ----------------------------------------------------------------------
#  --- (1) Similarity-transform estimator  -----------------------------
# ----------------------------------------------------------------------
def estimate_srt(prev_gray: np.ndarray,
                 curr_gray: np.ndarray,
                 *,
                 max_corners: int   = 200,
                 quality: float     = 0.01,
                 min_distance: int  = 30,
                 ransac_thresh: float = 3.0) -> np.ndarray:
    """Return 3×3 homography that warps *curr_gray* onto *prev_gray*."""
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners= int(max_corners * 0.5),      # up to 600 – 800 pts per frame
        qualityLevel= quality,      # accept weaker corners
        minDistance= min_distance,  # allow denser sampling
)
    if prev_pts is None:
        return np.eye(3, dtype=np.float32)

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                   curr_gray,
                                                   prev_pts,
                                                   None)
    good = (status.squeeze() == 1)
    if np.count_nonzero(good) < 4:
        return np.eye(3, dtype=np.float32)

    prev_pts = prev_pts[good].reshape(-1, 2)
    curr_pts = curr_pts[good].reshape(-1, 2)

    H_affine, _ = cv2.estimateAffinePartial2D(
        curr_pts,          # src  (from)
        prev_pts,          # dst  (to)
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000,
        confidence=0.99
    )
    if H_affine is None:
        return np.eye(3, dtype=np.float32)

    return np.vstack([H_affine, [0.0, 0.0, 1.0]]).astype(np.float32)


# ----------------------------------------------------------------------
#  --- (2) Video stabiliser  ------------------------------------------
# ----------------------------------------------------------------------
def video_stabilization(in_path: str,
                        out_path: str,
                        params: Optional[Dict[str, Any]] = None,
                        textGUI: Optional[Callable[[str], None]] = None) -> None:
    if params is None:
        params = {}
    gauss_sigma  = params.get('gauss_sigma' , 1.0)
    max_corners  = params.get('max_corners' , 200)
    quality      = params.get('quality'     , 0.01)
    min_distance = params.get('min_distance', 30)
    ransac_thr   = params.get('ransac_thresh', 3.0)

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input video not found: {in_path}")

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {in_path}")

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  = cap.get(cv2.CAP_PROP_FPS)
    nfrm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(out_path,
                             cv2.VideoWriter_fourcc(*'XVID'),
                             fps, (w, h))

    if textGUI:
        textGUI(f"Loaded {nfrm} frames ({w}×{h})")

    # --- bootstrap ---
    ok, prev_bgr = cap.read()
    if not ok:
        raise RuntimeError("Failed to read first frame.")
    writer.write(prev_bgr)

    prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY),
                                 (0, 0), gauss_sigma)
    H_cum = np.eye(3, dtype=np.float32)

    # --- main loop ---
    for idx in range(1, nfrm):
        ok, curr_bgr = cap.read()
        if not ok:
            break
        if textGUI:
            textGUI(f"Processing frame {idx+1} / {nfrm}")

        curr_gray = cv2.GaussianBlur(cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY),
                                     (0, 0), gauss_sigma)

        H = estimate_srt(prev_gray, curr_gray,
                         max_corners=max_corners,
                         quality=quality,
                         min_distance=min_distance,
                         ransac_thresh=ransac_thr)
        H_cum = H @ H_cum

        stab = cv2.warpAffine(curr_bgr, H_cum[:2, :], (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        writer.write(stab)
        prev_gray = curr_gray

    cap.release()
    writer.release()
    if textGUI:
        textGUI("Stabilisation finished.")


# ----------------------------------------------------------------------
#  --- (3) MSE meter  --------------------------------------------------
# ----------------------------------------------------------------------
def average_frame_mse(video_path: str) -> Tuple[float, int]:
    """
    Compute average MSE between consecutive grayscale frames.
    Returns (avg_mse, number_of_pairs).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("No frames in video.")
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mse_sum = 0.0
    pairs   = 0

    while True:
        ok, curr = cap.read()
        if not ok:
            break
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mse_sum += np.mean((curr - prev) ** 2)
        pairs   += 1
        prev     = curr

    cap.release()
    avg = mse_sum / pairs if pairs else float("nan")
    return avg, pairs


# ----------------------------------------------------------------------
#  --- (4) Driver  -----------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    IN_PATH  = r"C:\Users\Bar\Desktop\University\semester_8\video_processing\final_project\Inputs\IMG_4197.avi"
    OUT_PATH = r"C:\Users\Bar\Desktop\University\semester_8\video_processing\final_project\Outputs\STAB.avi"

    def gui(msg: str) -> None:
        print(msg, end="\r")

    print("=== 1) Stabilising …")
    video_stabilization(IN_PATH, OUT_PATH, textGUI=gui)

    print("\n=== 2) Computing MSE …")
    mse_raw,  n1 = average_frame_mse(IN_PATH)
    mse_stab, n2 = average_frame_mse(OUT_PATH)

    print(f"\nFrames compared : {n1} (raw)  /  {n2} (stabilised)")
    print(f"Average MSE     : {mse_raw:8.2f}  →  {mse_stab:8.2f}")
    if np.isfinite(mse_raw) and mse_raw > 0:
        print(f"Improvement     : ×{mse_raw / mse_stab:5.2f}")
