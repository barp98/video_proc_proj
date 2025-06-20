#!/usr/bin/env python3
"""
tracking.py – **Particle filter + mask‑guided re‑weighting**
===========================================================

This revision fuses the alpha‑matte detection into the particle filter so
boxes no longer drift or jitter.  Each frame provides a *measurement*
(the mask‑centroid), and particles are re‑weighted by **both** colour
similarity **and** positional agreement with that measurement.

Pipeline per frame
------------------
1. **Detection (alpha mask)** → tight bbox → measurement centre *(mx,my)*.
2. **Propagation** – Gaussian motion noise applied to *(cx,cy)*.
3. **Weighting**   –
   • *w_pos* = exp( −‖p−m‖² / 2σₘ² )   (σₘ = `MEAS_STD`).
   • *w_col* = exp( −β·Bhattacharyya² ).
   • *w* = *w_pos* × *w_col*.
4. **Estimation** – weighted average centre.
5. **Resampling** – systematic.

Hyper‑parameters can be tuned at the top of the file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

# ───────────────────────── default paths (patched by main.py) ───────────────
PROJECT    = Path(__file__).resolve().parents[1]
COLOR_PATH = PROJECT / "Outputs" / "matted_325106854_207234550.avi"
ALPHA_PATH = PROJECT / "Outputs" / "alpha_325106854_207234550.avi"
OUT_PATH   = PROJECT / "Outputs" / "OUTPUT_325106854_207234550.avi"

# ───────────────────────── hyper‑parameters ─────────────────────────────────
N_PARTICLES = 300            # particle count
MOTION_STD  = 12.0           # propagation noise σ (pixels)
MEAS_STD    = 25.0           # measurement (mask‑centre) σ
HIST_BINS   = (16, 16, 16)   # HSV histogram granularity
BETA        = 18.0           # colour weight sharpness
MASK_THR    = 25             # binary threshold on alpha matte

# ───────────────────────── utilities ───────────────────────────────────────

def _tight_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return -1, -1, 0, 0
    return int(ys.min()), int(xs.min()), int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)


def _hsv_hist(patch: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HIST_BINS, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.reshape(-1)


def _compare_hist(h1: np.ndarray, h2: np.ndarray) -> float:
    return cv2.compareHist(h1.astype("float32"), h2.astype("float32"), cv2.HISTCMP_BHATTACHARYYA)

# ───────────────────────── main routine ────────────────────────────────────

def generate_tracking(
    colour_path: Path | str = COLOR_PATH,
    alpha_path: Path | str = ALPHA_PATH,
    out_path: Path | str = OUT_PATH,
    *,
    n_particles: int = N_PARTICLES,
    motion_std: float = MOTION_STD,
    meas_std: float = MEAS_STD,
) -> Dict[int, Tuple[int, int, int, int]]:
    t0 = time.time()

    colour_path, alpha_path, out_path = map(Path, (colour_path, alpha_path, out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap_c = cv2.VideoCapture(str(colour_path))
    cap_a = cv2.VideoCapture(str(alpha_path))
    if not cap_c.isOpened():
        raise FileNotFoundError(colour_path)
    if not cap_a.isOpened():
        raise FileNotFoundError(alpha_path)

    fps = cap_c.get(cv2.CAP_PROP_FPS) or 30
    W   = int(cap_c.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap_c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw  = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (W, H))

    # ─── bootstrap on frame 0 ──────────────────────────────────────────
    ok, frame0 = cap_c.read(); oka, alpha0 = cap_a.read()
    if not (ok and oka):
        raise RuntimeError("Cannot read first frame(s)")

    mask0 = cv2.threshold(cv2.cvtColor(alpha0, cv2.COLOR_BGR2GRAY), MASK_THR, 255, cv2.THRESH_BINARY)[1]
    r0, c0, h0, w0 = _tight_bbox(mask0)
    if h0 == 0 or w0 == 0:
        raise RuntimeError("Empty mask in first frame – cannot bootstrap tracker.")

    target_hist = _hsv_hist(frame0[r0:r0 + h0, c0:c0 + w0])
    rng = np.random.default_rng()
    particles = np.column_stack([
        rng.normal(c0 + w0 / 2, motion_std, n_particles),
        rng.normal(r0 + h0 / 2, motion_std, n_particles),
    ])
    weights = np.ones(n_particles) / n_particles

    tracking: Dict[int, Tuple[int, int, int, int]] = {0: [r0, c0, h0, w0]}
    cv2.rectangle(frame0, (c0, r0), (c0 + w0, r0 + h0), (0, 255, 0), 2)
    vw.write(frame0)

    frame_idx = 1
    meas_var2 = (meas_std ** 2) * 2  # pre‑compute 2σ² denominator

    def _clip(pa: np.ndarray) -> None:
        pa[:, 0] = np.clip(pa[:, 0], w0 / 2, W - w0 / 2)
        pa[:, 1] = np.clip(pa[:, 1], h0 / 2, H - h0 / 2)

    bbox_from_centre = lambda cx, cy: (int(cy - h0 / 2), int(cx - w0 / 2), int(h0), int(w0))

    # ─── streaming loop ────────────────────────────────────────────────
    while True:
        ok, frame = cap_c.read(); oka, alpha = cap_a.read()
        if not ok:
            break

        # measurement from mask
        m_mask = cv2.threshold(cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY), MASK_THR, 255, cv2.THRESH_BINARY)[1]
        mr, mc, mh, mw = _tight_bbox(m_mask)
        if mh == 0 or mw == 0:
            # fallback to previous measurement
            mr, mc = tracking[frame_idx - 1][:2]
        mx, my = mc + mw / 2, mr + mh / 2

        # propagate
        particles += rng.normal(0, motion_std, particles.shape)
        _clip(particles)

        # weight
        for i, (cx, cy) in enumerate(particles):
            # positional weight
            dist2 = (cx - mx) ** 2 + (cy - my) ** 2
            w_pos  = np.exp(-dist2 / meas_var2)

            # colour weight
            y, x, h, w = bbox_from_centre(cx, cy)
            patch = frame[y:y + h, x:x + w]
            if patch.size == 0:
                w_col = 1e-3
            else:
                bh = _compare_hist(target_hist, _hsv_hist(patch))
                w_col = np.exp(-BETA * bh * bh)
            weights[i] = w_pos * w_col + 1e-8
        weights /= weights.sum()

        # state estimate
        cx_hat, cy_hat = (weights @ particles)
        y_hat, x_hat, h_hat, w_hat = bbox_from_centre(cx_hat, cy_hat)

        # resample
        cdf = np.cumsum(weights)
        u0 = rng.random() / n_particles
        idxs = np.searchsorted(cdf, u0 + np.arange(n_particles) / n_particles)
        particles = particles[idxs]
        weights.fill(1.0 / n_particles)

        # draw & record
        cv2.rectangle(frame, (x_hat, y_hat), (x_hat + w_hat, y_hat + h_hat), (0, 255, 0), 2)
        vw.write(frame)
        tracking[frame_idx] = [y_hat, x_hat, h_hat, w_hat]
        if frame_idx % 50 == 0:
            print(f"Tracked {frame_idx} frames…", end="\r", flush=True)
        frame_idx += 1

    # ─── teardown ────────────────────────────────────────────────────
    cap_c.release(); cap_a.release(); vw.release()
    with out_path.with_name("tracking.json").open("w", encoding="utf-8") as fp:
        json.dump(tracking, fp, indent=2)

    print(f"\n✓  Tracking complete in {time.time() - t0:.1f}s → {out_path.name}")
    return tracking

# ───────────────────────── CLI ─────────────────────────────────────────────
if __name__ == "__main__":
    generate_tracking()
