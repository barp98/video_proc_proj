# background_and_foreground_auto.py
# --------------------------------------------------------------------
# Median‑background builder + fully automatic foreground extractor
# **Improved 2025‑06‑13 – rev‑G**
#  • Now *always* keeps only the single largest connected component.
#    (No more SECONDARY_RATIO or small‑blob logic.)
#  • Threshold offset remains −8 and final close + dilate for outline.
# --------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
import cv2, numpy as np, random


THR_OFFSET = -8   # pixels brighter than (Otsu − 8) are foreground


def _log(msg: str, gui_cb: Optional[Callable[[str], None]]) -> None:
    gui_cb(msg) if gui_cb else print(msg)


# ────────────────────────── main worker ────────────────────────────────

def extract_person(
    video_path: str | Path,
    out_color_path: str | Path,
    out_bin_path: str | Path,
    textGUI: Optional[Callable[[str], None]] = None,
) -> None:
    """Extract the moving person from a stabilised video – *rev‑G* keeps
    exactly one contour (the biggest) per frame, eliminating residual noise
    from small detached parts."""

    # ── 0. meta ───────────────────────────────────────────────────────
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    nfrm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── 1. median background ──────────────────────────────────────────
    _log("Step 1/4: computing median background …", textGUI)
    stride = max(1, nfrm // 200)
    idx    = np.arange(0, nfrm, stride, dtype=int)
    bg_acc = np.empty((len(idx), H, W, 3), np.uint8)

    for i, f in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frm = cap.read()
        if not ok:
            continue
        bg_acc[i] = frm

    median_bg = np.median(bg_acc.astype(np.float32), 0).astype(np.uint8)
    cv2.imwrite(str(Path(out_color_path).with_name("MEDIAN_BG.png")), median_bg)

    # ── 2. auto‑param probe ───────────────────────────────────────────
    _log("Step 2/4: auto‑tuning …", textGUI)
    probe_frames = random.sample(range(nfrm), k=min(10, nfrm))
    areas       = []
    ksize       = max(3, (min(H, W) // 300) | 1)  # ≈ 0.3 % shorter dim
    kernel_big  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    kernel_dot  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_h15  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_close= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_dil  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    hsv_bg = cv2.cvtColor(median_bg, cv2.COLOR_BGR2HSV)

    for f in probe_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frm = cap.read()
        if not ok:
            continue
        diff_gray = cv2.cvtColor(cv2.absdiff(cv2.cvtColor(frm, cv2.COLOR_BGR2HSV), hsv_bg),
                                 cv2.COLOR_BGR2GRAY)
        thr, _ = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (diff_gray > max(1, thr + THR_OFFSET)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_big)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            areas.append(max(cv2.contourArea(c) for c in cnts))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    min_area = 0.06 * float(np.median(areas)) if areas else 0.0015 * W * H
    _log(f"  ↳ k={ksize}px, min_area≈{min_area:.0f}", textGUI)

    # ── 3. writers ────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw_col = cv2.VideoWriter(str(out_color_path), fourcc, fps, (W, H))
    vw_bin = cv2.VideoWriter(str(out_bin_path), fourcc, fps, (W, H), False)

    # ── 4. main pass ──────────────────────────────────────────────────
    _log("Step 3/4: extracting …", textGUI)
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        diff_gray = cv2.cvtColor(cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), hsv_bg),
                                 cv2.COLOR_BGR2GRAY)
        thr, _ = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (diff_gray > max(1, thr + THR_OFFSET)).astype(np.uint8) * 255

        # morphology: close → dilate → median → remove speckles/lines
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
        mask = cv2.dilate(mask, kernel_big, 1)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_dot)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_h15)

        # keep **only** the largest connected component
        num_lbl, lbl_img, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_lbl > 1:
            largest_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask[:] = 0
            mask[lbl_img == largest_lbl] = 255

        # final outline restore
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.dilate(mask, kernel_dil, 1)

        vw_col.write(cv2.bitwise_and(frame, frame, mask=mask))
        vw_bin.write(mask)

        f += 1
        if f % 50 == 0:
            _log(f"  {f}/{nfrm} …", textGUI)

    cap.release(); vw_col.release(); vw_bin.release()
    _log("Step 4/4: done!", textGUI)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    OUT  = ROOT / "Outputs"; OUT.mkdir(exist_ok=True, parents=True)
    extract_person(OUT / "STAB.avi", OUT / "extracted_auto.avi", OUT / "binary_auto.avi")
