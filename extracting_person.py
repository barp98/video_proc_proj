# Code/foreground_extraction.py
# -------------------------------------------------------------------
# Extract the foreground person from a stabilized video given a clean
# background image.  No GUI windows are opened; everything is written
# directly to Outputs/.
# -------------------------------------------------------------------
from pathlib import Path
import cv2
import numpy as np

def extract_foreground(
        video_path: str | Path,
        bg_path: str | Path,
        out_color_path: str | Path,
        out_bin_path: str | Path,
        thrsh_const: int = 35,           # pixel-difference threshold (0–255)
        morph_kernel: int = 5,           # square size for morphological ops
        min_area_frac: float = 0.02      # discard blobs smaller than 2 % of frame
    ) -> None:
    """
    Parameters
    ----------
    video_path      path to stabilized_ID1_ID2.avi
    bg_path         path to background.jpg  (clean background)
    out_color_path  path for extracted_ID1_ID2.avi (BGR, fourcc='MJPG')
    out_bin_path    path for binary_ID1_ID2.avi    (8-bit, 0/255, 'MJPG')
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare writers -------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vw_color = cv2.VideoWriter(str(out_color_path), fourcc, fps,   (W, H))
    vw_bin   = cv2.VideoWriter(str(out_bin_path),   fourcc, fps,   (W, H), isColor=False)

    # Load & resize background once ----------------------------------
    bg = cv2.imread(str(bg_path))
    if bg is None:
        raise FileNotFoundError(f"Cannot read {bg_path}")
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)

    # Pre-allocate for speed
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    min_area = min_area_frac * W * H

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1. Pixel-wise difference in HSV for modest illumination robustness
        diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                           cv2.cvtColor(bg,   cv2.COLOR_BGR2HSV))
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 2. Threshold  ------------------------------------------------
        #    If thrsh_const<0 use Otsu; otherwise fixed threshold
        if thrsh_const < 0:
            _, mask = cv2.threshold(diff_gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(diff_gray, thrsh_const, 255,
                                    cv2.THRESH_BINARY)

        # 3. Morphological clean-up -----------------------------------
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 4. Keep only the largest contour (removes noise & door/poster blobs)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:                                # guard against empty frame
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= min_area:
                mask[:] = 0                         # reset entire mask
                cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
            else:
                mask[:] = 0                         # no blob big enough

        # 5. Apply mask to get the extracted color frame ---------------
        extracted = cv2.bitwise_and(frame, frame, mask=mask)

        # 6. Write frames ---------------------------------------------
        vw_color.write(extracted)
        vw_bin.write(mask)          # still 0/255, but represents {0,1}

    # Release everything
    cap.release()
    vw_color.release()
    vw_bin.release()


# -------------------------------------------------------------------
# Convenience CLI  (so you can call this file directly if you wish)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Adjust the three paths below to your project layout if needed
    ROOT   = Path(__file__).resolve().parents[1]      # project root (…/Final_Project)
    INPUTS = ROOT / "Inputs"
    OUTPUT = ROOT / "Outputs"

    extract_foreground(
        video_path      = OUTPUT / "STAB.avi",
        bg_path         = OUTPUT / "MEDIAN_BG.png",
        out_color_path  = OUTPUT / "extracted_ID1_ID2.avi",
        out_bin_path    = OUTPUT / "binary_ID1_ID2.avi",
        thrsh_const     = 26,     # tweak here if clothes/background differ
        morph_kernel    = 3,
        min_area_frac   = 0.01
    )
