import os
import time
import json
import cv2
import numpy as np

ID1 = '123456789'
ID2 = '987654321'

# --- Correct relative paths ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Code/
BASE_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..'))  # FinalProject_...

input_dir = os.path.join(BASE_DIR, 'Inputs')
output_dir = os.path.join(BASE_DIR, 'Outputs')
background_path = os.path.join(input_dir, 'background.jpg')
input_video_path = os.path.join(input_dir, 'INPUT.avi')

# Output filenames
stabilized_path = os.path.join(output_dir, f'stabilized_{ID1}_{ID2}.avi')
extracted_path = os.path.join(output_dir, f'extracted_{ID1}_{ID2}.avi')
binary_path = os.path.join(output_dir, f'binary_{ID1}_{ID2}.avi')
matted_path = os.path.join(output_dir, f'matted_{ID1}_{ID2}.avi')
alpha_path = os.path.join(output_dir, f'alpha_{ID1}_{ID2}.avi')
output_path = os.path.join(output_dir, f'OUTPUT_{ID1}_{ID2}.avi')
timing_json = os.path.join(output_dir, 'timing.json')
tracking_json = os.path.join(output_dir, 'tracking.json')

os.makedirs(output_dir, exist_ok=True)

# Timing log
timing = {}
start_time = time.time()

def write_video(filename, frames, fps, is_color=True):
    print(f"Writing video: {filename}")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=is_color)
    for frame in frames:
        out.write(frame if is_color else frame.astype(np.uint8))
    out.release()
    print(f"Finished writing video: {filename}")

# 1. Stabilization using advanced LK-based method
def stabilize_video(input_path):
    print("[INFO] Starting stabilization...")
    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), "Cannot open input video"
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=20, blockSize=3)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    original_frames = [prev.copy()]
    stabilized_frames = [prev.copy()]
    transforms = []

    for i in range(1, n_frames):
        ret, curr = cap.read()
        if not ret:
            print(f"[WARNING] Frame {i} could not be read, stopping early.")
            break
        print(f"[INFO] Estimating motion: Frame {i}/{n_frames}")
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        original_frames.append(curr.copy())

        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if prev_pts is None:
            transforms.append([0, 0, 0])
            prev_gray = curr_gray
            continue

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        m = cv2.estimateAffinePartial2D(good_prev, good_curr)[0]
        if m is None:
            dx, dy, da = 0, 0, 0
        else:
            dx = m[0, 2]
            dy = m[1, 2]
            da = np.arctan2(m[1, 0], m[0, 0])
        transforms.append([dx, dy, da])
        prev_gray = curr_gray

    cap.release()

    def smooth_trajectory(transforms, radius=5):
        smoothed = []
        for i in range(len(transforms)):
            start = max(0, i - radius)
            end = min(len(transforms), i + radius + 1)
            window = transforms[start:end]
            weights = np.exp(-0.5 * ((np.arange(len(window)) - (i - start)) / radius)**2)
            weights /= np.sum(weights)
            smooth = np.dot(weights, window)
            smoothed.append(smooth)
        return np.array(smoothed)

    print("[INFO] Smoothing camera trajectory...")
    trajectory = np.cumsum(transforms, axis=0)
    smoothed = smooth_trajectory(trajectory, radius=10)
    corrections = smoothed - trajectory
    print("[INFO] Trajectory smoothed.")

    cap = cv2.VideoCapture(input_path)
    ret, _ = cap.read()
    for i in range(len(corrections)):
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Could not read frame {i} during writing phase.")
            break
        print(f"[INFO] Stabilizing and writing frame {i+1}/{len(corrections)}")
        dx, dy, da = corrections[i]
        cos, sin = np.cos(da), np.sin(da)
        transform = np.array([[cos, -sin, dx], [sin, cos, dy]], dtype=np.float32)
        stabilized = cv2.warpAffine(frame, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        stabilized_frames.append(stabilized)

    cap.release()
    print("[INFO] Stabilization complete.")
    return stabilized_frames, fps

# Stabilize video
stabilized_frames, fps = stabilize_video(input_video_path)
write_video(stabilized_path, stabilized_frames, fps)
timing['stabilized'] = time.time() - start_time

# 2. Background Subtraction
print("[INFO] Starting enhanced background subtraction using median model...")
def subtract_background(frames):
    n = len(frames)
    indices = list(range(0, n, 50))  # sample every 50 frames
    selected = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in indices]
    background = np.median(np.stack(selected, axis=2), axis=2).astype(np.uint8)

    # Save median background for visualization
    background_img_path = os.path.join(output_dir, 'median_background.png')
    cv2.imwrite(background_img_path, background)
    print(f"[INFO] Saved median background to {background_img_path}")

    # Save median background for visualization (no duplicate and no gray_stack error)
    background_img_path = os.path.join(output_dir, 'median_background.png')
    cv2.imwrite(background_img_path, background)
    print(f"[INFO] Saved median background to {background_img_path}")

    extracted_frames = []
    binary_frames = []

    for i, frame in enumerate(frames):
        # Suppress motion at corners to reduce ghosting artifacts from posters and door edges
        corner_margin = 25  # size of corner area to suppress
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(background, gray)
        # Optional blur to suppress noise
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, binary = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)

        # Mask corners to avoid false positives in static high-detail zones
        h, w = binary.shape
        binary[:corner_margin, :] = 0
        binary[:, :corner_margin] = 0
        binary[:, -corner_margin:] = 0
        binary[-corner_margin:, :] = 0

        # Morphological cleaning
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        binary = cv2.dilate(binary, np.ones((13, 13), np.uint8), iterations=1)

        # First filtering: erode to clean small details
        binary_eroded = cv2.erode(binary, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(binary_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_clean = np.zeros_like(binary)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 1000:
                cv2.drawContours(mask_clean, [largest], -1, 255, -1)

        # Second filtering: multiply with original thresholded mask
        mask_clean = cv2.bitwise_and(mask_clean, binary)

        # Final dilation to merge split regions
        mask_clean = cv2.dilate(mask_clean, np.ones((7, 7), np.uint8), iterations=1)

        extracted = cv2.bitwise_and(frame, frame, mask=mask_clean)
        extracted_frames.append(extracted)
        binary_frames.append(mask_clean)

    print("[INFO] Enhanced background subtraction complete.")
    return extracted_frames, binary_frames

extracted_frames, binary_frames = subtract_background(stabilized_frames)
write_video(extracted_path, extracted_frames, fps)
write_video(binary_path, binary_frames, fps, is_color=False)
timing['extracted'] = time.time() - start_time

# 3. Matting
# ... (unchanged)

# 4. Tracking
# ... (unchanged)

# Save JSONs
# ... (unchanged)


    

# 3. Matting
print("Starting matting...")
background_img = cv2.imread(background_path)
background_resized = cv2.resize(background_img, (stabilized_frames[0].shape[1], stabilized_frames[0].shape[0]))
matted_frames = []
alpha_frames = []

for fg, mask in zip(extracted_frames, binary_frames):
    alpha = mask.astype(np.float32) / 255.0
    fg_float = fg.astype(np.float32) / 255.0
    bg_float = background_resized.astype(np.float32) / 255.0
    composite = alpha[..., None] * fg_float + (1 - alpha[..., None]) * bg_float
    matted_frames.append((composite * 255).astype(np.uint8))
    alpha_frames.append((alpha * 255).astype(np.uint8))

write_video(matted_path, matted_frames, fps)
write_video(alpha_path, alpha_frames, fps, is_color=False)
timing['matted'] = time.time() - start_time
print("Matting complete.")

# 4. Tracking
print("Starting tracking...")
tracker = cv2.TrackerCSRT_create()
init_bbox = None
tracking_data = {}
output_frames = []

for idx, frame in enumerate(matted_frames):
    if idx == 0:
        contours, _ = cv2.findContours(binary_frames[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        init_bbox = (x, y, w, h)
        tracker.init(frame, init_bbox)
        tracking_data[idx] = [y, x, h, w]
    else:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            tracking_data[idx] = [y, x, h, w]
    output_frames.append(frame)

write_video(output_path, output_frames, fps)
timing['tracking'] = time.time() - start_time
print("Tracking complete.")

# Save JSONs
print("Saving JSON files...")
with open(timing_json, 'w') as f:
    json.dump(timing, f)

with open(tracking_json, 'w') as f:
    json.dump(tracking_data, f)
print("All done!")
