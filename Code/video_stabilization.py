import cv2
import numpy as np
import os

# Paths
input_path = os.path.join('..', 'Inputs', 'INPUT2.avi')
output_path = os.path.join('..', 'Outputs', 'STABILIZED.mp4')

# Feature detection + LK optical flow params
feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=20, blockSize=3)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Load video
cap = cv2.VideoCapture(input_path)
assert cap.isOpened(), "Cannot open input video"
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# Read first frame
ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Lists for frames
original_frames = [prev.copy()]
stabilized_frames = [prev.copy()]
transforms = []

print("[INFO] Starting motion estimation...")

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

# === SMOOTHING WITH GAUSSIAN FILTER ===
print("[INFO] Smoothing camera trajectory...")
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

trajectory = np.cumsum(transforms, axis=0)
smoothed = smooth_trajectory(trajectory, radius=10)
corrections = smoothed - trajectory
print("[INFO] Trajectory smoothed.")

# === APPLY TRANSFORMS ===
cap = cv2.VideoCapture(input_path)
ret, _ = cap.read()

print("[INFO] Applying transformations and writing output...")

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
    out.write(stabilized)

cap.release()
out.release()

# === MSE BETWEEN FRAMES ===
print("[INFO] Computing MSE values...")

def mse_list(frames):
    total = 0
    for i in range(1, len(frames)):
        total += np.mean((frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)) ** 2)
    return total / (len(frames) - 1)

mse_orig = mse_list(original_frames)
mse_stab = mse_list(stabilized_frames)

print(f"\n[RESULT] Video stabilized and saved to {output_path}")
print(f"[RESULT] Original video frame-to-frame MSE:    {mse_orig:.2f}")
print(f"[RESULT] Stabilized video frame-to-frame MSE: {mse_stab:.2f}")
