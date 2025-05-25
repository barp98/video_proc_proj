import cv2
import numpy as np
import os

def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_padded = np.pad(curve, (radius, radius), 'edge')
    return np.convolve(curve_padded, f, mode='valid')

def smooth_trajectory(trajectory, radius=30):
    smoothed = np.copy(trajectory)
    for i in range(3):  # dx, dy, da
        smoothed[:, i] = moving_average(trajectory[:, i], radius)
    return smoothed

def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 1):
        success, curr = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        m, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
        if m is None:
            m = np.eye(2, 3)

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        transforms[i] = [dx, dy, da]

        prev_gray = curr_gray

    # Compute trajectory and smooth it
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth_trajectory(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    # Reset and apply transforms
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, first_frame = cap.read()
    out.write(first_frame)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    for i in range(n_frames - 1):
        success, frame = cap.read()
        if not success:
            break

        dx, dy, da = transforms_smooth[i]
        m = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da),  np.cos(da), dy]
        ])

        stabilized = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out.write(stabilized)

    cap.release()
    out.release()
