import cv2
import numpy as np
import os
import json
import time

def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
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

        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.03, minDistance=20, blockSize=3)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        if prev_pts is None or curr_pts is None:
            continue

        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        if len(prev_pts) < 6 or len(curr_pts) < 6:
            continue

        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC, ransacReprojThreshold=3)
        if m is None:
            m = np.eye(2, 3)

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [dx, dy, da]
        prev_gray = curr_gray

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, frame = cap.read()
    out.write(frame)

    for i in range(n_frames - 1):
        success, frame = cap.read()
        if not success:
            break

        dx, dy, da = transforms_smooth[i]
        m = np.array([[np.cos(da), -np.sin(da), dx],
                      [np.sin(da),  np.cos(da), dy]])

        frame_stabilized = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out.write(frame_stabilized)

    cap.release()
    out.release()

def smooth(trajectory):
    smoothed = np.copy(trajectory)
    radius = 30
    for i in range(3):
        smoothed[:, i] = moving_average(trajectory[:, i], radius)
    return smoothed

def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.pad(curve, (radius, radius), 'edge')
    return np.convolve(f, np.ones(window_size)/window_size, mode='valid')

def save_timing_json(start_time, output_file):
    timing = {
        "stabilize_ID1_ID2.avi": round(time.time() - start_time, 2)
    }
    with open(output_file, 'w') as f:
        json.dump(timing, f)

def compute_avg_mse(video_path):
    cap = cv2.VideoCapture(video_path)
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    mse_list = []

    while True:
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr_gray, prev_gray)
        mse = np.mean(diff.astype(np.float32) ** 2)
        mse_list.append(mse)
        prev_gray = curr_gray

    cap.release()
    return np.mean(mse_list)

def main():
    input_file = os.path.join("..", "Inputs", "INPUT.avi")
    output_file = os.path.join("..", "Outputs", "stabilize_ID1_ID2.avi")
    timing_file = os.path.join("..", "Outputs", "timing.json")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    start_time = time.time()
    stabilize_video(input_file, output_file)
    save_timing_json(start_time, timing_file)

    mse_input = compute_avg_mse(input_file)
    mse_stab = compute_avg_mse(output_file)
    print("Avg frame-to-frame MSE - Input:", mse_input)
    print("Avg frame-to-frame MSE - Stabilized:", mse_stab)

if __name__ == '__main__':
    main()