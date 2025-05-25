import os
import time
import json
import cv2
import numpy as np
from stabilize import stabilize_video

# === CONFIG ===
ID1, ID2 = "123456789", "987654321"  # Replace with your real IDs

# === PATH SETUP ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
INPUT_DIR = os.path.join(ROOT_DIR, "Inputs")
OUTPUT_DIR = os.path.join(ROOT_DIR, "Outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_PATH = os.path.join(INPUT_DIR, "INPUT.avi")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"stabilize_{ID1}_{ID2}.avi")

# === TIMER AND PROCESS ===
timing_data = {}
start_time = time.time()

# Step 1: Stabilization
stabilize_video(INPUT_PATH, OUTPUT_PATH)

timing_data["stabilize"] = time.time() - start_time

# === SAVE TIMING DATA ===
with open(os.path.join(OUTPUT_DIR, "timing.json"), "w") as f:
    json.dump(timing_data, f, indent=2)

# === MSE FUNCTIONS ===
def compute_mse(video_path1, video_path2, max_frames=100):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    total_mse = 0
    count = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2 or count >= max_frames:
            break

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        mse = np.mean((gray1.astype("float") - gray2.astype("float")) ** 2)
        total_mse += mse
        count += 1

    cap1.release()
    cap2.release()

    return total_mse / count if count else float('inf')

def compute_temporal_mse(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    _, prev = cap.read()
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    total_mse = 0
    count = 0

    while True:
        ret, curr = cap.read()
        if not ret or count >= max_frames:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        mse = np.mean((curr_gray.astype("float") - prev.astype("float")) ** 2)
        total_mse += mse
        prev = curr_gray
        count += 1

    cap.release()
    return total_mse / count if count else float('inf')

# === RUN MSE ANALYSIS ===
print("Evaluating stabilization quality with MSE...")

temporal_mse_input = compute_temporal_mse(INPUT_PATH)
temporal_mse_stabilized = compute_temporal_mse(OUTPUT_PATH)
framewise_mse = compute_mse(INPUT_PATH, OUTPUT_PATH)

print("\n===== MSE Report =====")
print(f"Temporal MSE (before stabilization):  {temporal_mse_input:.2f}")
print(f"Temporal MSE (after stabilization):   {temporal_mse_stabilized:.2f}")
print(f"Framewise MSE (input vs stabilized):  {framewise_mse:.2f}")
print("======================\n")

# === SAVE REPORT ===
with open(os.path.join(OUTPUT_DIR, "mse_report.txt"), "w") as f:
    f.write(f"Temporal MSE (input): {temporal_mse_input:.2f}\n")
    f.write(f"Temporal MSE (stabilized): {temporal_mse_stabilized:.2f}\n")
    f.write(f"Framewise MSE: {framewise_mse:.2f}\n")
