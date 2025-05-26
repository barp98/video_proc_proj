import cv2
import numpy as np
import os

def extract_foreground(stabilized_path, output_extracted_path, output_binary_path):
    # Open the stabilized input video
    cap = cv2.VideoCapture(stabilized_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {stabilized_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Set up output videos
    out_extracted = cv2.VideoWriter(output_extracted_path, fourcc, fps, (frame_width, frame_height))
    out_binary = cv2.VideoWriter(output_binary_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Create the background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction to get mask
        fg_mask = back_sub.apply(frame)

        # Clean mask: morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        # Binarize: convert to 0 and 1
        _, binary_mask = cv2.threshold(fg_mask, 127, 1, cv2.THRESH_BINARY)

        # Extract person by masking the frame
        extracted = cv2.bitwise_and(frame, frame, mask=binary_mask.astype(np.uint8))

        # Save extracted frame
        out_extracted.write(extracted)

        # Save binary mask scaled to 0-255 for visualization
        out_binary.write((binary_mask * 255).astype(np.uint8))

    # Release everything
    cap.release()
    out_extracted.release()
    out_binary.release()
    print("Background subtraction complete. Videos saved.")

# Optional: Direct run for testing
if __name__ == "__main__":
    input_stabilized = os.path.join("..", "Outputs", "STABILIZED.avi")
    output_extracted = os.path.join("..", "Outputs", "extracted_ID1_ID2.avi")
    output_binary = os.path.join("..", "Outputs", "binary_ID1_ID2.avi")

    print("Looking for:", os.path.abspath(input_stabilized))
    extract_foreground(input_stabilized, output_extracted, output_binary)
