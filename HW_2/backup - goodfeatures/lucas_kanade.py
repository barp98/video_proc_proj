import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata



# FILL IN YOUR ID
ID1 = 123456789
ID2 = 987654321


PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.

    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.

    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.

    Returns:
        pyramid: list. A list of np.ndarray of images.

    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    pyramid = [image.copy()]  # Original image is the first level

    for i in range(num_levels):
        smoothed = signal.convolve2d(pyramid[-1], PYRAMID_FILTER, mode='same', boundary='symm')
        decimated = smoothed[::2, ::2]
        pyramid.append(decimated)
        #print(f"[Pyramid] Level {i + 1}: shape = {decimated.shape}")

    return pyramid

def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform one Lucas-Kanade Step.

    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.

    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixel’s neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same')
    It = I2 - I1

    h, w = I1.shape
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)

    # In order to stay inside the image borders.
    border_index = window_size // 2

    for i in range(border_index, h - border_index):
        for j in range(border_index, w - border_index):
            Ix_vec = Ix[i - border_index:i + border_index + 1,
                        j - border_index:j + border_index + 1].flatten()
            Iy_vec = Iy[i - border_index:i + border_index + 1,
                        j - border_index:j + border_index + 1].flatten()
            It_vec = It[i - border_index:i + border_index + 1,
                        j - border_index:j + border_index + 1].flatten()

            B = np.array((Ix_vec, Iy_vec), dtype=np.float64).T
            try:
                dp = np.matmul(
                        np.matmul(
                            -np.linalg.inv(np.matmul(B.T, B)),
                            B.T),
                        It_vec)
                du[i, j], dv[i, j] = dp
            except np.linalg.LinAlgError:
                continue

    return du, dv

def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """
    original_image = image.copy()
    original_I_shape = image.shape

    # If needed, normalize and resize u and v to match image shape
    if u.shape != original_I_shape:
        normalize_u_factor = original_I_shape[1] / u.shape[1]
        u = normalize_u_factor * cv2.resize(u, image.T.shape)
        normalize_v_factor = original_I_shape[0] / v.shape[0]
        v = normalize_v_factor * cv2.resize(v, image.T.shape)

    h, w = image.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    gridpoints = np.concatenate(
        (np.expand_dims(rows.flatten(), 1),
         np.expand_dims(cols.flatten(), 1)),
        axis=1
    )

    # Add the flow vectors
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    gridpoints_to_interpolate = np.concatenate(
        (np.expand_dims((rows + v).flatten(), 1),
         np.expand_dims((cols + u).flatten(), 1)),
        axis=1
    )

    # Bi-linear interpolation
    interpolation = griddata(gridpoints,
                             image.flatten(),
                             gridpoints_to_interpolate,
                             method='linear',
                             fill_value=np.nan,
                             rescale=False)

    image_warp = interpolation.reshape(image.shape)

    # Fill "holes" with original image values
    image_warp[np.isnan(image_warp)] = original_image[np.isnan(image_warp)]

    return image_warp



def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.
    """
    # Resize image to match pyramid assumptions
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels))))
    IMAGE_SIZE = (w_factor * (2 ** num_levels), h_factor * (2 ** num_levels))

    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)

    # Create image pyramids
    Pyr1 = build_pyramid(I1, num_levels)
    Pyr2 = build_pyramid(I2, num_levels)

    # Start from zero flow at coarsest level
    u = np.zeros(Pyr1[num_levels].shape)
    v = np.zeros(Pyr1[num_levels].shape)

    for level in range(num_levels, 0, -1):
        I2_warp = warp_image(Pyr2[level], u, v)
        for step in range(max_iter):
            du, dv = lucas_kanade_step(Pyr1[level], I2_warp, window_size)
            u += du
            v += dv
            I2_warp = warp_image(Pyr2[level], u, v)

        # Scale flow up to next pyramid level
        u = 2 * cv2.resize(u, (Pyr1[level - 1].shape[1], Pyr1[level - 1].shape[0]))
        v = 2 * cv2.resize(v, (Pyr1[level - 1].shape[1], Pyr1[level - 1].shape[0]))

    return u, v



def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int) -> None:
    """Use LK Optical Flow to stabilize the video and save it to file."""

    # (1) Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # <<<<<< FIXED HERE!

    # (2) Open output video writer (now in color)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

    # (3) Read the first frame, convert to grayscale
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read first frame.")
        cap.release()
        out.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    out.write(cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR))  # <<<<<< FIXED HERE!

    # (4) Resize first frame to match pyramid constraints
    h_factor = int(np.ceil(prev_gray.shape[0] / (2 ** (num_levels))))
    w_factor = int(np.ceil(prev_gray.shape[1] / (2 ** (num_levels))))
    IMAGE_SIZE = (w_factor * (2 ** num_levels), h_factor * (2 ** num_levels))
    prev_gray = cv2.resize(prev_gray, IMAGE_SIZE)

    # (5) Initialize flow accumulators
    u_cumulative = np.zeros(IMAGE_SIZE[::-1])
    v_cumulative = np.zeros(IMAGE_SIZE[::-1])

    # (6) Loop over frames
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, IMAGE_SIZE)

        # (6.2) Optical flow between prev and curr
        u, v = lucas_kanade_optical_flow(prev_gray, curr_gray,
                                         window_size=window_size,
                                         max_iter=max_iter,
                                         num_levels=num_levels)

        # (6.3) Compute mean flow in valid region (exclude borders)
        half_win = window_size // 2
        valid_u = u[half_win:-half_win, half_win:-half_win]
        valid_v = v[half_win:-half_win, half_win:-half_win]

        mean_u = np.mean(valid_u)
        mean_v = np.mean(valid_v)

        # (6.4) Replace with mean
        u.fill(mean_u)
        v.fill(mean_v)

        # (6.5) Accumulate
        u_cumulative += u
        v_cumulative += v

        # (6.7) Warp current frame
        stabilized = warp_image(curr_gray, u_cumulative, v_cumulative)

        # Resize back to original size
        stabilized_resized = cv2.resize(stabilized, (width, height))
        out.write(cv2.cvtColor(np.uint8(stabilized_resized), cv2.COLOR_GRAY2BGR))  # <<<<<< FIXED HERE!

        # (6.6) Update previous
        prev_gray = curr_gray.copy()

        # Optional debug save:
        # cv2.imwrite(f"debug/frame_{pbar.n:04d}.png", stabilized)

        pbar.update(1)

    # (7) Graceful shutdown
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pbar.close()



import time

def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Faster implementation of a single Lucas-Kanade Step."""

    SMALL_IMAGE_THRESHOLD = 5000
    if I1.size <= SMALL_IMAGE_THRESHOLD:
        return lucas_kanade_step(I1, I2, window_size)

    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)

    # Compute gradients
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same', boundary='symm')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same', boundary='symm')
    It = I2 - I1

    # Harris corner detection
    corners = cv2.goodFeaturesToTrack(I2.astype(np.float32), maxCorners=60,
                                      qualityLevel=0.01, minDistance=window_size)
    if corners is None:
        return du, dv

    corners = np.round(corners[:, 0, :]).astype(int)
    half_w = window_size // 2
    h, w = I1.shape

    for (x, y) in corners:
        if x < half_w or x >= w - half_w or y < half_w or y >= h - half_w:
            continue

        Ix_window = Ix[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1].flatten()
        Iy_window = Iy[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1].flatten()
        It_window = It[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1].flatten()

        B = np.vstack((Ix_window, Iy_window)).T
        try:
            dp = np.linalg.lstsq(B, -It_window, rcond=None)[0]
            du[y, x], dv[y, x] = dp
        except np.linalg.LinAlgError:
            continue

    return du, dv


def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow using sparse step, multiple iterations per level,
    but only one warp per pyramid level (optimization)."""

    # Resize image to match pyramid assumptions
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels))))
    IMAGE_SIZE = (w_factor * (2 ** num_levels), h_factor * (2 ** num_levels))

    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)

    # Create image pyramids
    Pyr1 = build_pyramid(I1, num_levels)
    Pyr2 = build_pyramid(I2, num_levels)

    # Start from zero flow at coarsest level
    u = np.zeros(Pyr1[num_levels].shape)
    v = np.zeros(Pyr1[num_levels].shape)

    for level in range(num_levels, 0, -1):
        I2_warp = warp_image(Pyr2[level], u, v)
        for step in range(max_iter):
            du, dv = faster_lucas_kanade_step(Pyr1[level], I2_warp, window_size)
            u += du
            v += dv
            I2_warp = warp_image(Pyr2[level], u, v)

        # Scale flow up to next pyramid level
        u = 2 * cv2.resize(u, (Pyr1[level - 1].shape[1], Pyr1[level - 1].shape[0]))
        v = 2 * cv2.resize(v, (Pyr1[level - 1].shape[1], Pyr1[level - 1].shape[0]))

    return u, v

def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    # (1) Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # <<<<<< FIXED HERE!

    # (2) Open output video writer (now in color)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

    # (3) Read the first frame, convert to grayscale
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read first frame.")
        cap.release()
        out.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    out.write(cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR))  # <<<<<< FIXED HERE!

    # (4) Resize first frame to match pyramid constraints
    h_factor = int(np.ceil(prev_gray.shape[0] / (2 ** (num_levels))))
    w_factor = int(np.ceil(prev_gray.shape[1] / (2 ** (num_levels))))
    IMAGE_SIZE = (w_factor * (2 ** num_levels), h_factor * (2 ** num_levels))
    prev_gray = cv2.resize(prev_gray, IMAGE_SIZE)

    # (5) Initialize flow accumulators
    u_cumulative = np.zeros(IMAGE_SIZE[::-1])
    v_cumulative = np.zeros(IMAGE_SIZE[::-1])

    # (6) Loop over frames
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, IMAGE_SIZE)

        # (6.2) Optical flow between prev and curr
        u, v = faster_lucas_kanade_optical_flow(prev_gray, curr_gray,
                                         window_size=window_size,
                                         max_iter=max_iter,
                                         num_levels=num_levels)

        # (6.3) Compute mean flow in valid region (exclude borders)
        half_win = window_size // 2
        valid_u = u[half_win:-half_win, half_win:-half_win]
        valid_v = v[half_win:-half_win, half_win:-half_win]

        mean_u = np.mean(valid_u)
        mean_v = np.mean(valid_v)

        # (6.4) Replace with mean
        u.fill(mean_u)
        v.fill(mean_v)

        # (6.5) Accumulate
        u_cumulative += u
        v_cumulative += v

        # (6.7) Warp current frame
        stabilized = warp_image(curr_gray, u_cumulative, v_cumulative)

        # Resize back to original size
        stabilized_resized = cv2.resize(stabilized, (width, height))
        out.write(cv2.cvtColor(np.uint8(stabilized_resized), cv2.COLOR_GRAY2BGR))  # <<<<<< FIXED HERE!

        # (6.6) Update previous
        prev_gray = curr_gray.copy()

        # Optional debug save:
        # cv2.imwrite(f"debug/frame_{pbar.n:04d}.png", stabilized)

        pbar.update(1)

    # (7) Graceful shutdown
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pbar.close()


def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
    """Calculate LK Optical Flow to stabilize the video and crop stabilization artifacts.

    Args:
        input_video_path: str. Path to input video.
        output_video_path: str. Path to output stabilized video.
        window_size: int. Size of window used in LK.
        max_iter: int. Number of LK iterations per level.
        num_levels: int. Number of pyramid levels.
        start_rows: int. Top rows to crop after warp.
        end_rows: int. Bottom rows to crop after warp.
        start_cols: int. Left columns to crop after warp.
        end_cols: int. Right columns to crop after warp.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Open output video writer (still original size after crop + resize)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read first frame.")
        cap.release()
        out.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    out.write(cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR))

    h_factor = int(np.ceil(prev_gray.shape[0] / (2 ** (num_levels))))
    w_factor = int(np.ceil(prev_gray.shape[1] / (2 ** (num_levels))))
    IMAGE_SIZE = (w_factor * (2 ** num_levels), h_factor * (2 ** num_levels))
    prev_gray = cv2.resize(prev_gray, IMAGE_SIZE)

    u_cumulative = np.zeros(IMAGE_SIZE[::-1])
    v_cumulative = np.zeros(IMAGE_SIZE[::-1])

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, IMAGE_SIZE)

        u, v = faster_lucas_kanade_optical_flow(prev_gray, curr_gray,
                                                window_size=window_size,
                                                max_iter=max_iter,
                                                num_levels=num_levels)

        half_win = window_size // 2
        valid_u = u[half_win:-half_win, half_win:-half_win]
        valid_v = v[half_win:-half_win, half_win:-half_win]
        mean_u = np.mean(valid_u)
        mean_v = np.mean(valid_v)
        u.fill(mean_u)
        v.fill(mean_v)

        u_cumulative += u
        v_cumulative += v

        stabilized = warp_image(curr_gray, u_cumulative, v_cumulative)

        # Crop borders to reduce black artifacts
        cropped = stabilized[start_rows:-end_rows, start_cols:-end_cols]
        resized_back = cv2.resize(cropped, (width, height))
        out.write(cv2.cvtColor(np.uint8(resized_back), cv2.COLOR_GRAY2BGR))

        prev_gray = curr_gray.copy()
        pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pbar.close()



