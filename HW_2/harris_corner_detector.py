"""Harris Corner Detector."""
import numpy as np
import cv2
from scipy import signal
from matplotlib import pyplot as plt


# replace these with your IDs:
ID1 = '325106854'
ID2 = '207234550'

# Harris corner detector parameters - you may change them.
K = 0.05
CHECKERBOARD_THRESHOLD = 2e9
GIRAFFE_THRESHOLD = 1e8
BUTTERFLY_IMAGE = 'butterfly.jpg'

# Do not change the following constants:
# input images:
CHECKERBOARD_IMAGE = 'checkerboard.jpg'
GIRAFFE_IMAGE = 'giraffe.jpg'
# result images:
TEST_BLOCKS_FUNCTIONS_IMAGE = f'{ID1}_{ID2}_test_tiles_funcs.png'
IMAGE_AND_CORNERS = f'{ID1}_{ID2}_image_corners.png'
RESPONSE_BW_IMAGE = f'{ID1}_{ID2}_response_black_and_white.png'
RESPONSE_RGB_IMAGE = f'{ID1}_{ID2}_response_rgb.png'


def bgr_image_to_rgb_image(bgr_image):
    """Convert Blue-Green-Red image to Red-Green-Blue image.

    Args:
        bgr_image: np.ndarray of shape: (height, width, 3).

    Returns:
        rgb_image: np.ndarray of shape: (height, width, 3). Take the input
        image and in the third dimension, swap the first and last slices.
    """
    
    rgb_image = bgr_image.copy()
    rgb_image = bgr_image[:, :, [2, 1, 0]]
    """INSERT YOUR CODE HERE."""
    return rgb_image


def black_and_white_image_to_tiles(arr, nrows, ncols):
    """Convert the image to a series of non-overlapping nrowsXncols tiles.

    Args:
        arr: np.ndarray of shape (h, w).
        nrows: the number of rows in each tile.
        ncols: the number of columns in each tile.
    Returns:
        ((h//nrows) * (w //ncols) , nrows, ncols) np.ndarray.
    Hint: Use only shape, reshape and swapaxes to implement this method.
    Take inspiration from: https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays
    """


    """INSERT YOUR CODE HERE.
    REPLACE THE RETURNED VALUE WITH YOUR OWN IMPLEMENTATION.
    """
    h, w = arr.shape
    if h % nrows != 0 or w % ncols != 0:
        raise ValueError("Image dimensions must be evenly divisible by tile size.")

    reshaped = arr.reshape(h // nrows, nrows, w // ncols, ncols)
    swapped = reshaped.swapaxes(1, 2)
    tiles = swapped.reshape(-1, nrows, ncols)

    return tiles


def image_tiles_to_black_and_white_image(arr, h, w):
    """Convert the series of tiles back to a hxw image.

    Args:
        arr: np.ndarray of shape (nTiles, nRows, nCols).
        h: the height of the original image.
        w: the width of the original image.
    Returns:
        (h, w) np.ndarray.
    Hint: Use only shape, reshape and swapaxes to implement this method.
    Take inspiration from: https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays
    """
    n, nrows, ncols = arr.shape
    tile_rows = h // nrows
    tile_cols = w // ncols

    # Step 1: reshape to grid of tiles
    reshaped = arr.reshape(tile_rows, tile_cols, nrows, ncols)

    # Step 2: reverse the earlier swapaxes to align pixel layout
    swapped = reshaped.swapaxes(1, 2)

    # Step 3: reshape back to full image
    new_image = swapped.reshape(h, w)

    return new_image


def test_tiles_functions(to_save=False):
    """Show the butterfly image, its split to tiles and the reassembled
    image from tiles back to image."""
    butterfly_image = cv2.imread(BUTTERFLY_IMAGE, 0)
    plt.subplot(1, 3, 1)
    plt.title('original image')
    plt.imshow(butterfly_image, cmap='gray')
    plt.colorbar()
    tiles = black_and_white_image_to_tiles(butterfly_image, 25, 25)
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 12, 8 * i + j + 1 + 4 * (i + 1))
            plt.imshow(tiles[4 * i + j], cmap='gray')
            plt.title(f'tile #{4 * i + j + 1}')
            plt.xticks([])
            plt.yticks([])

    height, width = butterfly_image.shape
    reassembled_image = image_tiles_to_black_and_white_image(tiles, height,
                                                              width)
    plt.subplot(1, 3, 3)
    plt.title('re-assembled image')
    plt.imshow(reassembled_image, cmap='gray')

    fig = plt.gcf()
    fig.set_size_inches((20, 7))
    if to_save:
        plt.savefig(TEST_BLOCKS_FUNCTIONS_IMAGE)
    else:
        plt.show()


def create_grad_x_and_grad_y(input_image):
    """Calculate the gradients across the x and y-axes.

    Args:
        input_image: np.ndarray. Image array.
    Returns:
        tuple (Ix, Iy): The first is the gradient across the x-axis and the
        second is the gradient across the y-axis.

    Recipe:
    If the image is an RGB image, convert it to grayscale using OpenCV's
    cvtColor. Otherwise, the input image is already in grayscale.
    Then, create a one pixel shift (to the right) image and fill the first
    column with zeros.
    Ix will be the difference between the grayscale image and the shifted
    image.
    Iy will be obtained in a similar manner, this time you're requested to
    shift the image from top to bottom by 1 row. Fill the first row with zeros.
    Finally, in order to ignore edge pixels, remove the first column from Ix
    and the first row from Iy.
    Return (Ix, Iy).
    """
    # Get image dimensions
    if len(input_image.shape) == 2:
        # this is the case of a black and white image
        nof_color_channels = 1
        height, width = input_image.shape

    else:
        # this is the case of an RGB image
        nof_color_channels = 3
        height, width, _ = input_image.shape

    """INSERT YOUR CODE HERE.
    REPLACE THE VALUES FOR Ix AND Iy WITH THE GRADIENTS YOU COMPUTED.
    """
    if len(input_image.shape) == 2:
        gray_image = input_image.copy()
    else:
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    gray = gray_image.astype(np.float32)

    # Compute Ix: gradient along x-axis (horizontal)
    shifted_x = np.zeros_like(gray)
    shifted_x[:, 1:] = gray[:, :-1]
    Ix = gray - shifted_x
    Ix[:, 0] = 0  # zero out first column

    # Compute Iy: gradient along y-axis (vertical)
    shifted_y = np.zeros_like(gray)
    shifted_y[1:, :] = gray[:-1, :]
    Iy = gray - shifted_y
    Iy[0, :] = 0  # zero out first row

    return Ix, Iy


def calculate_response_image(input_image: np.ndarray, K: float) -> np.ndarray:
     # Compute Ix and Iy using your provided function
    Ix, Iy = create_grad_x_and_grad_y(input_image)

    # Compute Ix^2, Iy^2, and Ix*Iy
    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    Ixy = np.multiply(Ix, Iy)

    # Define 5x5 kernel of ones
    g = np.ones((5, 5), dtype=np.float32)

    # Convolve with kernel using signal.convolve2d
    Sxx = signal.convolve2d(Ix2, g, mode='same')
    Syy = signal.convolve2d(Iy2, g, mode='same')
    Sxy = signal.convolve2d(Ixy, g, mode='same')

    # Compute determinant and trace of the structure tensor
    det_M = np.multiply(Sxx, Syy) - np.square(Sxy)
    trace_M = Sxx + Syy

    # Compute Harris response
    response_image = det_M - K * np.square(trace_M)

    return response_image



def our_harris_corner_detector(input_image, K, threshold):
    """Calculate the corners for input image with parameters K and threshold.
    Args:
        input_image: np.ndarray. Image array.
        K: float. the K from the equation: R ≈ det(M) −k∙[trace M] ^2
        threshold: float. minimal response value for a point to be detected
        as a corner.
    Returns:
        output_image: np.ndarray with the height and width of the input
        image. This should be a binary image with all zeros except from ones
        in pixels with corners.
    """

    # (1) calculate the response image from the input image and the parameter K.
    response_image = calculate_response_image(input_image, K)

    # Save original shape for cropping later
    h_orig, w_orig = response_image.shape

    # (2.1) Pad the image so it's divisible by 25x25
    tile_height, tile_width = 25, 25
    pad_h = (tile_height - (h_orig % tile_height)) % tile_height
    pad_w = (tile_width - (w_orig % tile_width)) % tile_width
    padded_response = np.pad(response_image, ((0, pad_h), (0, pad_w)), mode='constant')
    h_padded, w_padded = padded_response.shape

    # (2.1 continued) convert the padded response image to tiles
    tiles = black_and_white_image_to_tiles(padded_response, tile_height, tile_width)

    # (2.2) Non-Max Suppression: keep only max value per tile
    suppressed_tiles = np.zeros_like(tiles)
    for idx in range(len(tiles)):
        tile = tiles[idx]
        if tile.size > 0:
            max_idx = np.argmax(tile)
            max_pos = np.unravel_index(max_idx, tile.shape)
            suppressed_tiles[idx][max_pos] = tile[max_pos]

    # (3) Convert the result tiles-tensor back to an image.
    suppressed_image = image_tiles_to_black_and_white_image(suppressed_tiles, h_padded, w_padded)

    # Crop back to original shape (undo the padding)
    suppressed_image = suppressed_image[:h_orig, :w_orig]

    # (4) Threshold to generate the binary corner output
    output_image = np.zeros_like(suppressed_image)
    output_image[suppressed_image > threshold] = 1

    return output_image






def plot_response_for_black_an_white_image(input_image, response_image,
                                           to_save=False):
    """Plot the original black and white image, the response image and a
    Zoom-in on an interesting region."""
    plt.clf()
    plt.subplot(1, 3, 1)
    plt.title('original image')
    plt.imshow(input_image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('response image')
    plt.imshow(response_image, cmap='jet')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('response image - zoom in\n on (130:170, 230:270)')
    plt.imshow(response_image[130:170, 230:270], cmap='jet')
    plt.colorbar()
    fig = plt.gcf()
    fig.set_size_inches((14, 7))
    if to_save:
        plt.savefig(RESPONSE_BW_IMAGE)
    else:
        plt.show()


def plot_response_for_rgb_image(input_image, response_image, to_save=False):
    """Plot the original RGB image, the response image and a Zoom-in on an
    interesting region."""
    plt.clf()
    plt.subplot(1, 3, 1)
    plt.title('original image')
    plt.imshow(bgr_image_to_rgb_image(input_image))
    plt.subplot(1, 3, 2)
    plt.title('response image')
    plt.imshow(response_image, cmap='jet')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('response image - zoom in\n on (40:120, 420:500)')
    plt.imshow(response_image[40:120, 420:500], cmap='jet')
    plt.colorbar()
    fig = plt.gcf()
    fig.set_size_inches((14, 7))
    if to_save:
        plt.savefig(RESPONSE_RGB_IMAGE)
    else:
        plt.show()


def create_corner_plots(black_and_white_image, black_and_white_image_corners,
                        grb_image, rgb_image_corners, to_save=False):
    """Plot the two images with the corners in the same plot."""
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(black_and_white_image, cmap='gray')
    corners = np.where(black_and_white_image_corners == 1)
    plt.plot(corners[1], corners[0], 'ro')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    rgb_image = bgr_image_to_rgb_image(grb_image)
    plt.imshow(rgb_image)
    corners = np.where(rgb_image_corners == 1)
    plt.plot(corners[1], corners[0], 'ro')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches((14, 7))
    if to_save:
        plt.savefig(IMAGE_AND_CORNERS)
    else:
        plt.show()


def main(to_save=True):
    test_tiles_functions(to_save)
    # Read checkerboard image as grayscale
    checkerboard = cv2.imread(CHECKERBOARD_IMAGE, 0)
    # Read giraffe image
    giraffe = cv2.imread(GIRAFFE_IMAGE)

    # checkerboard response image
    checkerboard_response_image = calculate_response_image(checkerboard, K)
    plot_response_for_black_an_white_image(checkerboard,
                                           checkerboard_response_image,
                                           to_save)

    # giraffe response image
    giraffe_response_image = calculate_response_image(giraffe, K)
    plot_response_for_rgb_image(giraffe, giraffe_response_image, to_save)

    # CALL YOUR FUNCTION TO FIND THE CORNER PIXELS
    checkerboard_corners = our_harris_corner_detector(checkerboard, K,
                                                      CHECKERBOARD_THRESHOLD)
    giraffe_corners = our_harris_corner_detector(giraffe, K, GIRAFFE_THRESHOLD)

    # create the output plot.
    create_corner_plots(checkerboard, checkerboard_corners, giraffe,
                        giraffe_corners, to_save)


if __name__ == "__main__":
    main(to_save=True)
