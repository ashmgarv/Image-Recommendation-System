from skimage import feature
import numpy as np
import cv2


def process_img(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    lbp_feature_vector = []

    # Turn the image into 100 * 100 smaller np arrays
    hch = turn_into_100c100(img, 100, 100)

    # Perform lbp using 1 as the radius and 8 as the number of points to be considered for each sub_array
    # Returns a histogram of features for each sub_array
    radius = 1
    num_points = 8
    for row in hch:
        lbp_window = feature.local_binary_pattern(row, num_points, radius)
        histogram = np.histogram(lbp_window.flatten(), bins=2 ** num_points)
        lbp_feature_vector.extend(histogram[0])

    return {
        'path' : str(img_path),
        'lbp' : lbp_feature_vector
    }


def turn_into_100c100(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))


