import cv2
import numpy as np
import pickle
from pymongo import MongoClient
from dynaconf import settings

from sklearn.preprocessing import MinMaxScaler


def moment_1(window):
    """
    Calculates the first Moment of the given image window.

    Args:
        window: An m x n numpy matrix.

    Returns:
        The mean of the color distribution in the given window.
    """
    return window.flatten().mean()


def moment_2(window):
    """
    Calculates the second Moment of the given image window.

    Args:
        window: An m x n numpy matrix.

    Returns:
        The standard deviation of the color distribution in the given window.
    """
    return window.flatten().std()


def moment_3(window):
    """
    Calculates the third Moment of the given image window.

    Args:
        window: An m x n numpy matrix.

    Returns:
        The skewness of the color distribution in the given window.
    """
    std = window.std()
    if std == 0 or window.size == 0:
        return 0
    return np.power((window - window.mean()) / window.std(), 3).sum() / window.size
    # mean = window.flatten().mean()
    # temp = np.fromfunction(lambda x,y: (window[x,y] - mean) ** 3, window.shape, dtype=int).flatten().sum() / (window.shape[0] * window.shape[1])
    # if temp >= 0:
    #     return temp ** 1/3
    # temp *= -1
    # return (temp ** 1/3) * -1


def img_moment(img, win_h, win_w, invert=False):
    """
    Divides the image into windows of win_h x win_w and calculates the three moments for each of these windows.

    Args:
        img: The image read by cv2 in YUV format.
        win_h: The height of the window to split img.
        win_w: The width of the window to split img.

    Returns:
        Lists containing the Color Moments for each window, one for each channel.
    """
    if invert:
        img = cv2.bitwise_not(img)
    y, u, v = cv2.split(img)
    img_h, img_w, chans = img.shape

    moments = []
    # Ignore the left out pixels? Or perhaps take another window overlapping the
    # last window created.
    for i in range(0, img_h, win_h):
        if i + win_h > img_h:
            break
        for j in range(0, img_w, win_w):
            if j + win_w > img_w:
                break

            mfw = []
            win = y[i:i + win_h, j:j + win_w]
            mfw.extend([moment_1(win), moment_2(win), moment_3(win)])

            win = u[i:i + win_h, j:j + win_w]
            mfw.extend([moment_1(win), moment_2(win), moment_3(win)])

            win = v[i:i + win_h, j:j + win_w]
            mfw.extend([moment_1(win), moment_2(win), moment_3(win)])

            # mfw = []
            # win = y[i:i + win_h, j:j + win_w]
            # mfw.extend([moment_1(win), moment_2(win), moment_3(MinMaxScaler(feature_range=(0, 10)).fit_transform(win))])

            # win = u[i:i + win_h, j:j + win_w]
            # mfw.extend([moment_1(win), moment_2(win), moment_3(MinMaxScaler(feature_range=(0, 10)).fit_transform(win))])

            # win = v[i:i + win_h, j:j + win_w]
            # mfw.extend([moment_1(win), moment_2(win), moment_3(MinMaxScaler(feature_range=(0, 10)).fit_transform(win))])

            moments.append(mfw)

    return np.array(moments)


def process_img(img_path, win_h, win_w, invert=False):
    img = cv2.imread(str(img_path))
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    moments = img_moment(img_yuv, win_h, win_w, invert)
    return {
        "path": str(img_path),
        "moments": moments
    }



def get_all_vectors(coll, f={}):
    all_image_names = []
    all_vectors = []
    for row in coll.find(f).sort([('path',1)]):
        all_image_names.append(row['path'])
        moments = pickle.loads(row['moments']).flatten()
        all_vectors.append(moments)

    return all_image_names, np.array(all_vectors)

