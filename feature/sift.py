import timeit
import utils
import cv2
import numpy as np

from dynaconf import settings
from pymongo import MongoClient

import copyreg
import pickle

# Taken from https://stackoverflow.com/a/48832618
# Adds support to pickle cv2.KeyPoint objects
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

def parse_sift_op(out):
    data = out.split("\n")
    keypoints, var = (int(i) for i in data[0].split(" "))
    data = data[1:]

    tmp = []
    keys = []
    desc = []
    for line in data:
        if len(line.strip()) == 0:
            continue
        if not line.startswith(" "):
            if len(tmp) == var + 4:
                keys.append(cv2.KeyPoint(tmp[1], tmp[0], tmp[2], tmp[3]))
                desc.append(tmp[4:])
            elif len(keys) != 0:
                raise Exception("Expected {} sized vector, got {}".format(var + 4, len(tmp)))
            tmp = []
        tmp.extend([float(i) if '.' in i else int(i) for i in line.strip().split(" ")])

    if len(tmp) > 0:
        keys.append(cv2.KeyPoint(tmp[1], tmp[0], tmp[2], tmp[3]))
        # keys.append((tmp[1], tmp[0], tmp[2], tmp[3],))
        desc.append(tmp[4:])

    if len(keys) != keypoints:
        raise Exception("Expected {} keypoints, got {}".format(keypoints, len(keys)))
    return [keys, np.array(desc)]

def img_sift(img_path, sift_opencv):
    # TODO: Resize image?
    # sift binary does not ally images having > 1800 pixels in any dimension.
    # Also we have to maintain aspect ratio.

    # img = cv2.imread(str(img_path))
    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if sift_opencv:
        # sift = cv2.xfeatures2d.SIFT_create()
        # res = sift.detectAndCompute(img_gray, None)
        res = sift_opencv.detectAndCompute(img_gray, None)
        # cv2.KeyPoint cannot be pickled. So multiprocessing with this is not
        # possible
        # res = [[(i.pt[0], i.pt[1], i.size, i.angle,) for i in res[0]], res[1]]
        res = [res[0], res[1]]
        return res

    img_data = cv2.imencode(".pgm", img_gray)[1].tostring()

    ret, out, err = utils.talk([settings.SIFT.BIN_PATH], settings._root_path, stdin=img_data, stdout=True)
    if err.startswith("Finding keypoints") and err.endswith("keypoints found.\n"):
        err = None

    if ret != 0 or err != None:
        raise Exception("Error occured: {}".format(err))

    try:
        return parse_sift_op(out)
    except Exception as e:
        raise Exception("Invalid output from sift binary: {}\n{}".format(e, out))

def process_img(img_path, use_opencv):
    return {
            "path": str(img_path),
            "sift": img_sift(str(img_path), cv2.xfeatures2d.SIFT_create() if use_opencv else None)
            }

def find_nearest_kps(kps, kp):
    try:
        best_two = np.sort(np.sum(np.power(kps - kp, 2), axis=1))[:2]
    except:
        import pdb
        pdb.set_trace()
        pass
    # return best_two[1]/best_two[0] >= 1.5
    return 10 * 10 * best_two[0] < 6 * 6 * best_two[1]

# Compare img2 to img1
# Returns the score of match for img1
def compare_one(img1, img2):
    # For each keypoint in img2, we look for the top two closest keypoints in
    # img1. Then using a threshold, we either accept a match, or reject the
    # match.
    res = [1 for i in range(0, len(img2['sift'][1])) if find_nearest_kps(img1['sift'][1], img2['sift'][1][i]) == True]
    return (img1['path'], sum(res))

def compare_many(imgs, img):
    times = []

    def compare_one(img1, img2):
        img1['sift'] = pickle.loads(img1['sift'])
        s = timeit.default_timer()
        res = [1 for i in range(0, len(img2['sift'][1])) if find_nearest_kps(img1['sift'][1], img2['sift'][1][i]) == True]
        e = timeit.default_timer()
        times.append(e-s)
        return (img1['path'], sum(res))

    res = np.array([compare_one(i, img) for i in imgs], dtype=[('x', object), ('y', float)])

    times = np.array(times)
    print("Took AVG {} for each comparision".format(times.mean()))
    print("Took Total {} for all comparisions".format(times.sum()))

    return res

def similarity(img_path, k):
    img_data = process_img(str(img_path), True)

    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    coll = client.db[settings.SIFT.COLLECTION]

    data = list(coll.find())
    d = np.array(
        compare_many(data, img_data),
        dtype=[('x', object), ('y', float)])
    d.sort(order="y")

    return d[-1 * k:]

def visualize_sift(img_path, op_path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray, kp[0], img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(str(op_path / '{}_keypoints.jpg'.format(img_path.resolve().name)), img)
