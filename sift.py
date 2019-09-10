import utils
import cv2
import numpy as np

from dynaconf import settings

def parse_sift_op(out):
    data = out.split("\n")
    keypoints, var = (int(i) for i in data[0].split(" "))
    data = data[1:]

    keys = []
    tmp = []
    for line in data:
        if len(line.strip()) == 0:
            continue
        if not line.startswith(" "):
            if len(tmp) == var + 4:
                keys.append(tmp)
            elif len(keys) != 0:
                raise Exception("Expected {} sized vector, got {}".format(var + 4, len(tmp)))
            tmp = []
        tmp.extend([float(i) if '.' in i else int(i) for i in line.strip().split(" ")])

    if len(tmp) > 0:
        keys.append(tmp)

    if len(keys) != keypoints:
        raise Exception("Expected {} keypoints, got {}".format(keypoints, len(keys)))

    return keys

def img_sift(img_path):
    # TODO: Resize image?
    # sift binary does not ally images having > 1800 pixels in any dimension.
    # Also we have to maintain aspect ratio.

    img = cv2.imread(str(img_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_data = cv2.imencode(".pgm", img_gray)[1].tostring()

    ret, out, err = utils.talk([settings.SIFT.BIN_PATH], settings._root_path, stdin=img_data, stdout=True)
    if err.startswith("Finding keypoints") and err.endswith("keypoints found.\n"):
        err = None

    if ret != 0 or err != None:
        raise Exception("Error occured: {}".format(err))

    try:
        return np.array(parse_sift_op(out))
    except Exception as e:
        raise Exception("Invalid output from sift binary: {}\n{}".format(e, out))

def process_img(img_path):
    return {
        "path": str(img_path),
        "sift": img_sift(str(img_path))
    }

