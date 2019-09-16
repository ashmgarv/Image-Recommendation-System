import timeit
import cv2
import numpy as np
import argparse
import pickle

from pathlib import Path
from pymongo import MongoClient
from dynaconf import settings
from feature.moment import CompareMoment, Moment
from feature.sift import CompareSift, Sift

import output
from tqdm import tqdm, trange
from multiprocessing import Pool


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-k', '--k-nearest', type=int, required=True)
    parser.add_argument('-i', '--image', type=str, required=True)
    return parser


def calc_sim(img_path, k, model):
    """
    Find the top k similar images from the database to the provided input image.

    Args:
        img_path: Path of the input image.
        k: The number of matches to be selected.
        model: The model to be used for matching.
    """
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)

    if model == "moment":
        coll = client.db[settings.MOMENT.COLLECTION]
        m = Moment(settings.WINDOW.WIN_HEIGHT, settings.WINDOW.WIN_WIDTH)
        c = CompareMoment(m.process_img(img_path) , [
                              settings.MOMENT.W_Y_1, settings.MOMENT.W_Y_2,
                              settings.MOMENT.W_Y_3
                          ], [
                              settings.MOMENT.W_U_1, settings.MOMENT.W_U_2,
                              settings.MOMENT.W_U_3
                          ], [
                              settings.MOMENT.W_V_1, settings.MOMENT.W_V_2,
                              settings.MOMENT.W_V_3
                          ])
    elif model == "sift":
        coll = client.db[settings.SIFT.COLLECTION]
        s = Sift(bool(settings.SIFT.USE_OPENCV))
        c = CompareSift(s.process_img(img_path))

    res = []
    p = Pool(processes=10)
    pbar = tqdm(total=coll.count_documents({}))

    for r in p.imap_unordered(c.compare_one, coll.find(), chunksize=100):
        res.append(r)
        pbar.update()

    res = np.array(res, dtype=[('x', object), ('y', float)])
    res.sort(order="y")

    if model == "moment":
        return res[0:k]
    elif model == "sift":
        return np.flip(res[-1 * k:])


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists() or not img_path.is_file():
        raise Exception("Invalid image path.")

    if args.k_nearest == 0:
        raise Exception("Need k > 0")

    s = timeit.default_timer()
    ranks = calc_sim(img_path, args.k_nearest, args.model)
    e = timeit.default_timer()
    print("Took {} to calculate".format(e - s))

    output.write_to_file("op_temp.html",
                         "{}-{}.html".format(img_path.resolve().name, args.model),
                         ranks=ranks,
                         key=str(img_path),
                         title="TEST")
