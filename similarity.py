import timeit
import moment
import cv2
import numpy as np
import argparse

from pathlib import Path
from pymongo import MongoClient
from dynaconf import settings

import output

def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-k', '--k-nearest', type=int, required=True)
    parser.add_argument('-i', '--image', type=str, required=True)
    return parser

def calc_mom_sim(img_path, k):
    client = MongoClient(host=settings.HOST, port=settings.PORT, username=settings.USERNAME, password=settings.PASSWORD)
    coll = client.db[settings.MOMENT.COLLECTION]

    key_feats = moment.process_img(img_path, settings.WINDOW.WIN_HEIGHT, settings.WINDOW.WIN_WIDTH)
    k_y = np.array(key_feats["y_moments"], dtype=list)
    k_u = np.array(key_feats["u_moments"], dtype=list)
    k_v = np.array(key_feats["v_moments"], dtype=list)

    times = []

    def image_index(rec):
        s = timeit.default_timer()

        y = np.array(rec["y_moments"], dtype=list)
        u = np.array(rec["u_moments"], dtype=list)
        v = np.array(rec["v_moments"], dtype=list)

        tmp = np.absolute(k_y - y)
        tmp[:,0] *= settings.MOMENT.W_Y_1
        tmp[:,1] *= settings.MOMENT.W_Y_2
        tmp[:,2] *= settings.MOMENT.W_Y_3
        d_y = np.sum(tmp, axis=1)

        tmp = np.absolute(k_u - u)
        tmp[:,0] *= settings.MOMENT.W_U_1
        tmp[:,1] *= settings.MOMENT.W_U_2
        tmp[:,2] *= settings.MOMENT.W_U_3
        d_u = np.sum(tmp, axis=1)

        tmp = np.absolute(k_v - v)
        tmp[:,0] *= settings.MOMENT.W_V_1
        tmp[:,1] *= settings.MOMENT.W_V_2
        tmp[:,2] *= settings.MOMENT.W_V_3
        d_v = np.sum(tmp, axis=1)

        d = d_y + d_u + d_v
        res = (rec["path"], d.mean(),)

        e = timeit.default_timer()
        times.append(e-s)
        return res

    data = coll.find()
    d = np.array([image_index(data.next()) for _ in range(0, coll.count_documents({}))], dtype=[('x', object), ('y', float)])
    d.sort(order="y")

    times = np.array(times)
    print("Took AVG {} for each comparision".format(times.mean()))
    print("Took Total {} for all comparisions".format(times.sum()))

    return d[0:k]

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists() or not img_path.is_file():
        raise Exception("Invalid image path.")

    if args.k_nearest == 0:
        raise Exception("Need k > 0")

    s = timeit.default_timer()
    ranks = calc_mom_sim(str(img_path), args.k_nearest)
    e = timeit.default_timer()
    print("Took {} to calculate".format(e - s))

    output.write_to_file("op_temp.html", "{}.html".format(img_path.resolve().name), ranks=ranks, key=str(img_path), title="TEST")

