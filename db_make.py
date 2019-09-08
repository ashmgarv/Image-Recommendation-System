import moment
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool

from pymongo import MongoClient
from pathlib import Path

def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--window', type=str, required=True)
    parser.add_argument('-p', '--data-path', type=str, required=True)
    return parser

def build_db(data_path, win_h, win_w, batch=1000):
    client = MongoClient('mongodb://localhost:27017/')
    coll = client.db.img_features

    imgs = []

    for img_path in tqdm(data_path.iterdir()):
        img = cv2.imread(str(img_path))
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y, u, v = moment.img_moment(img_yuv, win_h, win_w)
        imgs.append({
            "path": str(img_path),
            "y_moments": y,
            "u_moments": u,
            "v_moments": v
        })

        if len(imgs) % batch == 0:
            coll.insert_many(imgs)
            imgs.clear()

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    win_h = 100
    win_w = 100
    try:
        win_h, win_w = [int(x) for x in args.window.split(",")]
    except:
        raise Exception("Invalid argument to --window (-w). Must be of the format --window=<height>,<width>.")

    path = Path(args.data_path)
    if not path.exists() or not path.is_dir():
        raise Exception("Invalid path provided.")

    build_db(path, win_h, win_w)
