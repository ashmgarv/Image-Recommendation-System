import time
import moment
import cv2
import numpy as np
import argparse
from tqdm import tqdm, trange
from multiprocessing import Pool

from pymongo import MongoClient
from pathlib import Path

WIN_H = 100
WIN_W = 100

def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--window', type=str, required=True)
    parser.add_argument('-p', '--data-path', type=str, required=True)
    parser.add_argument('-b', '--batch-size', type=int, required=True)
    parser.add_argument('-c', '--collection', type=str, required=True)
    return parser

def process_img(img_path):
    img = cv2.imread(str(img_path))
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y, u, v = moment.img_moment(img_yuv, WIN_H, WIN_W)
    return {
        "path": str(img_path),
        "y_moments": y,
        "u_moments": u,
        "v_moments": v
    }

def build_db(data_path, coll_name, batch=1000):
    client = MongoClient('mongodb://localhost:27017/')
    coll = client.db[coll_name]
    paths = list(data_path.iterdir())

    imgs = []
    p = Pool(processes=10)
    pbar = tqdm(total=len(paths))
    for img in p.imap_unordered(process_img, paths):
        imgs.append(img)
        pbar.update()
        if len(imgs) % batch == 0:
            coll.insert_many(imgs)
            imgs.clear()

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    try:
        WIN_H, WIN_H = [int(x) for x in args.window.split(",")]
    except:
        raise Exception("Invalid argument to --window (-w). Must be of the format --window=<height>,<width>.")

    path = Path(args.data_path)
    if not path.exists() or not path.is_dir():
        raise Exception("Invalid path provided.")

    batch = args.batch_size
    coll_name = args.collection

    build_db(path, coll_name, batch)
