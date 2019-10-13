import time
import cv2
import numpy as np
import argparse
from tqdm import tqdm, trange
from multiprocessing import Pool
from feature import moment, sift, lbp

from pymongo import MongoClient
from bson.binary import Binary
import pickle
from pathlib import Path
from dynaconf import settings
import pandas as pd

def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-d', '--data-path', type=str)
    parser.add_argument('-c', '--collection', type=str)
    return parser

def process_moment_img(img_path):
    res = moment.process_img(img_path.resolve(), settings.WINDOW.WIN_HEIGHT,
                             settings.WINDOW.WIN_WIDTH)
    res["moments"] = Binary(pickle.dumps(res["moments"], protocol=2))
    return res


def process_sift_img(img_path):
    res = sift.process_img(img_path.resolve(), bool(settings.SIFT.USE_OPENCV))
    res['sift'] = Binary(pickle.dumps(res['sift'], protocol=2))
    return res

def process_lbp_img(img_path):
    res = lbp.process_img(img_path.resolve())
    res['lbp'] = Binary(pickle.dumps(res['sift'], protocol=2))
    return res

def build_metadata_db(path):
    """function that read metadata file and populated mongoDB
    
    Arguments:
        path {str} -- path of datafolder to append to filenames (easier to filter)
    """
    #rebuilding accessories column and adding path column
    image_metadata = pd.read_csv(settings.IMAGES.METADATA_CSV)
    image_metadata['accessories'] = image_metadata['accessories'].replace( {0: 'without_acs',1: 'with_acs'})
    image_metadata['path'] = path + image_metadata['imageName'].astype(str)
    del image_metadata['imageName']

    #clear collection and insert
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    db = client[settings.DATABASE]
    coll = db[settings.IMAGES.METADATA_COLLECTION]
    coll.delete_many({})
    coll.insert_many(image_metadata.to_dict('records'))

def build_db(model, data_path, coll_name):
    """
    Extracts features from all the images given in the dataset and stores it in the Database

    Args:
        model: The model to use. This dictates the features to be extracted from the images.
        data_path: Path of the dataset.
        coll_name: Collection name in which to store data.
    """
    if data_path is None:
        data_path = Path(settings.DATA_PATH)

    if coll_name is None:
        if model == "moment":
            coll_name = settings.MOMENT.COLLECTION
        elif model == "sift":
            coll_name = settings.SIFT.COLLECTION
        else:
            return

    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    coll = client.db[coll_name]
    paths = list(data_path.iterdir())

    imgs = []
    p = Pool(processes=10)
    pbar = tqdm(total=len(paths))

    if model == "moment":
        fun = process_moment_img
    elif model == "sift":
        fun = process_sift_img
    elif model == "lbp":
        fun = process_lbp_img
    else:
        return

    for img in p.imap_unordered(fun, paths):
        imgs.append(img)
        pbar.update()
        if len(imgs) % settings.LOADER.BATCH_SIZE == 0:
            coll.insert_many(imgs)
            imgs.clear()

    if len(imgs) > 0:
        coll.insert_many(imgs)


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    if args.data_path:
        path = Path(args.data_path)
        if (not path.exists() or not path.is_dir()):
            raise Exception("Invalid path provided.")
    coll_name = args.collection

    print("inserting metadata")
    build_metadata_db(args.data_path)
    print("finished")

    build_db(args.model, None if not args.data_path else path,
             None if not args.collection else coll_name)
