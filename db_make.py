import argparse
import pickle
import pandas as pd
import sys

from bson.binary import Binary
from dynaconf import settings
from multiprocessing import Pool
from pathlib import Path
from pymongo import MongoClient
from tqdm import tqdm, trange

from feature import moment, sift, lbp, hog


def prepare_parser():
    parser = argparse.ArgumentParser()
    # If not provided, we rebuild the metadata instead
    parser.add_argument('-m', '--model', type=str, default=None)
    parser.add_argument('-u',
                        '--build_unlabeled',
                        default=False,
                        action='store_true')
    parser.add_argument('-master',
                        '--build_master',
                        default=False,
                        action='store_true')
    return parser


def process_moment_img(img_path):
    res = moment.process_img(img_path.resolve(), settings.WINDOW.WIN_HEIGHT,
                             settings.WINDOW.WIN_WIDTH)
    res["moments"] = Binary(pickle.dumps(res["moments"], protocol=2))
    return res


def process_moment_inv_img(img_path):
    res = moment.process_img(img_path.resolve(), settings.WINDOW.WIN_HEIGHT,
                             settings.WINDOW.WIN_WIDTH, True)
    res["moments"] = Binary(pickle.dumps(res["moments"], protocol=2))
    return res


def process_sift_img(img_path):
    res = sift.process_img(img_path.resolve())
    res['sift'] = Binary(pickle.dumps(res['sift'], protocol=2))
    return res


def process_lbp_img(img_path):
    res = lbp.process_img(img_path.resolve())
    res['lbp'] = Binary(pickle.dumps(res['lbp'], protocol=2))
    return res


def process_hog_img(img_path):
    res = hog.process_img(img_path.resolve())
    res['hog'] = Binary(pickle.dumps(res['hog'], protocol=2))
    return res


model_to_collection = {
    'moment': settings.MOMENT.COLLECTION,
    'moment_inv': settings.MOMENT.COLLECTION_INV,
    'sift': settings.SIFT.COLLECTION,
    'hog': settings.HOG.COLLECTION,
    'lbp': settings.LBP.COLLECTION
}

model_to_fun = {
    'moment': process_moment_img,
    'moment_inv': process_moment_inv_img,
    'sift': process_sift_img,
    'hog': process_hog_img,
    'lbp': process_lbp_img
}


def build_metadata_db(path, db_name, metadata_path):
    """function that read metadata file and populated mongoDB
    Arguments:
        path {pathlib.Path} -- path of datafolder to append to filenames (easier to filter)
    """
    #rebuilding accessories column and adding path column
    image_metadata = pd.read_csv(str(metadata_path.resolve()))
    image_metadata['accessories'] = image_metadata['accessories'].replace({
        0:
        'without_acs',
        1:
        'with_acs'
    })
    image_metadata['path'] = image_metadata['imageName'].map(
        lambda x: str(path.resolve() / x) if (path / x).is_file() else None)
    image_metadata = image_metadata.dropna(subset=['path'])

    #clear collection and insert
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    db = client[db_name]
    coll = db[settings.IMAGES.METADATA_COLLECTION]
    coll.delete_many({})
    coll.insert_many(image_metadata.to_dict('records'))


def build_db(model, data_path, db_name):
    """
    Extracts features from all the images given in the dataset and stores it in the Database
    Args:
        model: The model to use. This dictates the features to be extracted from the images.
        data_path: Path to build model from.
        db: Database to insert the models into.
    """
    coll_name = model_to_collection[model]

    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    coll = client[db_name][coll_name]
    paths = list(data_path.iterdir())

    imgs = []
    p = Pool(processes=10)
    pbar = tqdm(total=len(paths))

    fun = model_to_fun[model]

    for img in p.imap_unordered(fun, paths):
        imgs.append(img)
        pbar.update()
        if len(imgs) % settings.LOADER.BATCH_SIZE == 0:
            coll.insert_many(imgs)
            imgs.clear()

    p.close()
    p.join()
    pbar.close()
    sys.stdout.flush()

    if len(imgs) > 0:
        coll.insert_many(imgs)

    #if model is sift, generate and insert histogram vector
    if model == 'sift':
        sift.generate_histogram_vectors(coll)


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    #edge case
    if args.build_unlabeled and args.build_master: raise Exception("oi you cheeky wanker.")

    # Setting images folder
    data_path = Path(settings.path_for(settings.DATA_PATH))
    if args.build_unlabeled: data_path = Path(settings.path_for(settings.UNLABELED_DATA_PATH))
    elif args.build_master: data_path = Path(settings.path_for(settings.MASTER_DATA_PATH))
    
    # Setting database
    database = settings.QUERY_DATABASE if args.build_unlabeled else (settings.MASTER_DATABASE if args.build_master else settings.DATABASE) 
    
    # Setting metadata CSV
    metadata_path = Path(settings.path_for(settings.METADATA_CSV))
    if args.build_unlabeled: metadata_path = Path(settings.UNLABELED_METADATA_CSV)
    elif args.build_master: metadata_path = Path(settings.MASTER_METADATA_CSV)


    if (not data_path.exists() or not data_path.is_dir()):
        raise Exception("Invalid path provided.")

    if not args.model:
        print("inserting metadata for {} into {}".format(data_path, database))
        build_metadata_db(data_path, database, metadata_path)
    else:
        print("Building database for model {} from {}".format(
            args.model, str(data_path)))
        build_db(args.model, data_path, database)
    print("finished")
