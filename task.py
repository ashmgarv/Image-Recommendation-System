import timeit
from metric import distance
import cv2
import numpy as np
import argparse
import pickle

from pathlib import Path
from pymongo import MongoClient
from dynaconf import settings
import similarity

import output
from tqdm import tqdm, trange
from multiprocessing import Pool

from sklearn.decomposition import NMF
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import minmax_scale

mapping = {
    "male": 0,
    "female": 1,

    "fair": 0,
    "very fair": 1,
    "medium": 2,
    "dark": 3,

    "dorsal": 0,
    "palmar": 1,

    "right": 0,
    "left": 1
}

def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-k', '--k-nearest', type=int, required=True)
    parser.add_argument('-i', '--image', type=str, required=True)
    return parser


def calc_sim(img_path, k, model_name):
    """
    Find the top k similar images from the database to the provided input image.

    Args:
        img_path: Path of the input image.
        k: The number of matches to be selected.
        model: The model to be used for matching.
    """
    if model_name == "moment":
        opts = {
            'win_h': settings.WINDOW.WIN_HEIGHT,
            'win_w': settings.WINDOW.WIN_WIDTH,
            'weights': settings.MOMENT.WEIGHTS
        }
        coll_name = settings.MOMENT.COLLECTION
        model = similarity.MOMENT
    elif model_name == "sift":
        opts = {
            'use_opencv': settings.SIFT.USE_OPENCV
        }
        coll_name = settings.SIFT.COLLETION
        model = similarity.SIFT
    else:
        raise Exception("Invalid model selected")

    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)

    coll = client.db[coll_name]
    c = similarity.comparision(model, img_path=img_path, **opts)
    res = c.compare_many(coll.find(), coll.count_documents({}))
    return res[:k]

def task_1(img_path, model, k):
    client = MongoClient('mongodb://localhost:27017/')
    coll = client.db["img_moment_inv"]
    data = list(coll.find())

    # Metadata
    # hand_meta = list(client.db.hands_meta.find({"gender": "male"}))
    hand_meta = list(client.db.hands_meta.find())
    hand_meta = {i['imageName']: {
        "id": i["id"],
        "aspectOfHand": i["aspectOfHand"],
        "age": i["age"],
        "gender": i["gender"],
        "skinColor": i["skinColor"],
        "accessories": i["accessories"],
        "nailPolish": i["nailPolish"],
        "irregularities": i["irregularities"]
        } for i in hand_meta}

    # filter
    data = [d for d in data if d['path'].split("/")[-1] in hand_meta]
    meta = {i['path']: idx for idx,i in enumerate(data)}
    meta_rev = {idx: i['path'] for idx, i in enumerate(data)}

    data = [pickle.loads(i['moments']) for i in data]
    data = [i.flatten() for i in data]
    data = np.array(data)

    # NMF
    inc = np.amin(data.flatten())
    if inc < 0:
        data += (-1 * inc)

    nmf = NMF(n_components=20, init='random', random_state=0)
    u = nmf.fit_transform(data)
    h = nmf.components_

    # image path with a vector in the latent semantic space
    data_z = zip(meta, u[:,:20])
    # data_z = list(data_z)[:10]
    # image path for each latenet semantic in h
    feature_z = [(idx, meta_rev[np.argmax(np.dot(u[:,:20], i[:20]))]) for idx, i in enumerate(h)]

    output.write_to_file("visualize_data_z.html",
                         "data-z-{}-{}-nmf.html".format(img_path.resolve().name, model),
                         data_z=data_z,
                         title="TEST")

    output.write_to_file("visualize_feat_z.html",
                         "feat-z-{}-{}-nmf.html".format(img_path.resolve().name, model),
                         feature_z=feature_z,
                         title="TEST")


def calc_svd_sim(img_path, k):
    client = MongoClient('mongodb://localhost:27017/')
    coll = client.db["img_moment_inv"]
    data = list(coll.find())

    # Metadata
    # hand_meta = list(client.db.hands_meta.find({"gender": "male"}))
    hand_meta = list(client.db.hands_meta.find())
    hand_meta = {i['imageName']: {
        "id": i["id"],
        "aspectOfHand": i["aspectOfHand"],
        "age": i["age"],
        "gender": i["gender"],
        "skinColor": i["skinColor"],
        "accessories": i["accessories"],
        "nailPolish": i["nailPolish"],
        "irregularities": i["irregularities"]
        } for i in hand_meta}

    # filter
    data = [d for d in data if d['path'].split("/")[-1] in hand_meta]
    meta = {i['path']: idx for idx,i in enumerate(data)}

    data = [pickle.loads(i['moments']) for i in data]
    data = [i.flatten() for i in data]
    data = np.array(data)

    # SVD
    # u, s, vh = np.linalg.svd(data, full_matrices=False)
    # dims = 200

    # NMF
    # Results actually change even with a linear transformation

    # Increment all by min. Best so far.
    # import pdb
    # pdb.set_trace()
    inc = np.amin(data.flatten())
    if inc < 0:
        data += (-1 * inc)

    # Incrementing only the skew columns. No good.
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         if j % 3 == 2:
    #             data[i,j] += 100

    # Just scale those columns
    # for j in range(data.shape[1]):
    #     if j % 3 == 2:
    #         data[:,j] = minmax_scale(data[:,j], feature_range=(0,100))

    # Removing all the skew. Doesnt give great results.
    # data = data[:,[i for i in range(data.shape[1]) if i % 3 != 2]]

    # These dont do much good
    # data = MinMaxScaler(feature_range=(0, 10)).fit_transform(data)
    # data = RobustScaler().fit_transform(data)

    model = NMF(n_components=20, init='random', random_state=0)
    u = model.fit_transform(data)
    h = model.components_
    dims = u.shape[1]

    img_desc = u[meta[str(img_path)]]

    # Euclidean
    # d = np.sqrt(np.sum(np.power(u[:,:dims] - img_desc[:dims], 2), axis=1))
    d = distance.distance(u[:,:dims], img_desc[:dims], distance.EUCLIDEAN)

    # Manhattan
    # d = np.sum(u[:,:dims] - img_desc[:dims], axis=1)

    # Pearsons Corelation, rev
    # d = []
    # for i in range(0, u.shape[0]):
    #     d.append(np.corrcoef(u[i,:dims], img_desc[:dims])[0,1])

    # Cosine Similarity, rev
    # d = []
    # for i in range(0, u.shape[0]):
    #     d.append(np.dot(u[i,:dims], img_desc[:dims])/(np.linalg.norm(u[i,:dims])*np.linalg.norm(img_desc[:dims])))

    # Intersection similarity, rev
    # Looks like it needs positive values
    # Somehow, SVD may generate negative values. So this wont work.
    # d = []
    # for i in range(0, u.shape[0]):
    #     ma = sum([max(u[i,j], img_desc[j]) for j in range(0, dims)])
    #     mi = sum([min(u[i,j], img_desc[j]) for j in range(0, dims)])
    #     d.append(mi/float(ma))

    ranks = [(path, d[meta[path]]) for path in meta]
    ranks = np.array(ranks, dtype=[('x', object), ('y', float)])
    ranks.sort(order="y")

    return ranks[:k]
    # return np.flip(ranks[-1 * k:])

def task_8(k):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    hand_meta = list(client.db.hands_meta.find())
    try:
        img_meta = np.array([[
            meta["age"],
            mapping[meta["gender"]],
            mapping[meta["skinColor"]],
            meta["accessories"],
            meta["nailPolish"],
            mapping[meta["aspectOfHand"]],
            mapping[meta["hand"]],
            meta["irregularities"]] for meta in hand_meta])
    except KeyError:
        raise Exception("Invalid metadata detected")
        return

    model = NMF(n_components=k, init='random', random_state=0)
    u = model.fit_transform(img_meta)
    h = model.components_


def task_7(k):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    hand_meta = list(client.db.hands_meta.find())

    try:
        subjects = { meta['id']: [
            meta['age'],
            mapping[meta['gender']],
            mapping[meta['skinColor']]
        ] for meta in hand_meta }
    except KeyError:
        raise Exception("Invalid metadata detected")
        return

    subs = np.array([subjects[v] for v in subjects])

    # Generate subject, subject similarity
    sub_sub = np.array([distance.similarity(subs, s, distance.EUCLIDEAN) for s in subs])

    model = NMF(n_components=20, init='random', random_state=0)
    u = model.fit_transform(sub_sub)
    h = model.components_

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists() or not img_path.is_file():
        raise Exception("Invalid image path.")

    if args.k_nearest == 0:
        raise Exception("Need k > 0")

    s = timeit.default_timer()
    # ranks = calc_sim(img_path, args.k_nearest, args.model)
    # ranks = calc_svd_sim(img_path, args.k_nearest)
    # ranks = task_8(args.k_nearest)
    # ranks = task_7(args.k_nearest)
    task_1(img_path, args.model, args.k_nearest)
    e = timeit.default_timer()
    print("Took {} to calculate".format(e - s))

    # output.write_to_file("op_temp.html",
    #                      "{}-{}.html".format(img_path.resolve().name, args.model),
    #                      ranks=ranks,
    #                      key=str(img_path),
    #                      title="TEST")
