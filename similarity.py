import timeit
from metric import distance
import cv2
import numpy as np
import argparse
import pickle

from pathlib import Path
from pymongo import MongoClient
from dynaconf import settings
from feature.moment import CompareMoment
from feature.sift import CompareSift

import output
from tqdm import tqdm, trange
from multiprocessing import Pool

from sklearn.decomposition import NMF
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import minmax_scale

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
        c = CompareMoment(img_path, settings.WINDOW.WIN_HEIGHT, settings.WINDOW.WIN_WIDTH, settings.MOMENT.WEIGHTS)
    elif model == "sift":
        coll = client.db[settings.SIFT.COLLECTION]
        c = CompareSift(img_path, bool(settings.SIFT.USE_OPENCV))

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
    img_meta = []
    for meta in hand_meta:
        temp = []
        temp.append(meta["age"])
        temp.append(0 if meta["gender"] == "male" else 1)
        if meta["skinColor"] == "fair":
            temp.append(0)
        elif meta["skinColor"] == "dark":
            temp.append(1)
        elif meta["skinColor"] == "medium":
            temp.append(2)
        elif meta["skinColor"] == "very fair":
            temp.append(3)
        else:
            print("GOT {}".format(meta["skinColor"]))
            raise Exception
        temp.append(meta["accessories"])
        temp.append(meta["nailPolish"])
        temp.append(0 if meta["aspectOfHand"] == "dorsal" else 1)
        temp.append(0 if meta["hand"] == "right" else 1)
        temp.append(meta["irregularities"])

        img_meta.append(temp)

    img_meta = np.array(img_meta)

    model = NMF(n_components=20, init='random', random_state=0)
    u = model.fit_transform(img_meta)
    h = model.components_



def task_7(k):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    hand_meta = list(client.db.hands_meta.find())
    subjects = {}
    for meta in hand_meta:
        temp = []
        temp.append(meta["age"])
        temp.append(0 if meta["gender"] == "male" else 1)
        if meta["skinColor"] == "fair":
            temp.append(0)
        elif meta["skinColor"] == "dark":
            temp.append(1)
        elif meta["skinColor"] == "medium":
            temp.append(2)
        elif meta["skinColor"] == "very fair":
            temp.append(3)
        else:
            print("GOT {}".format(meta["skinColor"]))
            raise Exception

        subjects[meta["id"]] = temp

    subs = np.array([subjects[v] for v in subjects])

    # Generate subject, subject similarity
    # sub_sub = np.matmul(subs, subs.T)
    sub_sub = []
    for sub1 in subs:
        temp = []
        for sub2 in subs:
            # Euclidean
            d = np.sqrt(np.sum(np.power(sub1 - sub2, 2)))
            if d != 0:
                d = 1 / d

            # Manhattan
            # d = np.sum(sub1 - sub2)
            # if d != 0:
            #     d = 1 / d

            # Pearsons Corelation, rev
            # d = np.corrcoef(sub1, sub2)[0,1]

            # Cosine Similarity, rev
            # d = np.dot(sub1, sub2)/(np.linalg.norm(sub1) * np.linalg.norm(sub2))

            # Intersection similarity, rev
            # ma = sum([max(sub1[j], sub2[j]) for j in range(0, sub1.shape[0])])
            # mi = sum([min(sub1[j], sub2[j]) for j in range(0, sub1.shape[0])])
            # d = mi/float(ma)

            temp.append(d)
        sub_sub.append(temp)

    sub_sub = np.array(sub_sub)

    model = NMF(n_components=20, init='random', random_state=0)
    u = model.fit_transform(sub_sub)

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
    ranks = calc_svd_sim(img_path, args.k_nearest)
    # ranks = task_8(args.k_nearest)
    # ranks = task_7(args.k_nearest)
    e = timeit.default_timer()
    print("Took {} to calculate".format(e - s))

    output.write_to_file("op_temp.html",
                         "{}-{}.html".format(img_path.resolve().name, args.model),
                         ranks=ranks,
                         key=str(img_path),
                         title="TEST")
