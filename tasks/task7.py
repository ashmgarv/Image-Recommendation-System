import numpy as np
from pymongo import MongoClient
from multiprocessing import Pool
import argparse
from dynaconf import settings
from pprint import pprint
import sys
import os

sys.path.append('../')
import output
from feature_reduction.feature_reduction import reducer
from utils import get_metadata, get_term_weight_pairs
from metric import distance, similarity

from sklearn.preprocessing import MinMaxScaler

mapping = {
    "male": 0,
    "female": 1,

    "very fair": 0,
    "fair": 1,
    "medium": 2,
    "dark": 3,

    "dorsal": 0,
    "palmar": 1,

    "right": 0,
    "left": 1
}


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k_latent_semantics', type=int, required=True)
    return parser


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    meta = get_metadata()

    try:
        subjects = { m['id']: {
            "vec": [
                m['age'],
                mapping[m['gender']],
                mapping[m['skinColor']]
            ],
            "img": m["path"]
        } for m in meta }
    except KeyError:
        raise Exception("Invalid metadata detected")

    sub_img = []
    subs = []
    sub_ids = []
    for idx, s in enumerate(subjects):
        subs.append(subjects[s]["vec"])
        sub_img.append(subjects[s]["img"])
        sub_ids.append(s)
    subs = np.array(subs, dtype=float)

    # Scale the ages to better fit with the other binary values
    m = MinMaxScaler()
    subs[:,0] = m.fit_transform(subs[:,0].reshape(-1,1)).reshape(1,-1)

    # sub_sub = np.array([distance.similarity(subs, s, distance.EUCLIDEAN) for s in subs])
    sub_sub = np.array([similarity.similarity(subs, s, similarity.PEARSONS) for s in subs])

    vectors, eigen_values, latent_vs_old = reducer(
        sub_sub, args.k_latent_semantics, "nmf")

    get_term_weight_pairs(vectors, "sub_weight_{}.csv".format(args.k_latent_semantics))

    resp = [
        sorted([(sub_ids[idx], sub_img[idx], sim) for idx, sim in enumerate(sims)], key=lambda el: el[2], reverse=True)
    for sims in sub_sub]

    output.write_to_file("visualize_sub_sub.html",
                         "sub-sub-{}.html".format(args.k_latent_semantics),
                         sub_img=sub_img,
                         sub_ids=sub_ids,
                         resp=resp,
                         title="TEST")

