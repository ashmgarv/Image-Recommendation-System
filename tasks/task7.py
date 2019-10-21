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
from utils import get_metadata, get_term_weight_pairs, get_all_vectors
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
    # Mapping between image file path name and the metadata
    meta = {m['path']: m for m in meta}
    images, data_matrix = get_all_vectors("moment_inv")

    data_matrix = np.c_[data_matrix, np.array([[meta[i]['age'], mapping[meta[i]['gender']], mapping[meta[i]['skinColor']]] for i in images]) * [2,2,10]]

    # Image-Image similarity
    img_img = np.array([
        distance.similarity(data_matrix, img, distance.EUCLIDEAN) for img in data_matrix
    ])

    # sub id to list of their image index in images
    subs = {}
    # sub id to metadata if the subjects
    sub_meta = {}
    for img in meta:
        idx = images.index(img)
        if meta[img]['id'] not in subs:
            subs[meta[img]['id']] = []
        subs[meta[img]['id']].append(idx)
        sub_meta[meta[img]['id']] = meta[img]

    # sub id to order in matrix
    sub_to_idx = {sub: idx for idx, sub in enumerate(subs)}
    # index to sub id
    idx_to_sub = [0] * len(sub_to_idx)
    for sub in sub_to_idx:
        idx_to_sub[sub_to_idx[sub]] = sub
    # A subject subject similarity index
    sub_sub = np.zeros((len(subs), len(subs),))

    for sub1 in sub_to_idx:
        for sub2 in sub_to_idx:
            sub_sub[sub_to_idx[sub1], sub_to_idx[sub2]] = img_img[subs[sub1],:].take(subs[sub2], axis=1).mean()

    w, _, h = reducer(sub_sub, args.k_latent_semantics, "nmf")

    # Print term weigth pairs
    get_term_weight_pairs(w, "task7_{}.csv".format(args.k_latent_semantics))
    sub_weight = [
        sorted([("z{}".format(idx), weight,) for idx, weight in enumerate(row)], key=lambda x: x[1])
        for row in w
    ]

    output.write_to_file("visualize_task7.html",
                         "task7-{}.html".format(args.k_latent_semantics),
                         vectors=sub_weight,
                         subs=subs,
                         idx_to_sub=idx_to_sub,
                         images=images,
                         sub_meta=sub_meta,
                         title="TEST")

