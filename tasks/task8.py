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


mapping = {
    "without_acs": 0,
    "with_acs": 1,

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
    images = []
    img_meta = []

    try:
        for m in meta:
            images.append(m['path'])
            img_meta.append([
                m["age"],
                mapping[m["gender"]],
                mapping[m["skinColor"]],
                mapping[m["accessories"]],
                m["nailPolish"],
                mapping[m["aspectOfHand"].split()[0]],
                mapping[m["aspectOfHand"].split()[1]],
                m["irregularities"]])
    except KeyError:
        raise Exception("Invalid metadata detected")

    vectors, eigen_values, latent_vs_old = reducer(
        img_meta, args.k_latent_semantics, "nmf")

    get_term_weight_pairs(vectors, "task8_{}.csv".format(args.k_latent_semantics))
    get_term_weight_pairs(latent_vs_old, "task8_{}.csv".format(args.k_latent_semantics))

    # Extra Credit
    # image path with a vector in the latent semantic space
    data_z = zip(images, vectors)
    # image path for each latenet semantic in h
    feature_z = [(idx, images[np.argmax(np.dot(img_meta, i))]) for idx, i in enumerate(latent_vs_old)]

    output.write_to_file("visualize_data_z.html",
                         "task8-data-z-{}.html".format(args.k_latent_semantics),
                         data_z=data_z,
                         title="TEST")

    output.write_to_file("visualize_feat_z.html",
                         "task8-feat-z-{}.html".format(args.k_latent_semantics),
                         feature_z=feature_z,
                         title="TEST")

