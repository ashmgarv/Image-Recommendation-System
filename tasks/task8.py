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

    try:
        img_meta = np.array([[
            m["age"],
            mapping[m["gender"]],
            mapping[m["skinColor"]],
            mapping[m["accessories"]],
            m["nailPolish"],
            mapping[m["aspectOfHand"].split()[0]],
            mapping[m["aspectOfHand"].split()[1]],
            m["irregularities"]] for m in meta])
    except KeyError:
        raise Exception("Invalid metadata detected")

    vectors, eigen_values, latent_vs_old = reducer(
        img_meta, args.k_latent_semantics, "nmf")

    get_term_weight_pairs(vectors, "8_img_{}.csv".format(args.k_latent_semantics))
    get_term_weight_pairs(latent_vs_old, "8_feat_{}.csv".format(args.k_latent_semantics))

