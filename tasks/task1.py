import numpy as np
from pymongo import MongoClient
from multiprocessing import Pool
import argparse
from dynaconf import settings
from pprint import pprint
import sys

sys.path.append('../')
import output
from utils import get_all_vectors, get_term_weight_pairs
from feature_reduction.feature_reduction import reducer


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-k', '--k_latent_semantics', type=int, required=True)
    parser.add_argument(
        '-frt', '--feature_reduction_technique', type=str, required=True)
    return parser


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()

    images, vectors = get_all_vectors(args.model)

    # reducer automatically maps feature_reduction_technique to the right function
    vectors, eigen_values, latent_vs_old = reducer(
        vectors, args.k_latent_semantics, args.feature_reduction_technique)

    pprint(get_term_weight_pairs(vectors), indent=4)
    pprint(get_term_weight_pairs(latent_vs_old), indent=4)

    # Extra Credit
    # image path with a vector in the latent semantic space
    data_z = zip(images, vectors)
    # image path for each latenet semantic in h
    feature_z = [(idx, images[np.argmax(np.dot(vectors, i[:args.k_latent_semantics]))]) for idx, i in enumerate(latent_vs_old)]

    output.write_to_file("visualize_data_z.html",
                         "data-z-{}-{}.html".format(args.model, args.feature_reduction_technique),
                         data_z=data_z,
                         title="TEST")

    output.write_to_file("visualize_feat_z.html",
                         "feat-z-{}-{}.html".format(args.model, args.feature_reduction_technique),
                         feature_z=feature_z,
                         title="TEST")

