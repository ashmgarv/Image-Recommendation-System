import numpy as np
from pymongo import MongoClient
from multiprocessing import Pool
import argparse
from dynaconf import settings
from pprint import pprint
import sys

sys.path.append('../')
from feature_reduction.utils import get_all_vectors
from feature_reduction.feature_reduction import reducer
from feature_reduction.utils import get_term_weight_pairs


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
    pprint(get_term_weight_pairs(latent_vs_old), indent=4)
