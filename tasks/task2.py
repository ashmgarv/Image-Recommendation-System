import numpy as np
from pymongo import MongoClient
from multiprocessing import Pool
import argparse
from dynaconf import settings
from pprint import pprint
import sys
import os

sys.path.append('../')
from feature.moment import get_all_vectors
from feature_reduction.feature_reduction import reducer
from feature_reduction.utils import get_term_weight_pairs


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-k', '--k_latent_semantics', type=int, required=True)
    parser.add_argument(
        '-frt', '--feature_reduction_technique', type=str, required=True)
    parser.add_argument('-i', '--image_name', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str)
    return parser


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    #get the absolute data path
    data_path = settings.DATA_PATH if args.data_path is None else args.data_path
    data_path = os.path.abspath(data_path) + '/'

    #get query image name and vector
    print(data_path + args.image_name)
    query_path, query_vector = get_all_vectors(f={
        'path': {
            '$eq': data_path + args.image_name
        }
    })
    print(query_vector)

    # Get all vectors and run dim reduction on them.
    # Also pass query vector to apply the same scale and dim reduction transformation
    all_images, all_vectors = get_all_vectors()
    reduced_dim_vectors, _, _, reduced_query_vector = reducer(
        all_vectors,
        args.k_latent_semantics,
        args.feature_reduction_technique,
        query_vector = query_vector[0].reshape(1, -1)
    )
    print(reduced_dim_vectors.shape)
    print(reduced_query_vector.shape)
