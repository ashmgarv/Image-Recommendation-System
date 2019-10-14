import numpy as np
from pathlib import Path
from pymongo import MongoClient
from multiprocessing import Pool
import argparse
from dynaconf import settings
from pprint import pprint
import sys
import os

sys.path.append('../')
from output import write_to_file
from metric.distance import distance
from feature.moment import get_all_vectors
from feature_reduction.feature_reduction import reducer
from utils import get_term_weight_pairs, get_all_vectors


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-k', '--k_latent_semantics', type=int, required=True)
    parser.add_argument('-frt', '--feature_reduction_technique', type=str, required=True)
    parser.add_argument('-n', '--related_images', type=int, required=True)
    parser.add_argument('-i', '--image_name', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str)
    return parser


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    #get the absolute data path
    data_path = Path(settings.path_for(settings.DATA_PATH) if args.data_path is None else args.data_path)

    #get query image name and vector
    if os.sep not in args.image_name:
        query_path = data_path / args.image_name
    else:
        query_path = data_path

    query_path, query_vector = get_all_vectors(args.model, f={
        'path': {
            '$eq': str(query_path.resolve())
        }
    })

    # Get all vectors and run dim reduction on them.
    # Also pass query vector to apply the same scale and dim reduction transformation
    all_images, all_vectors = get_all_vectors(args.model)
    reduced_dim_vectors, _, _, reduced_query_vector = reducer(
        all_vectors,
        args.k_latent_semantics,
        args.feature_reduction_technique,
        query_vector = query_vector[0].reshape(1, -1)
    )
    
    #calculate distance measures across every image in the reduced label vector set.
    distances = distance(reduced_dim_vectors, reduced_query_vector, 0)
    ranks = [(all_images[i], distances[i]) for i in range(len(distances))]
    ranks.sort(key = lambda t: t[1])
    write_to_file("op_temp.html",
        "{}-{}-{}.html".format(args.image_name, args.model, args.feature_reduction_technique),
        ranks=ranks[:args.related_images],
        key=query_path[0],
        title="TEST")
