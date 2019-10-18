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
from utils import get_all_vectors, filter_images, get_term_weight_pairs


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-k', '--k_latent_semantics', type=int, required=True)
    parser.add_argument(
        '-frt', '--feature_reduction_technique', type=str, required=True)
    parser.add_argument('-l', '--label', type=str, required=True)
    return parser


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    #filter image
    paths = filter_images(args.label)

    #get query image name and vector
    images, data_matrix = get_all_vectors(args.model, f={
        'path': {
            '$in': paths
        }
    })

    # Get all vectors and run dim reduction on them.
    vectors, eigen_values, latent_vs_old = reducer(
        data_matrix, args.k_latent_semantics, args.feature_reduction_technique)

    file_name = "images_vs_latent_{}_{}_{}.csv".format(args.model, args.feature_reduction_technique, args.label)
    get_term_weight_pairs(vectors, file_name)

    file_name = "latent_vs_features_{}_{}_{}.csv".format(args.model, args.feature_reduction_technique, args.label)
    get_term_weight_pairs(latent_vs_old, file_name)

    # Extra Credit
    # image path with a vector in the latent semantic space
    data_z = zip(images, vectors)
    # image path for each latenet semantic in h
    feature_z = [(idx, images[np.argmax(np.dot(data_matrix, i))]) for idx, i in enumerate(latent_vs_old)]

    output.write_to_file("visualize_data_z.html",
                         "data-z-{}-{}.html".format(args.model, args.feature_reduction_technique),
                         data_z=data_z,
                         title="TEST")

    output.write_to_file("visualize_feat_z.html",
                         "feat-z-{}-{}.html".format(args.model, args.feature_reduction_technique),
                         feature_z=feature_z,
                         title="TEST")

