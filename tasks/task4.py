import numpy as np
from pathlib import Path
import argparse
from dynaconf import settings
import sys
import os

sys.path.append('../')
from output import write_to_file
from metric.distance import distance
from feature_reduction.feature_reduction import reducer
from utils import get_all_vectors, filter_images


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-k', '--k_latent_semantics', type=int, required=True)
    parser.add_argument('-frt', '--feature_reduction_technique', type=str, required=True)
    parser.add_argument('-l', '--label', type=str, required=True)
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
        query_path = Path(args.image_name)
        args.image_name = query_path.name

    query_path, query_vector = get_all_vectors(args.model, f={
        'path': {
            '$eq': str(query_path.resolve())
        }
    })

    # Use filter_images to filter all vectors query and run dim reduction on them.
    # Also pass query vector to apply the same scale and dim reduction transformation
    label_images = filter_images(args.label)
    label_images, label_vectors = get_all_vectors(args.model, f={
        'path': {
            '$in': label_images
        }
    })

    #Run dimensionality reduction across label vectors and pass the query vector to apply the same to it as well.
    reduced_dim_vectors, _, _, reduced_query_vector = reducer(
        label_vectors,
        args.k_latent_semantics,
        args.feature_reduction_technique,
        query_vector = query_vector[0].reshape(1, -1)
    )
    
    #calculate distance measures across every image in the reduced label vector set.
    distances = distance(reduced_dim_vectors, reduced_query_vector, 0)
    ranks = [(label_images[i], distances[i]) for i in range(len(distances))]
    ranks.sort(key = lambda t: t[1])
    write_to_file("op_temp.html",
        "{}-{}-{}-{}.html".format(args.image_name, args.model, args.feature_reduction_technique, args.label),
        ranks=ranks[:args.related_images],
        key=query_path[0],
        title="TEST")
