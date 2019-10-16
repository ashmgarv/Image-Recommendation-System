from pathlib import Path
import argparse
from dynaconf import settings
import sys
import os

sys.path.append('../')
from metric.distance import distance
from feature_reduction.feature_reduction import reducer
from utils import get_all_vectors, filter_images, get_centroid, get_negative_label

def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-k', '--k_latent_semantics', type=int, required=True)
    parser.add_argument('-frt', '--feature_reduction_technique', type=str, required=True)
    parser.add_argument('-l', '--label', type=str, required=True)
    parser.add_argument('-i', '--image_name', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str)
    return parser

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    # get the absolute data path
    data_path = Path(settings.path_for(settings.DATA_PATH) if args.data_path is None else args.data_path)

    # get query image name and vector
    if os.sep not in args.image_name:
        query_path = data_path / args.image_name
    else:
        query_path = data_path

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

    # Run dimensionality reduction across label vectors and pass the query vector to apply the same to it as well.
    reduced_dim_vectors, _, _, reduced_query_vector = reducer(
        label_vectors,
        args.k_latent_semantics,
        args.feature_reduction_technique,
        query_vector=query_vector[0].reshape(1, -1)
    )

    #Compute centroid for the given label.
    centroid_labels =  get_centroid(reduced_dim_vectors)

    distances_to_centroid = []
    #calculate distance to every image from the centroid .
    for data_point in reduced_dim_vectors:
        distances_to_centroid.append(distance(data_point, centroid_labels, 0))

    #Compute distance of the given image vector from centroid of given label.
    distance_label = distance(centroid_labels, reduced_query_vector, 0)

    #Check if distance_label falls in between the min and max of the distances vector, if yes, given label if the answer.
    if min(distances_to_centroid) <= distance_label <= max(distances_to_centroid):
        print(f"Label of the given image is : {args.label.upper()}")
    else:
        print(f"Label of the given image is : {get_negative_label(args.label).upper()}")






