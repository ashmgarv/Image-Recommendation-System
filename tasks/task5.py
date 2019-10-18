import numpy as np
from sklearn.svm import OneClassSVM
from hyperopt import Trials, fmin, hp, tpe
from dynaconf import settings
import argparse
from pathlib import Path
import os

import sys
sys.path.append('../')

from utils import get_all_vectors, filter_images
from feature_reduction.feature_reduction import reducer
from optimize_fit import get_ideal_clf

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
    data_path = Path(settings.path_for(settings.DATA_PATH) if args.data_path is None else args.data_path)

    #get query image name and vector
    if os.sep not in args.image_name:
        query_path = data_path / args.image_name
    else:
        query_path = data_path
    
    # Get the best fit classifier
    clf = get_ideal_clf(args.model, args.label, args.k_latent_semantics, args.feature_reduction_technique)
    
    # Get query vector and positive label vectors
    _, query_vector = get_all_vectors(args.model, f={
        'path': {
            '$eq': str(query_path.resolve())
        }
    })
    plabel_images_paths = filter_images(args.label)
    _, plabel_vectors = get_all_vectors(args.model, f={'path': {'$in': plabel_images_paths}})

    #transform query vector to latent semantic space of the +ve label vectors
    reduced_plabel_vectors, _, _, reduced_query_vector = reducer(
        plabel_vectors,
        args.k_latent_semantics,
        args.feature_reduction_technique,
        query_vector = query_vector
    )

    #fit and predict. 1 if same label, -1 if opposite
    clf.fit(reduced_plabel_vectors)
    prediction = clf.predict(reduced_query_vector.reshape(1,-1))[0]
    if prediction == 1: print('image is {}'.format(args.label))
    else: 
        label_pairs = [
            ['left', 'right'],
            ['dorsal', 'palmar'],
            ['male', 'female'],
            ['with_acs', 'without_acs'],
        ]
        temp = list(filter(lambda each: each[0] == args.label or each[1] == args.label, label_pairs))
        temp = list(temp)[0]
        temp.remove(args.label)
        print('image is {}'.format(temp[0]))
