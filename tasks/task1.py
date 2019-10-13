import numpy as np
from pymongo import MongoClient
from multiprocessing import Pool
import argparse
from dynaconf import settings
from pprint import pprint
import sys

sys.path.append('../')
from feature.moment import get_all_vectors as get_all_moment_vectors
from feature_reduction.feature_reduction import reducer
from feature_reduction.utils import get_term_weight_pairs

def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-k', '--k_latent_semantics', type=int, required=True)
    parser.add_argument('-frt','--feature_reduction_technique', type=str, required=True)
    return parser

def get_all_vectors(model):
    """get all vectors present in the model
    
    Arguments:
        model {str} -- model name : sift, moment, hog, lbp
    
    Returns:
        image_labels, vectors -- array of image names and corresponding vectors (2 variables)
    """
    if model == 'moment':
        coll_name = settings.MOMENT.collection
        coll = client.db[coll_name]
        return get_all_moment_vectors(coll)

    if model == 'sift':
        pass
    if model == 'hog':
        pass
    if model == 'lbp':
        pass

if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    images, vectors = get_all_vectors(args.model)
    
    # reducer automatically maps feature_reduction_technique to the right function
    vectors, eigen_values, latent_vs_old = reducer(
        vectors, 
        args.k_latent_semantics, 
        args.feature_reduction_technique
    )
    pprint(get_term_weight_pairs(latent_vs_old), indent=4)
    