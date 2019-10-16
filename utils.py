import os
import shutil
from subprocess import Popen, PIPE, STDOUT
import pandas as pd
from sklearn.cluster import KMeans
from pymongo import MongoClient
from dynaconf import settings

from feature import moment, sift, lbp, hog

def dummy(*args, **kwargs):
    raise NotImplementedError


vectors_getters = {
    "moment": {
        "coll": settings.MOMENT.collection,
        "func": moment.get_all_vectors
     },
    "moment_inv": {
        "coll": settings.MOMENT.collection_inv,
        "func": moment.get_all_vectors
     },
    "sift": {
        "coll": settings.SIFT.collection,
        "func": sift.get_all_vectors
    },
    "lbp": {
        "coll": settings.LBP.collection,
        "func": lbp.get_all_vectors
    },
    "hog": {
        "coll": settings.HOG.collection,
        "func": hog.get_all_vectors
    }
}


def filter_images(label):
    """filters images based on label
    
    Arguments:
        data_directory {str} -- directory of hands data with trailing /
        label {str} -- valid label to filter
    
    Returns:
        list -- list of image paths with given label
    """
    #map on which column to check based on filter values
    filter_to_column = {
        'left': 'aspectOfHand',
        'right': 'aspectOfHand',
        'dorsal': 'aspectOfHand',
        'palmar': 'aspectOfHand',
        'male': 'gender',
        'female': 'gender',
        'with_acs': 'accessories',
        'without_acs': 'accessories'
    }
    if label not in filter_to_column.keys():
        raise Exception('invalid filter. valid filters are ' + ', '.join(list(filter_to_column.keys())))

    #get image paths where the filter holds true
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    coll = client.db[settings.IMAGES.METADATA_COLLECTION]
    column = filter_to_column[label]
    
    filter_image_paths = []
    for row in coll.find({column: {'$regex': label}}, {'path':1}):
        filter_image_paths.append(row['path'])
    return filter_image_paths

def get_all_vectors(model, f={}):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    inst = vectors_getters[model]
    coll = client.db[inst["coll"]]

    return inst["func"](coll, f)


def get_metadata(f={}):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    return list(client.db[settings.IMAGES.METADATA_COLLECTION].find(f))


def get_term_weight_pairs(components):
    """returns array of weights of original features for each latent dimension
    
    Arguments:
        components {np.array} -- numpy array of size no_of_latent_semantics * no_of_original_features
    
    Returns:
        list -- list of feature weight pairs
    """
    term_weight_pairs = []
    for weights in components:
        feature_weights = [(index, weights[index]) for index in range(len(weights))]
        feature_weights.sort(key = lambda ele: ele[1], reverse=True)
        term_weight_pairs.append(feature_weights)
    return term_weight_pairs

def get_centroid(matrix):
    km = KMeans(n_clusters=1).fit(matrix)
    return km.cluster_centers_.flatten()

def get_negative_label(label):
    negative_label_map = {
        'dorsal':'palmar',
        'palmar':'dorsal',
        'left':'right',
        'right':'left',
        'with_acs':'without_acs',
        'without_acs':'with_acs',
        'male':'female',
        'female':'male'
    }
    return negative_label_map[label]


