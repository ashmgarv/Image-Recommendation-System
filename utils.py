import os
import shutil
from subprocess import Popen, PIPE, STDOUT
import pandas as pd
from sklearn.cluster import KMeans
from pymongo import MongoClient
from dynaconf import settings
from pathlib import Path
import csv

from feature import moment, sift, lbp, hog
from output import print_term_weight_pairs

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


def filter_images(label, unlabelled_db=False):
    """filters images based on label
    
    Arguments:
        data_directory {str} -- directory of hands data with trailing /
        label {str} -- valid label to filter
    
    Returns:
        list -- list of image paths with given label
    """
    #map on which column to check based on filter values
    filter_to_column_values = {
        'left': {'column' :'aspectOfHand', 'values': ['dorsal left', 'palmar left']},
        'right': {'column': 'aspectOfHand', 'values': ['dorsal right', 'palmar right']},
        'dorsal': {'column': 'aspectOfHand', 'values': ['dorsal left', 'dorsal right']},
        'palmar': {'column': 'aspectOfHand', 'values': ['palmar left', 'palmar right']},
        'male': {'column': 'gender', 'values': ['male']},
        'female': {'column': 'gender', 'values': ['female']},
        'with_acs': {'column': 'accessories', 'values': ['with_acs']},
        'without_acs': {'column': 'accessories', 'values': ['without_acs']},
    }
    if label not in filter_to_column_values.keys():
        raise Exception('invalid filter. valid filters are ' + ', '.join(list(filter_to_column_values.keys())))

    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    database = settings.QUERY_DATABASE if unlabelled_db else settings.DATABASE
    coll = client[database][settings.IMAGES.METADATA_COLLECTION]

    #get column and filter values
    column = filter_to_column_values[label]['column']
    values = filter_to_column_values[label]['values']

    filter_image_paths = []
    for row in coll.find({column: {'$in': values}}, {'path':1}):
        filter_image_paths.append(row['path'])
    return filter_image_paths

def get_all_vectors(model, f={},unlabelled_db=False):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    inst = vectors_getters[model]
    database = settings.QUERY_DATABASE if unlabelled_db else settings.DATABASE
    coll = client[database][inst["coll"]]

    return inst["func"](coll, f)


def get_metadata(f={}, unlabelled_db=False):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    return list(
        client[settings.QUERY_DATABASE if unlabelled_db else settings.DATABASE]
        [settings.IMAGES.METADATA_COLLECTION].find(f))


"""
    Function:  To build a dictionary with subject-ID as key and
                data matrix(feature vectors per image) as value.
    Arguments: List of Subject-IDs.
    Returns:   Dictionary.
"""
def get_subject_image_vectors(model):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    subjects_id_list = list(client.db[settings.IMAGES.METADATA_COLLECTION].distinct("id"))
    subject_data = {}
    images_per_subject = {}
    k_list = []
    for sub in subjects_id_list:
        image_paths = list(client.db[settings.IMAGES.METADATA_COLLECTION].find({"id":sub},{"_id":0,"path":1}))
        images = []
        k_list.append(len(image_paths))
        for x in range(len(image_paths)):
            images.append(image_paths[x]['path'])
            images_per_subject[sub] = images
        _, subject_image_vectors = get_all_vectors(model, f={
            'path': {
                '$in': images
            }
        })
        subject_data[sub] = subject_image_vectors
    return subject_data,  min(k_list), images_per_subject

"""
    Returns subject attributes for a given subject-ID.
"""
def get_subject_attributes(subject_id):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    return list(client.db[settings.IMAGES.METADATA_COLLECTION].find({"id":subject_id},{"_id":0, "id" : 1,
        "age" : 1,
        "gender" : 1,
        "skinColor" : 1,
        #"accessories" : 1,
        #"nailPolish" : 1,
        #"aspectOfHand" : 1,
        #"path" : 1,
        #"irregularities" : 1
        }))

def get_term_weight_pairs(components, file_name):
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
    print_term_weight_pairs(term_weight_pairs, file_name)

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
