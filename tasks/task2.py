import numpy as np
from pathlib import Path
from pymongo import MongoClient
from multiprocessing import Pool
import argparse
from dynaconf import settings
import sys
import os
import pandas as pd

sys.path.append('../')
from output import write_to_file
from feature.moment import get_all_vectors
from feature_reduction.feature_reduction import reducer
from utils import get_all_vectors, filter_images
from classification.kmeans import Kmeans


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--n_clusters', type=int, required=True)
    return parser


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()
    
    n_clusters = args.n_clusters

    #get the absolute data path and models whose features to concatenate
    data_path = Path(settings.path_for(settings.DATA_PATH))
    models = settings.TASK2_CONFIG.MODELS

    #Fetch training data for dorsal and palmer images from LABELLED DB
    dorsal_paths = filter_images('dorsal')
    dorsal_vectors, palmar_vectors = np.array([]), np.array([])
    for model in models:
        dorsal_paths, temp = get_all_vectors(model, f={'path': {'$in': dorsal_paths}})
        if not dorsal_vectors.size: dorsal_vectors = temp
        else: dorsal_vectors = np.concatenate((dorsal_vectors, temp), axis=1)

        palmar_paths, temp = get_all_vectors(model, f={'path': {'$nin': dorsal_paths}})
        if not palmar_vectors.size: palmar_vectors = temp
        else: palmar_vectors = np.concatenate((palmar_vectors, temp), axis=1)
    
    #Fetch test data for dorsal and palmer images from LABELLED DB
    test_data = np.array([])
    for model in models:
        test_data_paths, temp = get_all_vectors(model, unlabelled_db=True)
        if not test_data.size: test_data = temp
        else: test_data = np.concatenate((test_data, temp), axis=1)
    
    #apply frt on dorsal and palmar vectors and project test data onto dorsal and palmar reduced feature spaces
    frt = settings.TASK2_CONFIG.FRT
    k = settings.TASK2_CONFIG.K
    dorsal_vectors, _, _, q_dorsal_vectors = reducer(dorsal_vectors, k, frt, query_vector=test_data)
    palmar_vectors, _, _, q_palmar_vectors = reducer(palmar_vectors, k, frt, query_vector=test_data)

    #Get centroids and centroid_labels for dorsal and palmar vectors
    print("Clustering dorsal vectors")
    dorsal_kmeans = Kmeans(dorsal_vectors, n_clusters)
    dorsal_kmeans.cluster()
    print("Clustering Palmar vectors")
    palmar_kmeans = Kmeans(palmar_vectors, n_clusters)
    palmar_kmeans.cluster()

    #predict with score comparison
    predicted_labels = []
    for i in range(len(test_data)):
        dorsal_score = dorsal_kmeans.get_silhoutte_score(q_dorsal_vectors[i])
        palmar_score = palmar_kmeans.get_silhoutte_score(q_palmar_vectors[i])
        if dorsal_score > palmar_score: predicted_labels.append(1) 
        else: predicted_labels.append(0)

    images = [path.split('/')[-1] for path in test_data_paths]
    labels = ['dorsal' if each == 1 else 'palmar' for each in predicted_labels]
    df = pd.DataFrame()
    df['images'] = images
    df['labels'] = labels
    print(df)