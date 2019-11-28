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
from feature_reduction.feature_reduction import reducer
from utils import get_all_vectors, filter_images
from classification.kmeans import Kmeans
from sklearn.metrics import accuracy_score


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--n_clusters', type=int, required=True)
    return parser

def predict_label(query, dorsal_kmeans, palmar_kmeans):
    dorsal_dist = dorsal_kmeans.get_closest(query, dorsal_kmeans.centroids, return_min=True)
    palmar_dist = palmar_kmeans.get_closest(query, palmar_kmeans.centroids, return_min=True)
    return 'dorsal' if dorsal_dist <= palmar_dist else 'palmar'

def generate_vec(n=False):
    
    #TRAIN DATA
    dorsal_paths = filter_images(label)
    dorsal_vectors, palmar_vectors = np.array([]), np.array([])
    for model in models:

        paths, temp = get_all_vectors(model, f={'path': {'$in': dorsal_paths}})
        if not dorsal_vectors.size: dorsal_vectors = temp
        else: dorsal_vectors = np.concatenate((dorsal_vectors, temp), axis=1)

        _, temp = get_all_vectors(model, f={'path': {'$nin': dorsal_paths}})
        if not palmar_vectors.size: palmar_vectors = temp
        else: palmar_vectors = np.concatenate((palmar_vectors, temp), axis=1)
    
    if n : dorsal_vectors, palmar_vectors = normalize(dorsal_vectors), normalize(palmar_vectors)
    
    #TEST DATA
    q_dorsal_paths = filter_images(label, unlabelled_db=True)
    q_dorsal_vectors, q_palmar_vectors = np.array([]), np.array([])
    for model in models:

        _, temp = get_all_vectors(model, f={'path': {'$in': q_dorsal_paths}}, unlabelled_db=True)
        if not q_dorsal_vectors.size: q_dorsal_vectors = temp
        else: q_dorsal_vectors = np.concatenate((q_dorsal_vectors, temp), axis=1)

        _, temp = get_all_vectors(model, f={'path': {'$nin': q_dorsal_paths}}, unlabelled_db=True)
        if not q_palmar_vectors.size: q_palmar_vectors = temp
        else: q_palmar_vectors = np.concatenate((q_palmar_vectors, temp), axis=1)

    q_dorsal_class = np.array([1] * len(q_dorsal_vectors))
    q_palmar_class = np.array([0] * len(q_palmar_vectors))
    
    test_data = np.vstack((q_dorsal_vectors, q_palmar_vectors))
    test_labels = np.concatenate((q_dorsal_class, q_palmar_class))
    
    if n : test_data = normalize(test_data)
    return dorsal_vectors, palmar_vectors, test_data, test_labels
    

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()
    n_clusters = args.n_clusters

    #get the absolute data path and models whose features to concatenate
    data_path = Path(settings.path_for(settings.DATA_PATH))
    model = settings.TASK2_CONFIG.MODEL

    #Fetch training data for dorsal and palmer images from LABELLED DB
    dorsal_paths = filter_images('dorsal')
    dorsal_paths, dorsal_vectors = get_all_vectors(model, f={'path': {'$in': dorsal_paths}})
    palmar_paths, palmar_vectors = get_all_vectors(model, f={'path': {'$nin': dorsal_paths}})
    
    #Fetch test data from UNLABELLED DB
    test_data_paths, test_data = get_all_vectors(model, unlabelled_db=True)

    #Get centroids and centroid_labels for dorsal and palmar vectors
    print("Clustering dorsal vectors")
    dorsal_kmeans = Kmeans(dorsal_vectors, n_clusters)
    dorsal_kmeans.cluster()
    print("Clustering Palmar vectors")
    palmar_kmeans = Kmeans(palmar_vectors, n_clusters)
    palmar_kmeans.cluster()

    #compare distance to dorsal and palmar centroid to label
    vec_func = np.vectorize(predict_label)
    labels = [predict_label(each, dorsal_kmeans, palmar_kmeans) for each in test_data]
    write_to_file("task4.html",
                    "task2-{}.html".format(n_clusters),
                    predictions=zip(test_data_paths, labels),
                    title="TEST")
    
    #write cluster image paths to HTML
    temp = pd.DataFrame(list(zip(dorsal_paths, dorsal_kmeans.closest)), columns = ['path', 'cluster'])
    dorsal_clusters = list(temp.groupby(['cluster'])['path'].apply(lambda x: x.values.tolist()))
    temp = pd.DataFrame(list(zip(palmar_paths, palmar_kmeans.closest)), columns = ['path', 'cluster'])
    palmar_clusters = list(temp.groupby(['cluster'])['path'].apply(lambda x: x.values.tolist()))
    write_to_file("clusters.html",
                    "cluster.html",
                    dorsal_clusters = dorsal_clusters,
                    palmar_clusters = palmar_clusters,
                    title = 'clusterbois'
    )
