import argparse
import sys
from dynaconf import settings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import softmax
from pathlib import Path

sys.path.append('../')
from feature_reduction.feature_reduction import reducer
from metric import distance, similarity
from utils import get_metadata, get_term_weight_pairs, get_all_vectors, filter_images

import pickle
from bson.binary import Binary
from pymongo import MongoClient

import output
import time
from joblib import Memory

POWER_ITR = 'power_iteration'
MATH = 'math_method'

mapping = {
    "without_acs": 0,
    "with_acs": 1,
    "male": 0,
    "female": 1,
    "very fair": 0,
    "fair": 1,
    "medium": 2,
    "dark": 3,
}

CACHE_DIR = Path(settings.path_for(settings.PPR.FEEDBACK.CACHE_DIR))
CACHE = Memory(str(CACHE_DIR), verbose=1)


def math_method(adj_matrix, alpha):
    return np.linalg.inv(
        np.identity(adj_matrix.shape[0]) - (alpha * adj_matrix))


def power_iteration(adj_matrix, alpha, seed):
    n = 0
    pi_1 = None
    pi_2 = np.copy(seed)

    while True:
        pi_1 = np.matmul(np.dot(alpha, adj_matrix), pi_2) + np.dot(
            (1 - alpha), seed)
        if np.allclose(pi_1, pi_2):
            print("Reached steady state")
            break
        pi_2 = pi_1
        n += 1

    return pi_1, n


def get_metadata_space(images):
    meta = get_metadata(master_db=True)
    # Mapping between image file path name and the metadata
    meta = {m['path']: m for m in meta}
    space = np.array([[
        meta[i]['age'], mapping[meta[i]['gender']],
        mapping[meta[i]['skinColor']], mapping[meta[i]["accessories"]],
        meta[i]["nailPolish"], meta[i]["irregularities"]
    ] for i in images])

    return meta, space


def get_data_matrix(feature):
    # Get labelled images
    images, data = get_all_vectors(feature, master_db=True)
    meta, meta_space = get_metadata_space(images)
    matrix = np.c_[data, meta_space]

    return images, meta, matrix


def prepare_data(k, frt, feature):
    min_max_scaler = MinMaxScaler()
    images, meta, matrix = get_data_matrix(feature)
    matrix = min_max_scaler.fit_transform(matrix)
    matrix, _, _ = reducer(matrix, k, frt)

    return images, meta, matrix


def prepare_ppr_graph_from_data(a_matrix, edges):
    graph = 1 / (euclidean_distances(a_matrix) + 1)
    np.fill_diagonal(graph, 0)

    if edges < len(graph):
        nth = np.partition(graph, -1 * edges, axis=1)[:, -1 * edges]
        graph[graph < nth[:, None]] = 0

    graph = (graph.T / graph.sum(axis=1)).T
    return graph


def build_data_inv(k_latent, frt, feature, edges, alpha):
    print("Building inverse matrix")
    images, _, matrix = prepare_data(k_latent, frt, feature)
    graph = prepare_ppr_graph_from_data(matrix, edges)
    return images, math_method(graph, alpha)


def build_power_iteration(k_latent, frt, feature, edges, alpha):
    images, _, matrix = prepare_data(k_latent, frt, feature)
    graph = prepare_ppr_graph_from_data(matrix, edges)
    return images, graph


build_data_inv = CACHE.cache(build_data_inv)
build_power_iteration = CACHE.cache(build_power_iteration)


def ppr_feedback(relevant_images, irrelevant_images):
    k_latent = 25
    frt = 'nmf'
    feature = 'moment'
    edges = 80
    alpha = 0.85

    #images, inv = build_data_inv(k_latent, frt, feature, edges, alpha)
    images, graph = build_power_iteration(k_latent, frt, feature, edges, alpha)
    img_to_label = {}
    for img in relevant_images:
        img_to_label[Path(settings.path_for(settings.MASTER_DATA_PATH)) /
                     img] = 'relevant'
    for img in irrelevant_images:
        img_to_label[Path(settings.path_for(settings.MASTER_DATA_PATH)) /
                     img] = 'irrelevant'

    seed = np.zeros(len(images), dtype=np.float32)
    for i, image in enumerate(images):
        label = img_to_label.get(image, None)
        if label == "relevant":
            seed[i] = 1.0
        elif label == "irrelevant":
            seed[i] = 0.0
        else:
            seed[i] = 0.5

    seed /= seed.sum()

    steady_state, _ = power_iteration(graph, 0.55, seed)
    #steady_state = np.dot(inv, (1 - alpha) * seed)

    result = [
        images[i] for i in np.flip(steady_state.argsort()) if images[i] not in
        [key for key in img_to_label if img_to_label[key] == 'relevant']
    ][:20]

    output.write_to_file(
        "task6.html",
        "xyz6.html",
        relevant=[i for i in img_to_label if img_to_label[i] == "relevant"],
        irrelevant=[
            i for i in img_to_label if img_to_label[i] == "irrelevant"
        ],
        result=result,
        keys=list(img_to_label.keys()),
        title="TEST")


if __name__ == '__main__':
    relevant_images = [
        'Hand_0007166.jpg', 'Hand_0007168.jpg', 'Hand_0008622.jpg',
        'Hand_0008628.jpg'
    ]
    irrelevant_images = [
        'Hand_0009376.jpg', 'Hand_0000902.jpg', 'Hand_0011283.jpg',
        'Hand_0008014.jpg'
    ]

    ppr_feedback(relevant_images, irrelevant_images)
