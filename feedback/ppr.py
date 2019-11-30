import argparse
import sys
from dynaconf import settings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import softmax
from pathlib import Path

from feature_reduction.feature_reduction import reducer
from metric import distance, similarity
from utils import get_metadata, get_term_weight_pairs, get_all_vectors, filter_images

import output
import time

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


def math_method(adj_matrix, alpha):
    return np.linalg.inv(
        np.identity(adj_matrix.shape[0]) - (alpha * adj_matrix))


def power_iteration(adj_matrix, alpha, seed):
    n = 0
    pi_1 = None
    np.random.seed()
    pi_2 = np.random.rand(len(seed))

    while True:
        pi_1 = np.matmul(np.dot(alpha, adj_matrix), pi_2) + np.dot(
            (1 - alpha), seed)
        if np.allclose(pi_1, pi_2):
            print("Reached steady state")
            break
        pi_2 = pi_1
        n += 1

    return pi_1, n


def get_metadata_space(images, unlabelled=False):
    meta = get_metadata(unlabelled_db=unlabelled)
    # Mapping between image file path name and the metadata
    meta = {m['path']: m for m in meta}
    space = np.array([[
        meta[i]['age'], mapping[meta[i]['gender']],
        mapping[meta[i]['skinColor']], mapping[meta[i]["accessories"]],
        meta[i]["nailPolish"], meta[i]["irregularities"]
    ] for i in images])

    return meta, space


def get_data_matrix(feature, unlabelled=False):
    # Get labelled images
    images, data = get_all_vectors(feature, unlabelled_db=unlabelled)

    # Get metadata
    meta = get_metadata(unlabelled_db=unlabelled)
    meta = {m['path']: m for m in meta}

    return images, meta, data


def prepare_data(k, frt, feature):
    min_max_scaler = MinMaxScaler()

    l_images, l_meta, l_matrix = get_data_matrix(feature)
    _, l_meta_matrix = get_metadata_space(l_images)
    l_matrix = np.c_[l_matrix, l_meta_matrix]

    u_images, u_meta, u_matrix = get_data_matrix(feature, unlabelled=True)
    _, u_meta_matrix = get_metadata_space(u_images, unlabelled=True)
    u_matrix = np.c_[u_matrix, u_meta_matrix]

    meta = l_meta
    meta.update(u_meta)

    matrix = min_max_scaler.fit_transform(np.vstack((
        l_matrix,
        u_matrix,
    )))
    matrix, _, _ = reducer(matrix, k, frt)

    return l_images + u_images, meta, matrix


def prepare_ppr_graph_from_data(a_matrix, edges):
    graph = 1 / (euclidean_distances(a_matrix) + 1)
    np.fill_diagonal(graph, 0)

    if edges < len(graph):
        nth = np.partition(graph, -1 * edges, axis=1)[:, -1 * edges]
        graph[graph < nth[:, None]] = 0

    graph = (graph.T / graph.sum(axis=1)).T
    return graph


def ppr_feedback(relevant_images, irrelevant_images):
    images, meta, matrix = prepare_data(58, 'nmf', 'moment_inv')
    graph = prepare_ppr_graph_from_data(matrix, 120)
    metadata = get_metadata() + get_metadata(unlabelled_db=True)

    image_meta = {m['path'].split('/')[-1]: m for m in metadata}

    img_to_label = {}
    for img in relevant_images:
        img_to_label[image_meta[img]['path']] = 'relevant'
    for img in irrelevant_images:
        img_to_label[image_meta[img]['path']] = 'irrelevant'

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

    #steady_state, _ = power_iteration(graph, 0.55, seed)
    inv = math_method(graph, 0.55)
    steady_state = np.dot(inv, (1 - 0.55) * seed)

    result = [(images[i], steady_state[i])
              for i in np.flip(steady_state.argsort()) if images[i] not in
              [key for key in img_to_label
               if img_to_label[key] == 'relevant']][:20]

    output.write_to_file("task3.html",
                         "xyz.html",
                         ranks=result,
                         keys=list(img_to_label.keys()),
                         title="TEST")

