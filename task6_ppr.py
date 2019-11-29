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


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k-edges', type=int, required=True)
    parser.add_argument('-t', '--result-size', type=int, required=True)
    parser.add_argument('--math', default=False, action='store_true')
    parser.add_argument('--feature', default='sift', type=str)
    # 58 works good too, but might include some accessories for some reason
    parser.add_argument('--k-latent', default=20, type=int)
    parser.add_argument('--frt', default='pca', type=str)
    parser.add_argument('--alpha', default=0.15, type=float)
    return parser


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


def prepare_ppr_with_label_nodes(labels, img_to_label, k, frt, feature,
                                 convergence, alpha, edges):
    min_max_scaler = MinMaxScaler()

    l_images, l_meta, l_matrix = get_data_matrix(feature)
    _, l_meta_matrix = get_metadata_space(l_images)
    l_matrix = np.c_[l_matrix, l_meta_matrix]

    l_matrix = min_max_scaler.fit_transform(l_matrix)

    u_images, u_meta, u_matrix = get_data_matrix(feature, unlabelled=True)
    _, u_meta_matrix = get_metadata_space(u_images, unlabelled=True)
    u_matrix = np.c_[u_matrix, u_meta_matrix]

    u_matrix = min_max_scaler.fit_transform(u_matrix)

    label_img_indices = {label: [] for label in labels}
    label_seeds = {label: np.zeros(len(l_matrix)) for label in labels}
    label_states = {label: None for label in labels}

    for i, img in enumerate(l_images):
        if img in img_to_label:
            label_img_indices[img_to_label[img]].append(i)
            label_seeds[img_to_label[img]][i] = 1

    for label in label_img_indices:
        label_seeds[label] /= len(label_img_indices[label])

    adj = 1 / (euclidean_distances(l_matrix) + 1)
    np.fill_diagonal(adj, 0)

    adj = (adj.T / adj.sum(axis=1)).T

    if convergence == POWER_ITR:
        for label in label_states:
            label_states[label], _ = power_iteration(adj, alpha,
                                                     label_seeds[label])
    elif convergence == MATH:
        inv = math_method(adj, alpha)
        for label in label_states:
            label_states[label] = np.dot(inv, (1 - alpha) * label_seeds[label])
    else:
        raise Exception(
            "Invalid convergence criteria for PPR: {}".format(convergence))

    label_nodes = {
        label: np.zeros(len(l_matrix) + len(u_matrix))
        for label in labels
    }

    for l in label_img_indices:
        for idx in label_img_indices[l]:
            label_nodes[l][idx] = label_states[l][idx]

    a_matrix = np.vstack((
        l_matrix,
        u_matrix,
    ))
    graph = 1 / (euclidean_distances(a_matrix) + 1)
    np.fill_diagonal(graph, 0)

    if edges < len(graph):
        nth = np.partition(graph, -1 * edges, axis=1)[:, -1 * edges]
        graph[graph < nth[:, None]] = 0

    graph = np.r_[graph, [label_nodes[label] for label in labels]]
    for node in [
            np.append(label_nodes[label], [0] * len(labels))
            for label in labels
    ]:
        graph = np.c_[graph, node]
    graph = (graph.T / graph.sum(axis=1)).T

    return l_images + u_images, graph


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


if __name__ == '__main__':
    metadata = get_metadata() + get_metadata(unlabelled_db=True)
    images = {m['path'].split('/')[-1]: m for m in metadata}
    relevant_images = [
        'Hand_0007166.jpg', 'Hand_0007168.jpg', 'Hand_0008622.jpg',
        'Hand_0008628.jpg'
    ]
    irrelevant_images = [
        'Hand_0009376.jpg', 'Hand_0000902.jpg', 'Hand_0011283.jpg',
        'Hand_0008014.jpg'
    ]

    img_to_label = {}
    for img in relevant_images:
        img_to_label[images[img]['path']] = 'relevant'
    for img in irrelevant_images:
        img_to_label[images[img]['path']] = 'irrelevant'
    """
    images, graph = prepare_ppr_with_label_nodes(['relevant', 'irrelevant'],
                                                 img_to_label, 58, 'nmf',
                                                 'moment_inv', 'math_method', 0.15,
                                                 90)
    """
    images, meta, matrix = prepare_data(58, 'nmf', 'moment_inv')

    graph = prepare_ppr_graph_from_data(matrix, 120)

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

    import pdb
    pdb.set_trace()
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
    #parser.add_argument('--dir', type=str, choices=('labelled', 'unlabelled', 'all'), default='all')
