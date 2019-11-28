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
from utils import get_metadata, get_term_weight_pairs, get_all_vectors

import output
import time

mapping = {
    "without_acs": 0,
    "with_acs": 1,
    "male": 0,
    "female": 1,
    "very fair": 0,
    "fair": 1,
    "medium": 2,
    "dark": 3,
    "dorsal": 0,
    "palmar": 1,
    "right": 0,
    "left": 1
}


def math_method(adj_matrix, alpha, seed_vector):
    a = np.linalg.inv(np.identity(adj_matrix.shape[0]) - (alpha * adj_matrix))
    return np.dot(a, (1 - alpha) * seed_vector)


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
    parser.add_argument('-K', '--k-dominant', type=int, required=True)
    parser.add_argument('-i', '--image-ids', nargs=3, required=True)
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
        meta[i]["nailPolish"], mapping[meta[i]["aspectOfHand"].split()[0]],
        mapping[meta[i]["aspectOfHand"].split()[1]], meta[i]["irregularities"]
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


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    # Parameters
    feature = args.feature
    k_latent_semantics = args.k_latent
    frt_technique = args.frt
    alpha = args.alpha

    query_images = []

    images, meta, matrix = prepare_data(k_latent_semantics, frt_technique,
                                        feature)

    for image_id in args.image_ids:
        unlabelled_path = Path(settings.path_for(settings.UNLABELED_DATA_PATH))
        labelled_path = Path(settings.path_for(settings.DATA_PATH))

        path = None
        if (labelled_path / image_id).exists() and \
           (labelled_path / image_id).is_file():
            path = (labelled_path / image_id)
        elif (unlabelled_path / image_id).exists() and \
             (unlabelled_path / image_id).is_file():
            path = (unlabelled_path / image_id)

        if path is None:
            raise Exception("Image '{}' not found in '{}' or '{}'".format(
                image_id, str(unlabelled_path.resolve()),
                str(labelled_path.resolve())))

        path_str = str(path.resolve())
        if path_str not in meta:
            raise Exception(
                "Metadata unavailable for {}. Did you run db_make? Does this image have the metadata in '{}' or '{}'?"
                .format(path_str, settings.path_for(settings.METADATA_CSV),
                        settings.path_for(settings.UNLABELED_METADATA_CSV)))
        query_images.append(path_str)

    # Image-Image similarity
    img_img = 1 / (euclidean_distances(matrix) + 1)
    np.fill_diagonal(img_img, 0)
    nth = np.partition(img_img, -1 * args.k_edges,
                       axis=1)[:, -1 * args.k_edges]
    # Mask all numbers below nth largest to 0
    img_img[img_img < nth[:, None]] = 0
    # Softmax to make all edges add upto 1, so as to interpret edge weight as
    # probabilities
    img_img = (img_img.T / img_img.sum(axis=1)).T

    seed_vector = np.array([
        1 / len(query_images) if img in query_images else 0
        for idx, img in enumerate(images)
    ])

    if args.math:
        steady_state = math_method(img_img, alpha, seed_vector)
    else:
        steady_state, num_iter = power_iteration(img_img, alpha, seed_vector)
        print("Converged after {} iterations".format(num_iter))

    #image_indices = np.flip(steady_state.argsort())[:args.k_dominant]
    image_indices = np.flip(steady_state.argsort())[:args.k_dominant +
                                                    len(query_images)]
    result = [(images[i], steady_state[i]) for i in image_indices
              if images[i] not in query_images][:args.k_dominant]

    output.write_to_file(
        "task3.html",
        "task3-{}-{}-{}.html".format(args.k_edges, args.k_dominant,
                                     '-'.join(args.image_ids)),
        ranks=result,
        keys=query_images,
        title="TEST")
