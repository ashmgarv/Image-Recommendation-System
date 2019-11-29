from dynaconf import settings
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
sys.path.append('../')
import utils
import classification.helper as helper

POWER_ITR = 'power_iteration'
MATH = 'math_method'


def math_method(adj_matrix, alpha):
    """
    a = np.linalg.inv(np.identity(adj_matrix.shape[0]) - (alpha * adj_matrix))
    return np.dot(a, (1 - alpha) * seed_vector), 0
    """
    return np.linalg.inv(
        np.identity(adj_matrix.shape[0]) - (alpha * adj_matrix))


def power_iteration(adj_matrix, alpha, seed):
    n = 0
    pi_1 = None
    np.random.seed()
    pi_2 = softmax(np.random.rand(len(seed)))

    while True:
        pi_1 = np.matmul(np.dot(alpha, adj_matrix), pi_2) + np.dot(
            (1 - alpha), seed)
        if np.allclose(pi_1, pi_2):
            break
        pi_2 = pi_1
        n += 1

    return pi_1, n


def prepare_ppr(train_data, test_data, frt, k, feature, edges, alpha,
                convergence):

    dorsal_seed = np.zeros(len(train_data), dtype=np.float32)
    palmar_seed = np.zeros(len(train_data), dtype=np.float32)

    dorsal_count = 0
    palmar_count = 0

    dorsal_index = []
    palmar_index = []

    for idx, item in enumerate(train_data):
        if item[-1] == 1.0:
            palmar_seed[idx] = 1.0
            palmar_count += 1
            palmar_index.append(idx)
        else:
            dorsal_seed[idx] = 1.0
            dorsal_count += 1
            dorsal_index.append(idx)

    dorsal_seed /= dorsal_count
    palmar_seed /= palmar_count

    l_matrix = np.array(train_data)[:, :-1]

    adj = 1 / (euclidean_distances(l_matrix) + 1)
    np.fill_diagonal(adj, 0)

    adj = (adj.T / adj.sum(axis=1)).T

    if convergence == POWER_ITR:
        ds, _ = power_iteration(adj, alpha, dorsal_seed)
        ps, _ = power_iteration(adj, alpha, palmar_seed)
    elif convergence == MATH:
        inv = math_method(adj, alpha)
        ds = np.dot(inv, (1 - alpha) * dorsal_seed)
        ps = np.dot(inv, (1 - alpha) * palmar_seed)
    else:
        raise Exception(
            "Invalid convergence criteria for PPR: {}".format(convergence))

    dorsal_node = np.zeros(len(train_data) + len(test_data))
    palmar_node = np.zeros(len(train_data) + len(test_data))

    for idx in dorsal_index:
        dorsal_node[idx] = ds[idx]
    for idx in palmar_index:
        palmar_node[idx] = ps[idx]

    a_matrix = np.vstack((
        l_matrix,
        np.array(test_data)[:, :-1],
    ))
    graph = 1 / (euclidean_distances(a_matrix) + 1)

    if edges < len(graph):
        nth = np.partition(graph, -1 * edges, axis=1)[:, -1 * edges]
        graph[graph < nth[:, None]] = 0

    graph = np.vstack((
        graph,
        dorsal_node,
        palmar_node,
    ))
    graph = np.c_[graph,
                  dorsal_node.tolist() + [0, 0],
                  palmar_node.tolist() + [0, 0]]
    np.fill_diagonal(graph, 0)
    graph = (graph.T / graph.sum(axis=1)).T

    return graph


def ppr_classifier(train_data, test_data, frt, k, feature, edges, alpha,
                   convergence):

    graph = prepare_ppr(train_data, test_data, frt, k, feature, edges, alpha,
                        convergence)
    predictions = np.zeros(len(test_data))

    palmar_symbol = 1.0
    dorsal_symbol = 0.0

    inv = None
    if convergence == MATH:
        inv = math_method(graph, alpha)

    for idx in range(len(test_data)):
        seed_vector = np.zeros(len(graph))
        seed_vector[len(train_data) + idx] = 1

        if convergence == POWER_ITR:
            steady_state, _ = power_iteration(graph, alpha, seed_vector)
        elif convergence == MATH:
            steady_state = np.dot(inv, (1 - alpha) * seed_vector)
        else:
            raise Exception(
                "Invalid convergence for PPR: {}".format(convergence))

        dorsal_prob = steady_state[len(steady_state) - 2]
        palmar_prob = steady_state[len(steady_state) - 1]

        predictions[idx] = \
            palmar_symbol if palmar_prob > dorsal_prob else dorsal_symbol

    return predictions.tolist()


def evaluate(dataset):
    n_folds = 3
    scores = helper.evaluate_algorithm(dataset, ppr_classifier, n_folds)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))


if __name__ == "__main__":
    from feature_reduction.feature_reduction import reducer
    min_max_scaler = MinMaxScaler()

    images, feature_space = utils.get_all_vectors(
        settings.PPR.CLASSIFIER.FEATURE)
    feature_space = min_max_scaler.fit_transform(feature_space)

    meta = utils.get_metadata()
    meta = {m['path']: m for m in meta}
    """
    u_images, u_feature_space = utils.get_all_vectors(
        settings.PPR.CLASSIFIER.FEATURE, unlabelled_db=True)
    u_feature_space = min_max_scaler.fit_transform(u_feature_space)

    matrix = np.vstack((
        feature_space,
        u_feature_space,
    ))

    matrix, eigen_values, latent_vs_old = reducer(
        matrix,
    """
    matrix, eigen_values, latent_vs_old = reducer(feature_space,
                                                  settings.PPR.CLASSIFIER.K,
                                                  settings.PPR.CLASSIFIER.FRT)

    dm = helper.build_matrix_with_labels(matrix, images, meta)
    """
    dm = helper.build_labelled_matrix(matrix, images + u_images,
                                      'aspectOfHand')
    """
    evaluate(dm)
