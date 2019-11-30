import argparse
import sys
from dynaconf import settings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.path.append('../')
from classification import helper
from classification.ppr_classifier import ppr_classifier
from classification.decision_tree import decision_tree
from classification.svm import run_svm
from feature_reduction.feature_reduction import reducer

import output
import utils


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-E', '--evaluate', action='store_true', default=False)
    parser.add_argument('-c',
                        '--classifier',
                        type=str,
                        required=True,
                        choices=['svm', 'decision', 'ppr'])

    # Can add classifier related args here, if required

    # PPR Classifier options
    parser.add_argument('--k-latent-semantics',
                        type=int,
                        default=settings.PPR.CLASSIFIER.K)
    parser.add_argument('--graph-edges',
                        type=int,
                        default=settings.PPR.CLASSIFIER.EDGES)
    parser.add_argument('--frt', type=str, default=settings.PPR.CLASSIFIER.FRT)
    parser.add_argument('--alpha',
                        type=float,
                        default=settings.PPR.CLASSIFIER.ALPHA)
    parser.add_argument('--model',
                        type=str,
                        default=settings.PPR.CLASSIFIER.FEATURE)
    parser.add_argument('--convergence',
                        type=str,
                        default=settings.PPR.CLASSIFIER.CONVERGENCE_METHOD)
    parser.add_argument('--ignore-metadata',
                        action='store_true',
                        default=settings.PPR.CLASSIFIER.IGNORE_METADATA)

    # Decision Tree options
    parser.add_argument('-d', '--max-depth', type=int)
    parser.add_argument('-s', '--max-size', type=int)

    return parser


class PreparePPRData(object):
    mapping = {
        "without_acs": 0,
        "with_acs": 1,
        "male": 0,
        "female": 1,
        "very fair": 0,
        "fair": 1,
        "medium": 2,
        "dark": 3,
        "right": 0,
        "left": 1
    }

    @classmethod
    def get_metadata_space(cls, images, unlabelled_db=False):
        meta = utils.get_metadata(unlabelled_db=unlabelled_db)
        # Mapping between image file path name and the metadata
        meta = {m['path']: m for m in meta}
        space = np.array([[
            meta[i]['age'], cls.mapping[meta[i]['gender']],
            cls.mapping[meta[i]['skinColor']],
            cls.mapping[meta[i]["accessories"]], meta[i]["nailPolish"],
            meta[i]["irregularities"]
        ] for i in images])

        return meta, space

    @classmethod
    def get_data_matrix(cls,
                        feature,
                        label=None,
                        unlabelled=False,
                        ignore_metadata=False):
        min_max_scaler = MinMaxScaler()

        f = None
        if label:
            label_images = utils.filter_images(label)
            f = {'path': {'$in': label_images}}

        # Build and scale feature matrix
        images, feature_space = utils.get_all_vectors(feature,
                                                      f=f,
                                                      unlabelled_db=unlabelled)
        feature_space = min_max_scaler.fit_transform(feature_space)
        # Not including metadata boosts accuracy of Set 2
        # Including metadata boosts accuracy of Set 1
        if ignore_metadata:
            meta = utils.get_metadata(unlabelled_db=unlabelled)
            # Mapping between image file path name and the metadata
            meta = {m['path']: m for m in meta}
            return images, meta, feature_space

        # Build and scale metadata matrix
        meta, metadata_space = cls.get_metadata_space(images,
                                                      unlabelled_db=unlabelled)
        metadata_space = min_max_scaler.fit_transform(metadata_space)

        # Column stack them
        data_matrix = np.c_[feature_space, metadata_space]

        return images, meta, data_matrix

    @classmethod
    def prepare_data(cls, feature, k_latent_semantics, frt_technique,
                     ignore_metadata):
        # Get the images from the folders specified in config
        # We expect the vectors to be build for the features for both the labelled
        # and unlabelled data.
        u_images, u_meta, u_matrix = cls.get_data_matrix(
            feature, unlabelled=True, ignore_metadata=ignore_metadata)

        l_images, l_meta, l_matrix = cls.get_data_matrix(
            feature, ignore_metadata=ignore_metadata)

        # Reduce the labeled and unlabeled matrix together
        old_matrix = np.vstack((
            l_matrix,
            u_matrix,
        ))

        matrix, _, _ = reducer(old_matrix, k_latent_semantics, frt_technique)

        r_l_matrix = matrix[:len(l_images)]
        r_u_matrix = matrix[len(l_images):]

        return l_images, u_images, l_meta, u_meta, r_l_matrix, r_u_matrix


def ppr_driver(args, evaluate=False):
    l_images, u_images, l_meta, u_meta, l_matrix, u_matrix = PreparePPRData.prepare_data(
        args.model, args.k_latent_semantics, args.frt, args.ignore_metadata)

    # Build training data
    labelled = helper.build_matrix_with_labels(l_matrix, l_images, l_meta)

    # prepare test data
    query = helper.prepare_matrix_for_evaluation(u_matrix)

    # Evaluate
    predictions = ppr_classifier(labelled,
                                 query,
                                 frt=args.frt,
                                 k=args.k_latent_semantics,
                                 feature=args.model,
                                 edges=args.graph_edges,
                                 alpha=args.alpha,
                                 convergence=args.convergence)

    dorsal_symbol = 0.0
    palmar_symbol = 1.0

    if evaluate:
        truth = [
            dorsal_symbol
            if u_meta[image]['aspectOfHand'].split(' ')[0] == 'dorsal' else
            palmar_symbol for image in u_images
        ]

        print(helper.get_accuracy(truth, predictions))

    # Visualization pending
    return zip(u_images, predictions)


def decision_tree_driver(args, evaluate=False):
    images, data_matrix = utils.get_all_vectors('moment')
    # Fetch unlabelled data (as provided in the settings)
    u_images, u_meta, unlabelled = helper.get_unlabelled_data('moment')

    matrix, _, _,um = reducer(
        data_matrix,
        30,
        "pca",
        query_vector=unlabelled
    )

    l_matrix = matrix[:len(images)]
    u_matrix = um[:len(u_images)]

    dm = helper.build_labelled_matrix(l_matrix, images, 'aspectOfHand')

    # prepare test data
    query = helper.prepare_matrix_for_evaluation(u_matrix)

    max_depth = 15
    min_size = 30

    prediction = decision_tree(dm, query, max_depth, min_size)

    dorsal_symbol = 0.0
    palmar_symbol = 1.0

    if evaluate:
        truth = [
            dorsal_symbol
            if u_meta[image]['aspectOfHand'].split(' ')[0] == 'dorsal' else
            palmar_symbol for image in u_images
        ]
        print(helper.get_accuracy(truth, prediction))

    return zip(u_images, prediction)

def svm_driver(args, evaluate=False):
    model = settings.SVM.CLASSIFIER.MODEL
    k = settings.SVM.CLASSIFIER.K
    frt = settings.SVM.CLASSIFIER.FRT
    image_paths, pred = run_svm(evaluate, model, k, frt)
    return zip(image_paths, pred)

classifiers = {
    'ppr': ppr_driver,
    'decision': decision_tree_driver,
    'svm': svm_driver
}

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    predictions = classifiers[args.classifier](args, args.evaluate)
    output.write_to_file("task4.html",
                         "task4-{}.html".format(args.classifier),
                         predictions=[(
                             item[0],
                             "palmar" if item[1] == 1.0 else "dorsal",
                         ) for item in predictions],
                         title="TEST")
