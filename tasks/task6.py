import numpy as np
from pathlib import Path
from tabulate import tabulate
import argparse
from dynaconf import settings
import datetime
import sys
import os

sys.path.append('../')
import output
from metric.distance import distance
from feature_reduction.feature_reduction import reducer
from utils import get_term_weight_pairs, get_all_vectors, filter_images, get_metadata, get_subject_image_vectors, get_subject_attributes


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject_id', type=int, required=True)
    parser.add_argument('-d', '--data_path', type=str)
    return parser


if __name__ == "__main__":
    t_start = datetime.datetime.now()
    parser = prepare_parser()
    args = parser.parse_args()
    subject_data = {}
    model = "lbp"
    reduc_tech = "svd"
    '''
        # For every subject, we get the feature vector of each image.
        # This will give us a Data Matrix for each subject.
        # Then apply dimensionality reduction for each subject and get 
            the term weight pairs for each subject (the (K_latent semantic x Features) matrix).
        # Flatten this matrix to get a 1-D array for each subject.
        # Now compare the query subjectID (1-D array for which,
            will already be in the Subject-ReducedFeature matrix formed above) with 
            all other subjects.
    '''
    # This method gives us the feature vectors for each Subject.
    # Value of K will be equal to minimum number of images for the subjects.
    subject_data, k, images_per_subject = get_subject_image_vectors(model)
    subjects_reduced_dim = {}
    for subject in subject_data.keys():
        # Apply reducer to get (K_latent semantic x Features) matrix.
        vectors, eigen_values, weight = reducer(subject_data[subject], k - 1,
                                                reduc_tech)
        weight = weight.flatten()
        subjects_reduced_dim[subject] = weight

    result = []
    # Compare each subject with the query subject.
    for sub_id in subjects_reduced_dim.keys():
        delta = distance(subjects_reduced_dim[args.subject_id],
                         subjects_reduced_dim[sub_id], 2)
        result.append((sub_id, round(delta, 4), images_per_subject[sub_id]))

    result = sorted(result, key=lambda delta: delta[1])

    per_sub_data_visual = {}
    m = 3
    for i in range(0, m + 1):
        res = get_subject_attributes(result[i][0])
        sim_index = result[i][1]
        # ID no    # Attributes # Similarity # ListofImages
        per_sub_data_visual[result[i][0]] = (res[0], sim_index, result[i][2])

    for key in per_sub_data_visual.keys():
        print("{} -> {}".format(per_sub_data_visual[key][1], per_sub_data_visual[key][0]))

    output.write_to_file("visualize_task6.html",
                         "sub-task6-{}.html".format(args.subject_id),
                         data=per_sub_data_visual.values(),
                         title="TEST")
