import numpy as np
import pandas as pd
import math
from dynaconf import settings

import argparse
import sys
sys.path.append('../')
from utils import get_all_vectors, filter_images
from feature_reduction.feature_reduction import reducer
from numpy import dot
from numpy.linalg import norm
from output import write_to_file

#Implement code for Cosine Similarity
def cos_similarity(v1, v2):
    v1mag = 0
    v2mag = 0
    dot_product = 0
    for indices in v1:
        v1mag = v1mag + (indices*indices)
    v1mag = math.sqrt(v1mag)
    for indices2 in v2:
        v2mag = v2mag + (indices2*indices2)
    v2mag = math.sqrt(v2mag)
    denominator =  v2mag*v1mag
    
    #calculate dot product
    for i in range(len(v1)):
        dot_product += v1[i] * v2[i]
    finalcossimilarity = dot_product/denominator
    return finalcossimilarity


# Initiate the argument parses
def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--Features', type=int, required=True)
    return parser

# Generating the vectors
def generate_vec():

    #getting dorsal vectors and class
    dorsal_paths = filter_images('dorsal')
    _, dorsal_vectors = get_all_vectors(model, f={'path': {'$in': dorsal_paths}})
    
    #getting palmar vectors and class
    palmar_paths = filter_images('palmar')
    _, palmar_vectors = get_all_vectors(model, f={'path': {'$in': palmar_paths}})
    
    #getting dorsal vectors and class
    test_data_paths, test_data = get_all_vectors(model, f={}, unlabelled_db=True)
    
    # Return the calculated values
    return dorsal_vectors, palmar_vectors, test_data, test_data_paths


if __name__ == "__main__":
    
    # Initiate cide to take inputs
    parser = prepare_parser()
    args = parser.parse_args()
    
    # Only 1 input to be taken, that is k (latent semantics), number of features to be extracted 
    k_each = args.Features
            
    # On extensive testing, the best feature extraction model was found out to be SIFT
    model = settings.TASK1_CONFIG.MODEL

    # On extensive testing, the best feature reduction technique was founf out to be PCA
    feature = settings.TASK1_CONFIG.FRT

    # Generating the vectors for Dorsal Labelled, Palmar labelled and the test vectors
    # Also fetching the labels of unlabelled images so as to check accuracy later
    dorsal_vectors, palmar_vectors, test_data, test_data_paths = generate_vec()

    # Applying PCA to Dorsal Images and fetching the 'k' latent semantics 
    reduced_dorsal_vectors, _, _, _, dorsal_pca = reducer(dorsal_vectors,k_each,feature,get_scaler_model=True)
    dorsal_variance_ratio = dorsal_pca.explained_variance_ratio_
    print("Computed ",k_each," Latent Semantics for Dorsal")

    # Applying PCA to Palmar Images and fetching the 'k' latent semantics 
    reduced_palmar_vectors, _, _, _, palmar_pca = reducer(palmar_vectors,k_each,feature,get_scaler_model=True)
    palmar_variance_ratio = palmar_pca.explained_variance_ratio_
    print("Computed ",k_each," Latent Semantics for Palmar")
    
    # Applying PCA to Test Images and fetching the 'k' latent semantics 
    reduced_test_data, _, _, _, test_pca = reducer(test_data,k_each,feature,get_scaler_model=True)
    test_variance_ratio = test_pca.explained_variance_ratio_

    # Initiate List that will store the total dorsal dot product scores for each test image   
    dorsal = []

    # Initiate List that will store the total palmar dot product scores for each test image   
    palmar = []

    # Calculate Cosine similarity with every test image with every dorsal image and store them in 'dorsal' list
    for row in reduced_test_data:
        dorsalI = 0
        row = row * test_variance_ratio
        for row1 in reduced_dorsal_vectors:
            row1 = row1 * dorsal_variance_ratio
            dorsalI = dorsalI + cos_similarity(row,row1)
            #dot(row, row1)/(norm(row)*norm(row1))
        dorsal.append(dorsalI)

    # Calculate Cosine similarity with every test image with every palmar image and store them in 'palmar' list
    for row2 in reduced_test_data:
        row2 = row2 * test_variance_ratio
        palmarI = 0
        for row3 in reduced_palmar_vectors:
            row3 = row3 * palmar_variance_ratio
            palmarI = palmarI + cos_similarity(row2,row3)
            #dot(row2, row3)/(norm(row2)*norm(row3))
        palmar.append(palmarI)

    # Calculation of accuracy scores
    p_label = []
    j=0 
    for i in range(len(reduced_test_data)):
        p_label.append(0) if palmar[i] < dorsal[i] else p_label.append(1)

    # Change 1 to Dorsal and 0 to Palmar for Visualization purpose
    final_list = []
    final_list = ["Dorsal" if i == 1 else "Palmar" for i in p_label]
    
    # Code to visualise the output. Check browser or output folder.
    write_to_file("task4.html",
                    "task1-{}.html".format(k_each),
                    predictions=zip(test_data_paths,final_list),
                    title="Task1")
    
            