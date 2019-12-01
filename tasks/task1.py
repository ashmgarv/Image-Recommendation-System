import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


import sys
sys.path.append('../')
from utils import get_all_vectors, filter_images
from feature_reduction.feature_reduction import reducer
from metric.distance import distance
from sklearn.neighbors.nearest_centroid import NearestCentroid
from numpy import dot
from numpy.linalg import norm


def generate_vec():
    #getting dorsal vectors and class
    dorsal_paths = filter_images('dorsal')
    _, dorsal_vectors = get_all_vectors(model, f={'path': {'$in': dorsal_paths}})
    
    #getting palmar vectors and class
    palmar_paths = filter_images('palmar')
    _, palmar_vectors = get_all_vectors(model, f={'path': {'$in': palmar_paths}})
    
    #getting dorsal vectors and class
    dorsal_paths = filter_images('dorsal', unlabelled_db=True)
    _, u_dorsal_vectors = get_all_vectors(model, f={'path': {'$in': dorsal_paths}}, unlabelled_db=True)
    dorsal_class = np.array([1] * len(u_dorsal_vectors))
    
    #getting palmar vectors and class
    palmar_paths = filter_images('palmar', unlabelled_db=True)
    _, u_palmar_vectors = get_all_vectors(model, f={'path': {'$in': palmar_paths}}, unlabelled_db=True)
    palmar_class = np.array([0] * len(u_palmar_vectors))

    test_data  = np.vstack((u_dorsal_vectors,u_palmar_vectors))
    test_labels = np.concatenate((dorsal_class,palmar_class))
 
    return dorsal_vectors, palmar_vectors, test_data, test_labels

def mahalano(x, data):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    #x_minus_mu = x - np.mean(data)
    cov = np.cov(data.T)
    inv_covmat = np.linalg.inv(cov)
    #left_term = np.dot(x_minus_mu, inv_covmat)
    #mahal = np.dot(left_term, x_minus_mu.T)
    mahal = mahalanobis(x,np.mean(data, axis=0),inv_covmat)
    return mahal


model_list = ['moment','hog','sift','moment_inv']

k_list = [30]
results = []
feature_list = ['pca','nmf','svd']


#test across model, k,
for feature_each in feature_list:
    for model_each in model_list:
        for k_each in k_list:

            print("Running ",feature_each,model_each,k_each)
            model = model_each
            k = k_each
            dorsal_vectors, palmar_vectors, test_data, test_labels = generate_vec()
            
            print(dorsal_vectors.shape)

            reduced_dorsal_vectors, _, _, _, dorsal_pca = reducer(dorsal_vectors,k_each,feature_each,get_scaler_model=True)
            if (feature_each == "pca"):
                dorsal_variance_ratio = dorsal_pca.explained_variance_ratio_
            reduced_palmar_vectors, _, _, _, palmar_pca = reducer(palmar_vectors,k_each,feature_each,get_scaler_model=True)
            if (feature_each == "pca"):
                palmar_variance_ratio = palmar_pca.explained_variance_ratio_
            reduced_test_data, _, _, _, test_pca = reducer(test_data,k_each,feature_each,get_scaler_model=True)
            if (feature_each == "pca"):
                test_variance_ratio = test_pca.explained_variance_ratio_          
            
            """
            reduced_dorsal_vectors, _, _,dorsal_variance_ratio = reducer(dorsal_vectors,k_each,feature_each)
            reduced_palmar_vectors, _, _, palmar_variance_ratio = reducer(palmar_vectors,k_each,feature_each)
            reduced_test_data, _, _, test_variance_ratio = reducer(test_data,k_each,feature_each)
            """

            dorsal = []
            palmar = []

            for row in reduced_test_data:
                dorsalI = 0
                if (feature_each == "pca"):
                    row = row * test_variance_ratio
                for row1 in reduced_dorsal_vectors:
                    if (feature_each == "pca"):
                        row1 = row1 * dorsal_variance_ratio
                    dorsalI = dorsalI + dot(row, row1)/(norm(row)*norm(row1))
                dorsal.append(dorsalI)

            for row2 in reduced_test_data:
                if (feature_each == "pca"):
                    row2 = row2 * test_variance_ratio
                palmarI = 0
                for row3 in reduced_palmar_vectors:
                    if (feature_each == "pca"):
                        row3 = row3 * palmar_variance_ratio
                    palmarI = palmarI + dot(row2, row3)/(norm(row2)*norm(row3))
                palmar.append(palmarI)

            p_label = []
            j=0
            for i in range(len(reduced_test_data)):
                p_label.append(0) if palmar[i] < dorsal[i] else p_label.append(1)
                if p_label[i]==test_labels[i]:
                    j = j + 1
            print((j/len(reduced_test_data))*100) 
            