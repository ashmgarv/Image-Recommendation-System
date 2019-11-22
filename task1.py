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

def get_initial_centroid(points, k):
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def get_closest(points, centroids, return_min=False):
    c_extended = centroids[:, np.newaxis]
    distances = np.sqrt(((points - c_extended)**2).sum(axis=2))
    
    if not return_min:
        closest_centroids = np.argmin(distances, axis = 0)
        return closest_centroids
    else:
        return np.min(distances)

def get_mean_centroids(points, centroids, closest):
    return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])

def get_final_centroids(points, c):
    centroids = get_initial_centroid(points, c)
    closest = get_closest(points, centroids)

    for _ in range(1000):
        closest = get_closest(points, centroids)
        new_centroids = get_mean_centroids(points, centroids, closest)
        if np.array_equal(centroids, new_centroids): 
#             print('converged at ', _+1)
            centroids = new_centroids.copy()
            break
        else:
            centroids = new_centroids.copy()
    
    return new_centroids, closest

def generate_vec():
    #getting dorsal vectors and class
    dorsal_paths = filter_images('dorsal')
    _, dorsal_vectors = get_all_vectors(model, f={'path': {'$in': dorsal_paths}})
    dorsal_class = np.array([1] * len(dorsal_vectors))
    print (dorsal_vectors.shape)
    #getting palmar vectors and class
    palmar_paths = filter_images('palmar')
    _, palmar_vectors = get_all_vectors(model, f={'path': {'$nin': palmar_paths}})
    palmar_class = np.array([0] * len(palmar_vectors))
    
    return dorsal_vectors, dorsal_class, palmar_vectors, palmar_class

def test(dorsal_vectors, dorsal_class, palmar_vectors, palmar_class):
    # Vectors for palmar and dorsal split into test and train
    vectors =  np.vstack((palmar_vectors, dorsal_vectors))
    labels = np.concatenate((palmar_class, dorsal_class))
    train_data, test_data, train_labels, test_labels = train_test_split(vectors, labels)
    
    #train dorsal and palmar data
    dorsal_train_data = train_data[np.where(train_labels == 1)[0]]
    palmar_train_data = train_data[np.where(train_labels == 0)[0]]
    
    #get dorsal centroids
    dorsal_centroids, _ = get_final_centroids(dorsal_train_data, c)
    palmar_centroids, _ = get_final_centroids(palmar_train_data, c)
    #predict label and accuracy
    pred_labels = []
    for each in test_data:
        dorsal_dist = get_closest(each.reshape(1,-1), dorsal_centroids, return_min=True)
        palmar_dist = get_closest(each.reshape(1,-1), palmar_centroids, return_min=True)
        p_label = 1 if dorsal_dist < palmar_dist else 0
        pred_labels.append(p_label)
    
    return accuracy_score(pred_labels, test_labels)

model_list = ['sift','moment']
k_list = [10,20,30]
c_list = [10,20,30]
results = []

#test across model, k, c:
for model_each in model_list:
    for c_each in c_list:
        for k_each in k_list:
            
            print("Running ", model_each,c_each,k_each)
            model = model_each
            c = c_each
            k = k_each
            dorsal_vectors, dorsal_class, palmar_vectors, palmar_class = generate_vec()
            
            scores = []
            for _ in range(100):
                scores.append(test(
                    dorsal_vectors, dorsal_class, palmar_vectors, palmar_class
                ))
            
            res = {
                'model': model,
                'c': c,
                'k': k,
                'score': np.mean(scores)
            }
            results.append(res)

results_df = pd.DataFrame(results)
results_df.sort_values(['score'])
print(results_df)