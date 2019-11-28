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


"""
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
"""
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

"""
def test(dorsal_vectors, dorsal_class, palmar_vectors, palmar_class):
    # Vectors for palmar and dorsal split into test and train
    vectors =  np.vstack((palmar_vectors, dorsal_vectors))
    labels = np.concatenate((palmar_class, dorsal_class))
    train_data, test_data, train_labels, test_labels = train_test_split(vectors, labels)
    
    #train dorsal and palmar data
    dorsal_train_data = train_data[np.where(train_labels == 1)[0]]
    palmar_train_data = train_data[np.where(train_labels == 0)[0]]
    
    #get dorsal centroids
    dorsal_centroids, _ = get_final_centroids(dorsal_train_data, 2)
    palmar_centroids, _ = get_final_centroids(palmar_train_data, 2)
    
    print("Function Call")
    print(dorsal_centroids,"\n",palmar_centroids)

    #predict label and accuracy
    pred_labels = []
    for each in test_data:
        dorsal_dist = get_closest(each.reshape(1,-1), dorsal_centroids, return_min=True)
        palmar_dist = get_closest(each.reshape(1,-1), palmar_centroids, return_min=True)
        p_label = 1 if dorsal_dist < palmar_dist else 0
        pred_labels.append(p_label)
    
    return accuracy_score(pred_labels, test_labels)
"""

model_list = ['sift','hog','moment']
#model_list = ['hog']
k_list = [10,15,20,25,30,35,40]
results = []

#test across model, k,
for model_each in model_list:
        for k_each in k_list:
            
            print("Running ", model_each,k_each)
            model = model_each
            k = k_each
            dorsal_vectors, palmar_vectors, test_data, test_labels = generate_vec()
            
            reduced_dorsal_vectors, _, _ = reducer(dorsal_vectors,k_each,"nmf")
            reduced_palmar_vectors, _, _ = reducer(palmar_vectors,k_each,"nmf")
            reduced_test_data, _, _ = reducer(test_data,k_each,"nmf")
            
            palmar = []
            dorsal = []

            for row in reduced_test_data:
                dorsal.append(mahalano(row,reduced_dorsal_vectors))
                palmar.append(mahalano(row,reduced_palmar_vectors))
            
            dorsal = np.asarray(dorsal)
            palmar = np.asarray(palmar)

            p_label = []
            j=0

            for i in range(len(dorsal)):
                p_label.append(0) if palmar[i] < dorsal[i] else p_label.append(1)
                if p_label[i]==test_labels[i]:
                    j = j + 1
            print(j/2) 
            """
            i=0
            
            for row in reduced_dorsal_vectors:
                dorsal.append(np.mean(row)) 
                i=i+1    
            print ('Dorsal: ', sum(dorsal)/len(dorsal))      

            dorsal_value = sum(dorsal)/len(dorsal)
            for row in reduced_palmar_vectors:
                palmar.append(np.mean(row)) 
                i=i+1    
            print ('Palmar: ', sum(palmar)/len(palmar))   
            
            palmar_value = sum(palmar)/len(palmar)
            
            i = 0
            j = 0
            #dorsal_dist = abs(np.mean(row)-dorsal_value)
            #palmar_dist = abs(np.mean(row)-palmar_value)
            
            #DORSAL MD
            dorsal_covariance_matrix = np.cov(reduced_dorsal_vectors, rowvar=False)  
            dorsal_inv_covariance_matrix = np.linalg.inv(dorsal_covariance_matrix)
            dorsal_vars_mean = []
            for i in range(reduced_dorsal_vectors.shape[0]):
                dorsal_vars_mean.append(list(reduced_dorsal_vectors.mean(axis=0))) 
            dorsal_diff = reduced_dorsal_vectors - dorsal_vars_mean
            dorsal_md = []
            for i in range(len(dorsal_diff)):
                dorsal_md.append(np.sqrt(dorsal_diff[i].dot(dorsal_inv_covariance_matrix).dot(dorsal_diff[i]))) 
            
            dorsal_md = np.asarray(dorsal_md)
            
            #PALMAR MD
            palmar_covariance_matrix = np.cov(reduced_palmar_vectors, rowvar=False)  
            palmar_inv_covariance_matrix = np.linalg.inv(palmar_covariance_matrix)
            palmar_vars_mean = []
            for i in range(reduced_palmar_vectors.shape[0]):
                palmar_vars_mean.append(list(reduced_palmar_vectors.mean(axis=0))) 
            palmar_diff = reduced_palmar_vectors - palmar_vars_mean
            palmar_md = []
            for i in range(len(palmar_diff)):
                palmar_md.append(np.sqrt(palmar_diff[i].dot(palmar_inv_covariance_matrix).dot(palmar_diff[i]))) 
            
            palmar_md = np.asarray(palmar_md)
            
            p_label = []
            j=0
            for i in range(len(palmar_md)):
                p_label.append(1) if palmar_md[i] < dorsal_md[i] else p_label.append(0)
                if p_label[i]==test_labels[i]:
                    j = j + 1
            print(j/2)
                
            for row1 in reduced dorsal vectors:
                for row in reduced_test_data:
                    dorsal_dist = mahalanobis(row, row1, np.linalg.inv(np.cov(palmar_value.T)))

            for row1 in reduced palmar vectors:
                for row in reduced_test_data:
                    palmar_dist = mahalanobis(row, row1, np.linalg.inv(np.cov(palmar_value.T)))
            

                    #dorsal_dist = mahalanobis(dorsal, np.mean(row),np.linalg.inv(np.cov(dorsal_value.T)))
                
                p_label.append(1) if dorsal_dist < palmar_dist else p_label.append(0)
                if p_label[i]==test_labels[i]:
                    j = j + 1
                i = i + 1
            print(j/2)
                
            
            #Fetch test images and apply feature extraction 
            #Apply dimensionality reduction to the above features.
            #Find mean of each image and calculate distance from each and assign values


            scores = []
            for _ in range(100):
                scores.append(test(
                    reduced_dorsal_vectors, dorsal_class, reduced_palmar_vectors, palmar_class
                ))
            
            res = {
                'model': model,
                'k': k_each,
                'score': np.mean(scores)
            }
            results.append(res)
            
results_df = pd.DataFrame(results)
results_df.sort_values(['score'])
print(results_df)
"""