import numpy as np
from dynaconf import settings
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm
import multiprocessing as mp

import sys
sys.path.append('../')
from feature_reduction.feature_reduction import reducer

class Kmeans():
    def __init__(self, points, n_clusters, *kwargs):
        #config for feature reduction and clustering
        self.max_iter = settings.TASK2_CONFIG.MAX_ITER
        self.n_clusters = n_clusters

        self.centroids = None
        self.closest = None
        self.points = points


    def get_initial_centroid(self, points, c):
        centroids = points.copy()
        np.random.shuffle(centroids)
        return centroids[:c]


    def get_closest(self, points, centroids, return_min=False):
        c_extended = centroids[:, np.newaxis]
        distances = np.sqrt(((points - c_extended)**2).sum(axis=2))
        if not return_min:
            closest_centroids = np.argmin(distances, axis = 0)
            return closest_centroids
        else:
            return np.min(distances)


    def get_mean_centroids(self, points, centroids, closest):
        mean_centroids = []
        for k in range(centroids.shape[0]):
            centroid_points = points[closest == k]
            if centroid_points.size:
                mean_centroids.append(centroid_points.mean(axis=0))
        return np.array(mean_centroids)

    
    def get_final_centroids(self, points, c):
        centroids = self.get_initial_centroid(points, c)
        closest = self.get_closest(points, centroids)

        for _ in range(1000):
            closest = self.get_closest(points, centroids)
            new_centroids = self.get_mean_centroids(points, centroids, closest)
            if np.array_equal(centroids, new_centroids): 
                centroids = new_centroids.copy()
                break
            else:
                centroids = new_centroids.copy()
        return new_centroids, closest
    
    
    def get_cluster_scores(self, points):
        centroids, closest = self.get_final_centroids(points, self.n_clusters)
        return (centroids, closest, silhouette_score(points, closest, metric='euclidean'))

    
    def cluster(self):
        cluster_scores = []

        print("Running {} iterations of Kmeans".format(self.max_iter))
        pool = mp.Pool(processes=mp.cpu_count())
        for res in tqdm(pool.imap_unordered(self.get_cluster_scores, ([self.points] * self.max_iter)),total=self.max_iter):
            cluster_scores.append(res)
        cluster_scores.sort(key = lambda t: t[2])
        pool.close()

        print("Max cluster silhoutte score: {0}".format(cluster_scores[-1][2]))
        self.centroids = cluster_scores[-1][0]
        self.closest = cluster_scores[-1][1]
    




