import numpy as np

PEARSONS = 0
COSINE = 1
INTERSECTION = 2

"""
All these similarity functions here calculates the similarity. Duh.
It takes 2 1D vectors and calculates the similarity between them in the usual way and returns a number - the similarity.
It also takes a 2D matrix and a 1D vector. In this case, these functions will calculate the similarity between each row in the 2D matrix with the 1D vector and returns a vector of the similarities.
If you send in anything else, I don't know what will happen. So don't.
"""

def pearsons(vec1, vec2):
    if len(vec1.shape) == 2:
        return np.corrcoef(vec1, vec2)[:-1,-1]
    return np.corrcoef(vec1, vec2)[0,1]


def cosine(vec1, vec2):
    if len(vec1.shape) == 2:
        return np.apply_along_axis(np.dot, 1, vec1, vec2) / (np.apply_along_axis(np.linalg.norm, 1, vec1) * np.linalg.norm(vec2))
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))


def intersection(vec1, vec2):
    if len(vec1.shape) == 2:
        return np.minimum(vec1, vec2).sum(axis=1) / np.maximum(vec1, vec2).sum(axis=1)
    return np.minimum(vec1, vec2).sum() / float(np.maximum(vec1, vec2).sum())

opt = [
    pearsons,
    cosine,
    intersection
]

def similarity(vec1, vec2, t):
    return opt[t](vec1, vec2)

