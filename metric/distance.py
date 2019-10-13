import numpy as np

EUCLIDEAN = 0
MANHATTAN = 1

"""
All these distance functions here calculates the distance. Duh.
It takes 2 1D vectors and calculates the distance between them in the usual way and returns a number - the distance.
It also takes a 2D matrix and a 1D vector. In this case, these functions will calculate the distance between each row in the 2D matrix with the 1D vector and returns a vector of the distances.

If you send in anything else, I don't know what will happen. So don't.

This module also converts these distances to similarity, if situation calls for it, by taking the inverse of the distances.
"""

def euclidean(vec1, vec2):
    if len(vec1.shape) == 2:
        return np.sqrt(np.sum(np.power(vec1 - vec2, 2), axis=1))
    return np.sqrt(np.sum(np.power(vec1 - vec2, 2)))


def manhattan(vec1, vec2):
    if len(vec1.shape) == 2:
        return np.sum(vec1 - vec2, axis=1)
    return np.sum(vec1 - vec2)


opt = [
    euclidean,
    manhattan,
]

def distance(vec1, vec2, t):
    return opt[t](vec1, vec2)

def similarity(vec1, vec2, t):
    d = opt[t](vec1, vec2)
    temp = np.where(1 - d < 0, d, np.absolute(1 - d))
    return np.reciprocal(temp, where=temp!=0, dtype=np.float16)
