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


# Perhaps this is buggy
# Now the problem is that if distance is 1, we cant possibly take 1/d as the
# similarity score.
def similarity(vec1, vec2, t):
    d = opt[t](vec1, vec2)
    # temp = np.where(1 - d < 0, d, np.absolute(1 - d))
    # res = np.reciprocal(temp, where=temp!=0, dtype=np.float16)

    # # Convert all 0 distances to similarity 1
    # d[np.abs(d) < np.finfo(np.float).eps] = 1.0
    # # Take reciprocal of distances. Similarity = 1/d
    # res = np.reciprocal(d, where=d!=0.0, dtype=np.float16)
    # # squish the bugs
    # if any(np.isnan(res)):
    #     import pdb
    #     pdb.set_trace()
    #     res = np.nan_to_num(res, copy=False, nan=1.0)
    # return res

    # https://stats.stackexchange.com/questions/158279/how-i-can-convert-distance-euclidean-to-similarity-score
    # http://www.uco.es/users/ma1fegan/Comunes/asignaturas/vision/Encyclopedia-of-distances-2009.pdf
    # https://stats.stackexchange.com/questions/53068/euclidean-distance-score-and-similarity
    return 1 / (1 + d)
