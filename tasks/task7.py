import numpy as np
from pymongo import MongoClient
from multiprocessing import Pool
import argparse
from dynaconf import settings
from pprint import pprint
import sys
import os

sys.path.append('../')
import output
from feature_reduction.feature_reduction import reducer
from utils import get_metadata
from metric import distance

from sklearn.preprocessing import MinMaxScaler

mapping = {
    "male": 0,
    "female": 1,

    "fair": 0,
    "very fair": 1,
    "medium": 2,
    "dark": 3,

    "dorsal": 0,
    "palmar": 1,

    "right": 0,
    "left": 1
}


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k_latent_semantics', type=int, required=True)
    return parser


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    meta = get_metadata()

    try:
        subjects = { m['id']: [
            m['age'],
            mapping[m['gender']],
            mapping[m['skinColor']]
        ] for m in meta }
    except KeyError:
        raise Exception("Invalid metadata detected")

    subs = np.array([subjects[s] for s in subjects], dtype=float)

    # Scale the ages to better fit with the other binary values
    m = MinMaxScaler()
    subs[:,0] = m.fit_transform(subs[:,0].reshape(-1,1)).reshape(1,-1)

    sub_sub = np.array([distance.similarity(subs, s, distance.EUCLIDEAN) for s in subs])

    vectors, eigen_values, latent_vs_old = reducer(
        sub_sub, args.k_latent_semantics, "nmf")

    pprint(vectors.tolist(), indent=4)

