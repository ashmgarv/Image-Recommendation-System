from pymongo import MongoClient
from dynaconf import settings
import numpy as np
import math
import sys

sys.path.append('../')
from utils import get_all_vectors, filter_images
from feature_reduction.feature_reduction import reducer

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def makeArrayBinary(arr, length, breadth):
    for i in range(length):
        for j in range(breadth):
            if arr[i][j] >= 0.5:
                arr[i][j] = 1
            else:
                arr[i][j] = 0
    return arr

#query_img = 'C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000002.jpg'
#relevant = ['C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000002.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000003.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000007.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000008.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000014.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000015.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000013.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0001527.jpg']
#irrelevant = ['C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000080.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000393.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000387.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000426.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000389.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0000374.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0001282.jpg','C:\\Users\\ankit\\Documents\\GitHub\\Data\\Hands\\Hand_0001870.jpg']

model = 'hog'
frt = 'svd'
k = 20

def feedback_probab(relevant, irrelevant, t):
    print(len(relevant))
    img_all, img_all_vec = get_all_vectors(model,f={},master_db=True)
    #f={'path': {'$nin': relevant}}

    img_all_vec_red, _, __ = reducer(img_all_vec, k, frt)
    img_all_vec_red = scale(img_all_vec_red, 0, 1)

    img_rel_vec_red=[]
    dict_all_red = {}
    for i in range(len(img_all)):
        name = img_all[i]
        dict_all_red[name] = img_all_vec_red[i]
        if name in relevant:
            img_rel_vec_red.append(img_all_vec_red[i])
    img_rel_vec_red = np.array(img_rel_vec_red)

    img_all_vec_red = makeArrayBinary(img_all_vec_red, img_all_vec_red.shape[0], img_all_vec_red.shape[1])
    img_rel_vec_red = makeArrayBinary(img_rel_vec_red, img_rel_vec_red.shape[0], img_rel_vec_red.shape[1])
    print(img_rel_vec_red.shape[0])
    R = len(relevant)
    N = len(img_all)

    p_list = []
    for j in range(k):
        r = 0
        for i in range(img_rel_vec_red.shape[0]):
            if img_rel_vec_red[i][j] == 1:
                r+=1
        p_list.append((r+0.5)/(R+1))

    n_list = []
    for j in range(k):
        n = 0
        for i in range(N):
            if img_all_vec_red[i][j] == 1:
                n+=1
        n_list.append(n)

    for i in range(k):
        n_list[i] = (n_list[i] - p_list[i] + 0.5)/(N-R+1)

    log_list = []
    for i in range(k):
        num = (p_list[i]*(1-n_list[i]))/(n_list[i]*(1-p_list[i]))
        if num>=0:
            log_list.append(math.log(num,2))
    log_list = np.array(log_list)

    new_result = []
    for name in dict_all_red.keys():
        sim = np.dot(dict_all_red[name], log_list)
        new_result.append((name, sim))

    new_result = sorted(new_result, key=lambda x:x[1], reverse=True)

    final = []
    for i in range(t):
        final.append(new_result[i][0])
    
    print(final)
    return final

