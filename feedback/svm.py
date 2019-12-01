from pymongo import MongoClient
from dynaconf import settings
import numpy as np
import sys

sys.path.append('../')
from utils import get_all_vectors, filter_images
from feature_reduction.feature_reduction import reducer
from classification.svm_train import SVM

def svm_feedback(relevant_images, irrelevant_images, images_to_display):
    model = settings.SVM.CLASSIFIER.MODEL
    k = settings.SVM.CLASSIFIER.K
    frt = settings.SVM.CLASSIFIER.FRT
    images_rel, data_matrix_rel = get_all_vectors(model,f={'path': {'$in': relevant_images}},master_db=True)
    images_irel, data_matrix_irel = get_all_vectors(model,f={'path': {'$in': irrelevant_images}},master_db=True)
    images_test, test_vector = get_all_vectors(model,f={},master_db=True)
    labelled_vectors, _, _, unlabelled_vectors  = reducer(np.vstack((data_matrix_rel,data_matrix_irel)), k, frt, query_vector=test_vector)
    rel_class = np.array([1] * len(data_matrix_rel))
    irel_class = np.array([-1] * len(data_matrix_irel))
    x_train = labelled_vectors
    x_train = np.array(x_train) * 2
    y_train = np.concatenate((rel_class,irel_class))
    svclassifier = SVM()
    svclassifier.fit(np.array(x_train), np.array(y_train))
    unlabelled_vectors = np.array(unlabelled_vectors) * 2
    y_pred = svclassifier.predict(unlabelled_vectors)
    c=0
    dic = {}
    for y in y_pred:
        if y==1:
            dic[images_test[c]] = unlabelled_vectors[c]
        c+=1
    length_dict = {}
    for key in dic.keys():
        length_dict[key]=np.dot(dic[key],svclassifier.w)
    sorted_dict = sorted(length_dict.items(), key=lambda x : x[1],reverse=True)
    list_img = []
    c=0
    for key,j in sorted_dict:
        if c<images_to_display:
            list_img.append(key)
            c+=1
        else:
            break
    return(list_img)