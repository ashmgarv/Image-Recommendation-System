from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pymongo import MongoClient
from dynaconf import settings
import numpy as np
import sys
from .svm_train import SVM

sys.path.append('../')
from utils import get_all_vectors, filter_images
from feature_reduction.feature_reduction import reducer

#prepare labelled data
def build_labelled(model):
    # DORSAL TRAIN DATA
    dorsal_paths = filter_images('dorsal')
    _, dorsal_vectors = get_all_vectors(model, f={'path':{'$in': dorsal_paths}})
    dorsal_class = np.array([-1] * len(dorsal_vectors))

    #PALMAR TRAIN DATA
    _, palmar_vectors = get_all_vectors(model, f={'path': {'$nin': dorsal_paths}})
    palmar_class = np.array([1] * len(palmar_vectors))

    #TRAIN DATA STACKED AND CLASSES
    train_data = np.vstack((dorsal_vectors, palmar_vectors))
    train_class = np.concatenate((dorsal_class, palmar_class))

    return train_data, train_class

#generate test data
def build_unlabelled(model):
    #DORSAL TEST_DATA
    dorsal_paths = filter_images('dorsal', unlabelled_db=True)
    _, u_dorsal_vectors = get_all_vectors(model, f={'path': {'$in': dorsal_paths}}, unlabelled_db=True)
    dorsal_class = np.array([-1] * len(u_dorsal_vectors))
    
    #PALMAR TEST DATA
    palmar_paths = filter_images('palmar', unlabelled_db=True)
    _, u_palmar_vectors = get_all_vectors(model, f={'path': {'$in': palmar_paths}}, unlabelled_db=True)
    palmar_class = np.array([1] * len(u_palmar_vectors))

    #STACK ALL TEST DATA AND LABELS
    test_data = np.vstack((u_dorsal_vectors, u_palmar_vectors))
    test_labels = np.concatenate((dorsal_class,palmar_class))

    return test_data, test_labels, np.concatenate((dorsal_paths,palmar_paths))

def run_svm(evaluate, model='lbp', k=30, frt='pca'):
    train_data, train_labels = build_labelled(model)
    test_data, test_labels, test_paths = build_unlabelled(model)
    labelled_vectors, _, _, unlabelled_vectors  = reducer(train_data, k, frt, query_vector=test_data)
    labelled_vectors *= 2
    unlabelled_vectors *= 2
    svclassifier = SVM()
    svclassifier.fit(labelled_vectors, train_labels)
    y_pred = svclassifier.predict(unlabelled_vectors)
    if evaluate:
        print(model,k,frt)
        print(classification_report(test_labels,y_pred))
    return test_paths, y_pred