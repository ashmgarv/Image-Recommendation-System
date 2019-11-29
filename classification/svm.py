from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pymongo import MongoClient
from dynaconf import settings
import numpy as np
import sys
from .svm_train import SVM

sys.path.append('../')
from utils import get_all_vectors, filter_images
from feature_reduction.feature_reduction import reducer

def build_labelled(data_matrix, images, metadata_type):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    db = client[settings.DATABASE]
    coll = db[settings.IMAGES.METADATA_COLLECTION]
    data = []
    for image in images:
        dt = coll.find_one({'path':image},{metadata_type:1})
        if 'palmar' in dt[metadata_type]:
            data.append(1)
        else:
            data.append(-1)
    return data_matrix,data

def build_unlabelled():
    dorsal_paths = filter_images('dorsal', unlabelled_db=True)
    _, u_dorsal_vectors = get_all_vectors("lbp", f={'path': {'$in': dorsal_paths}}, unlabelled_db=True)
    dorsal_class = np.array([-1] * len(u_dorsal_vectors))
    palmar_paths = filter_images('palmar', unlabelled_db=True)
    _, u_palmar_vectors = get_all_vectors("lbp", f={'path': {'$in': palmar_paths}}, unlabelled_db=True)
    palmar_class = np.array([1] * len(u_palmar_vectors))
    vectors, eigen_values, latent_vs_old  = reducer(np.vstack((u_dorsal_vectors,u_palmar_vectors)), 30, "pca")
    test_labels = np.concatenate((dorsal_class,palmar_class))
    return vectors,test_labels, np.concatenate((dorsal_paths,palmar_paths))

def run_svm():
    # preparing labelled data for training.
    images, data_matrix = get_all_vectors('lbp')
    vectors, eigen_values, latent_vs_old = reducer(data_matrix, 30, "pca")
    x,y = build_labelled(vectors, images, 'aspectOfHand')
    # preparing unlabelled data for predicting.
    x_test,y_test, image_paths_ul = build_unlabelled()
    # transforming the attributes to make them linearily seperable. 
    x = np.array(x) ** 2
    x_test = np.array(x_test) ** 2
    # trainig the model and predicting the classes for testing set and outputting the accuracy results.
    svclassifier = SVM()
    svclassifier.fit(np.array(x), np.array(y))
    y_pred = svclassifier.predict(x_test)
    print(classification_report(y_test,y_pred))
    return image_paths_ul, y_pred