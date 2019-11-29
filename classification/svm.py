from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pymongo import MongoClient
from dynaconf import settings
import numpy as np
import sys
sys.path.append('../')
from utils import get_all_vectors
from feature_reduction.feature_reduction import reducer
from svm_train import SVM

def build_matrix(data_matrix, images, metadata_type, unlabelled_db):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    db = client[settings.QUERY_DATABASE if unlabelled_db else settings.DATABASE]
    coll = db[settings.IMAGES.METADATA_COLLECTION]
    data = []
    for image in images:
        dt = coll.find_one({'path':image},{metadata_type:1})
        if 'palmar' in dt[metadata_type]:
            data.append(1)
        else:
            data.append(-1)
    return data_matrix,data

# preparing labelled data for training.
images, data_matrix = get_all_vectors('lbp')
vectors, eigen_values, latent_vs_old = reducer(data_matrix, 30, "pca")
x,y = build_matrix(vectors, images, 'aspectOfHand', False)
# preparing unlabelled data for predicting.
images, data_matrix = get_all_vectors('lbp',{},True)
vectors, eigen_values, latent_vs_old = reducer(data_matrix, 30, "pca")
x_test,y_test = build_matrix(vectors, images, 'aspectOfHand', True)
# transforming the attributes to make them linearily seperable. 
x = np.array(x) ** 2
x_test = np.array(x_test) ** 2
# trainig the model and predicting the classes for testing set and outputting the accuracy results.
svclassifier =SVM()
svclassifier.fit(np.array(x), np.array(y))
y_pred = svclassifier.predict(x_test)
print(classification_report(y_test,y_pred))