from random import randrange
import numpy as np
from dynaconf import settings
from pymongo import MongoClient

import sys
sys.path.append('../')
import utils


#Split a dataset into k small batches
def cv_split(dataset, n_batches):
    splits = list()
    copy = list(dataset)
    print (dataset.shape)
    batch_size = int(len(dataset) / n_batches)
    for i in range(n_batches):
        batch = list()
        while len(batch) < batch_size:
            index = randrange(len(copy))
            batch.append(copy.pop(index))
        splits.append(batch)
    return splits

#Evaluate any machine learning algorithm using a cross validation split
#Use args to pass arguments to your algorithm
def evaluate_algorithm(dataset, algorithm, n_batches, *args):
    batches = cv_split(dataset, n_batches)
    accuracies = list()
    for batch in batches:
        train = list(batches)
        remove_test_data(train, batch)
        train = sum(train, [])
        test = list()
        for row in batch:
            row_copy = list(row)
            test.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train, test, *args)
        
        #Prepare the ground truth array
        ground_truth = [row[-1] for row in batch]
        accuracy = get_accuracy(ground_truth, predicted)
        accuracies.append(accuracy)
    return accuracies

#Refferred from : https://stackoverflow.com/questions/3157374/how-do-you-remove-a-numpy-array-from-a-list-of-numpy-arrays
#Remove the test data from original data
def remove_test_data(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

#Calculate the accuracy
def get_accuracy(ground_truth, predicted):
    right_prediction = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] == predicted[i]:
            right_prediction += 1
    return right_prediction / float(len(ground_truth)) * 100.0


def prepare_matrix_for_evaluation(data_matrix):
    data_matrix = np.array(data_matrix, dtype=object)
    data_matrix = np.c_[data_matrix, [None] * len(data_matrix)]
    return data_matrix


def get_labelled_data(feature):
    # Get labelled images
    l_images, feature_space = utils.get_all_vectors(feature)

    # Get metadata
    meta = utils.get_metadata()
    meta = {m['path']: m for m in meta}

    return l_images, meta, feature_space


def get_unlabelled_data(feature):
    u_images, u_vectors = utils.get_all_vectors(feature)

    # Get metadata
    meta = utils.get_metadata()
    meta = {m['path']: m for m in meta}

    return u_images, meta, u_vectors


def build_matrix_with_labels(data_matrix, images, metadata):
    data = np.zeros(len(data_matrix))
    for idx, image in enumerate(images):
        if 'palmar' in metadata[image]['aspectOfHand']:
            data[idx] = 1.0
        else:
            data[idx] = 0.0

    data_matrix = np.c_[data_matrix, data]
    return data_matrix


def build_labelled_matrix(data_matrix, images, metadata_type):
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
            data.append(1.0)
        else:
            data.append(0.0)
    data = np.array(data).reshape(-1,1)
    data_matrix = np.append(data_matrix, data, axis=1)
    return data_matrix