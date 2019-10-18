import numpy as np
from sklearn.svm import OneClassSVM
from joblib import dump, load
from hyperopt import Trials, fmin, hp, tpe
from functools import partial
from tqdm import tqdm
from pathlib import Path
import sys
import os
from dynaconf import settings
from utils import get_all_vectors, filter_images
from feature_reduction.feature_reduction import reducer


def minimise_clf_loss(reduced_plabel_vectors, plabel_vectors, nlabel_vectors, frt_scaler, frt_model, frt, hyper_parameters):
    nu, gamma = hyper_parameters
    clf = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
    clf.fit(reduced_plabel_vectors)
    
    # Print negative accuracy
    correct_prediction = 0
    for each in nlabel_vectors:
        # For all vectors belonging to -ve label, scale, transform and predict 
        scaled_vector = frt_scaler.transform(each.reshape(1, -1))
        if frt in ['nmf', 'lda']:
            scaled_vector[scaled_vector < 0] = 0
        reduced_vector = frt_model.transform(scaled_vector)
        if clf.predict(reduced_vector)[0] == -1:
            correct_prediction += 1
    negative_rate = (correct_prediction/len(nlabel_vectors))
    
    # Print positive accuracy
    correct_prediction = 0
    for each in plabel_vectors:
        # For all vectors belonging to +ve label, scale, transform and predict
        scaled_vector = frt_scaler.transform(each.reshape(1, -1))
        if frt in ['nmf', 'lda']:
            scaled_vector[scaled_vector < 0] = 0
        reduced_vector = frt_model.transform(scaled_vector)
        if clf.predict(reduced_vector)[0] == 1:
            correct_prediction += 1
    positive_rate = (correct_prediction/len(plabel_vectors))
    
    # Ideal classifier will have this as minimum (0)
    return (1/(negative_rate + positive_rate)) - 0.5


def optimize_single_class_model(model, label, k, frt):

    print("running experiment for optimization")
    # Getting positive and nagative label vectors
    plabel_images_paths = filter_images(label)
    _, plabel_vectors = get_all_vectors(model, f={'path': {'$in': plabel_images_paths}})
    _, nlabel_vectors = get_all_vectors(model, f={'path': {'$nin': plabel_images_paths}})

    # Get the scaler and the model for the images belonging to that label
    reduced_plabel_vectors, _, _, frt_scaler, frt_model = reducer(
        plabel_vectors,
        k,
        frt,
        get_scaler_model=True
    )

    # Initialize a partial function for minimise_clf_loss() with pre-set variables
    tpe_trials = Trials()
    partial_func = partial(minimise_clf_loss, reduced_plabel_vectors, plabel_vectors, nlabel_vectors, frt_scaler, frt_model, frt)
    
    # set max number of trials and upper bound for gamma. LDA has its own specs because it's slow and we have to converge fast
    max_evals = settings.TASK5_CONFIG.MAX_EVALS
    gamma_max = sys.maxsize
    if frt == 'lda':
        gamma_max = settings.TASK5_CONFIG.LDA_GAMMA_MAX if settings.TASK5_CONFIG.LDA_GAMMA_MAX != 0 else sys.maxsize/2
        max_evals = settings.TASK5_CONFIG.LDA_MAX_EVALS if settings.TASK5_CONFIG.LDA_MAX_EVALS != 0 else settings.TASK5_CONFIG.MAX_EVALS
    if frt == 'nmf':
        gamma_max = settings.TASK5_CONFIG.NMF_GAMMA_MAX if settings.TASK5_CONFIG.NMF_GAMMA_MAX != 0 else sys.maxsize
        max_evals = settings.TASK5_CONFIG.NMF_MAX_EVALS if settings.TASK5_CONFIG.NMF_MAX_EVALS != 0 else settings.TASK5_CONFIG.MAX_EVALS
    search_space = (hp.uniform('nu', 0, 1,), hp.uniform('gamma', 0, gamma_max))

    # Running evals with the tpe algorithm
    tpe_best = fmin(fn=partial_func,
                    space=search_space,
                    algo=tpe.suggest,
                    trials=tpe_trials,
                    max_evals=max_evals,
                    show_progressbar=True)

    # Print and return the best params for the model
    optim_nu = tpe_best['nu']
    optim_gamma = tpe_best['gamma']
    print('optimal gamma is {}'.format(optim_gamma))
    print('optimal nu is {}'.format(optim_nu))
    return optim_nu, optim_gamma


def get_ideal_clf(model, label, k, frt):
    """returns a oneClassSVM with ideal nu and gamma. 
    
    Arguments:
        model {str} -- image model
        label {str} -- label to filter
        k {int} -- latent semantics
        frt {str} -- feature reduction technique
    
    Returns:
        sklearn.svm.OneClassSVM -- ideal classifier
    """
    clf = None
    clfs_directory = Path(settings.path_for(settings.TASK5_CONFIG.CLFS_DIRECTORY))

    #checks if classifier exists in clfs directory. 
    try:
        if not os.path.exists(clfs_directory):
            os.makedirs(clfs_directory)

        clf_file_name = '{}_{}_{}_{}.joblib'.format(model, frt, k, label)
        clf = load(clfs_directory / clf_file_name)
        print('loaded pre-computed classifier model')
    
    #create if doesn't exist
    except Exception as e:
        print('creating new classifer model')
        #call optimisation fun to run experiments and get ideal values for gamma and nu
        optim_nu, optim_gamma = optimize_single_class_model(model, label, k, frt)

        #create classifier with optimal params and save to clfs directory for future processing
        clf = OneClassSVM(nu=optim_nu, kernel="rbf", gamma=optim_gamma)
        clf_file_name = '{}_{}_{}_{}.joblib'.format(model, frt, k, label)
        dump(clf, clfs_directory /clf_file_name)

    return clf