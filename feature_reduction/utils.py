from feature.moment import get_all_vectors as get_all_moment_vectors

def get_all_vectors(model, f={}):
    """get all vectors present in the model
    
    Arguments:
        model {str} -- model name : sift, moment, hog, lbp
    
    Returns:
        image_labels, vectors -- array of image names and corresponding vectors (2 variables)
    """
    if model == 'moment':
        return get_all_moment_vectors(f=f)
    if model == 'sift':
        pass
    if model == 'hog':
        pass
    if model == 'lbp':
        pass

def get_term_weight_pairs(components):
    """returns array of weights of original features for each latent dimension
    
    Arguments:
        components {np.array} -- numpy array of size no_of_latent_semantics * no_of_original_features
    
    Returns:
        list -- list of feature weight pairs
    """
    term_weight_pairs = []
    for weights in components:
        feature_weights = [(index, weights[index]) for index in range(len(weights))]
        feature_weights.sort(key = lambda ele: ele[1], reverse=True)
        term_weight_pairs.append(feature_weights)
    return term_weight_pairs