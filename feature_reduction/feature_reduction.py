import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF, TruncatedSVD


def get_pca(vectors, k, **opts):
    """scales and applies PCA to vectors and opts['query_vector'] if present
    
    Arguments:
        vectors {np.array} -- numpy matrix of vectors
        k {int} -- k latent semantics
    
    Returns:
        PCA decomposition and reduced query vector if present
    """
    std_scaler = StandardScaler()
    scaled_values = std_scaler.fit_transform(vectors)

    k = min(k, vectors.shape[0], vectors.shape[1])
    pca = PCA(n_components=k)
    pca_vectors = pca.fit_transform(scaled_values)

    #if opts contains query vector, apply scaler and PCA transformation to the query vector and return
    if opts:
        if 'get_scaler_model' in opts and opts['get_scaler_model']:
            print("returning just the scaler and frt model")
            return pca_vectors, pca.explained_variance_, pca.components_, std_scaler, pca

        query_vector = opts['query_vector']
        scaled_query_vector = std_scaler.transform(query_vector)
        pca_query_vector = pca.transform(scaled_query_vector)
        return pca_vectors, pca.explained_variance_, pca.components_, pca_query_vector
    else:
        return pca_vectors, pca.explained_variance_, pca.components_


def get_svd(vectors, k, **opts):
    """scales and applied SVD to vectors and opts['query_vector] if present
    
    Arguments:
        vectors {np.array} -- vector matrix
        k {int} -- latent semantics
    
    Returns:
        SVD decomposition and reduced query vector if present
    """
    std_scaler = StandardScaler()
    scaled_values = std_scaler.fit_transform(vectors)

    k = min(k, vectors.shape[0], vectors.shape[1])
    svd = TruncatedSVD(n_components=k)
    svd_vectors = svd.fit_transform(scaled_values)

    #if opts contains query vector, apply scaler and SVD transformation to the query vector and return
    if opts:
        if 'get_scaler_model' in opts and opts['get_scaler_model']:
            print("returning just the scaler and frt model")
            return svd_vectors, svd.explained_variance_, svd.components_, std_scaler, svd

        query_vector = opts['query_vector']
        scaled_query_vector = std_scaler.transform(query_vector)
        svd_query_vector = svd.transform(scaled_query_vector)
        return svd_vectors, svd.explained_variance_, svd.components_, svd_query_vector
    else:
        return svd_vectors, svd.explained_variance_, svd.components_


def get_lda(vectors, k, **opts):
    """scales and applies LDA to vectors and opts['query_vectors'] if present
    
    Arguments:
        vectors {np.array} -- vector matrix
        k {int} -- latent semantic
    
    Returns:
        LDA decomposition and reduced query vector if present
    """
    min_max_scaler = MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(vectors)

    k = min(k, vectors.shape[0], vectors.shape[1])
    lda = LatentDirichletAllocation(n_components=k, verbose=0,learning_method='online',n_jobs=-1)
    lda_vectors = lda.fit_transform(scaled_values)

    #if opts contains query vector, apply scaler and PCA transformation to the query vector and return
    if opts:
        if 'get_scaler_model' in opts and opts['get_scaler_model']:
            print("returning just the scaler and frt model")
            return lda_vectors, None, lda.components_, min_max_scaler, lda

        query_vector = opts['query_vector']
        scaled_query_vector = min_max_scaler.transform(query_vector)
        if(np.min(scaled_query_vector)) < 0:
            print('negative found after scaling query vector. setting to 0')
            scaled_query_vector[scaled_query_vector < 0] = 0
        lda_query_vector = lda.transform(scaled_query_vector)
        return lda_vectors, None, lda.components_, lda_query_vector
    else:
        return lda_vectors, None, lda.components_

def get_nmf(vectors, k, **opts):
    """scales and applies NMF to vectors and opts['query_vector'] if present
    
    Arguments:
        vectors {np.array} -- vector matrix
        k {int} -- latent semantic
    
    Returns:
        NMF decomposition and reduced NMF vector if present
    """
    min_max_scaler = MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(vectors)

    k = min(k, vectors.shape[0], vectors.shape[1])
    nmf = NMF(n_components=k, init='random', random_state=0)
    nmf_vectors = nmf.fit_transform(scaled_values)

    #if opts contains query vector, apply scaler and PCA transformation to the query vector and return
    if opts:
        if 'get_scaler_model' in opts and opts['get_scaler_model']:
            print("returning just the scaler and frt model")
            return nmf_vectors, None, nmf.components_, min_max_scaler, nmf

        query_vector = opts['query_vector']
        scaled_query_vector = min_max_scaler.transform(query_vector)
        if(np.min(scaled_query_vector)) < 0:
            print('negative found after scaling query vector. setting to 0')
            scaled_query_vector[scaled_query_vector < 0] = 0
        nmf_query_vector = nmf.transform(scaled_query_vector)
        return nmf_vectors, None, nmf.components_, nmf_query_vector
    else:
        return nmf_vectors, None, nmf.components_

reducer_type = {
    "lda": get_lda,
    "pca": get_pca,
    "svd": get_svd,
    "nmf": get_nmf
}

def reducer(vectors, k, r, **opts):
    return reducer_type[r](vectors, k, **opts)