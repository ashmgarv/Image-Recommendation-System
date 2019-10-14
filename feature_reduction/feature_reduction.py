import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF


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

    pca = PCA(n_components=k)
    pca_vectors = pca.fit_transform(scaled_values)

    #if opts contains query vector, apply scaler and PCA transformation to the query vector and return
    if opts:
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

    svd_vectors, eigenvalues, latent_vs_old_features = np.linalg.svd(scaled_values, full_matrices=False)
    return svd_vectors[:,:k], eigenvalues[:k], latent_vs_old_features[:k,:]


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

    lda = LatentDirichletAllocation(n_components=k, verbose=2,random_state=0,learning_method='online',n_jobs=-1)
    lda_vectors = lda.fit_transform(scaled_values)

    #if opts contains query vector, apply scaler and PCA transformation to the query vector and return
    if opts:
        query_vector = opts['query_vector']
        scaled_query_vector = min_max_scaler.transform(query_vector)
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

    nmf = NMF(n_components=k, init='random', random_state=0)
    nmf_vectors = nmf.fit_transform(scaled_values)

    #if opts contains query vector, apply scaler and PCA transformation to the query vector and return
    if opts:
        query_vector = opts['query_vector']
        scaled_query_vector = min_max_scaler.transform(query_vector)
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
