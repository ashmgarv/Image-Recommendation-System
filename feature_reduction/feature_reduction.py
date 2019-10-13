import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation

LDA = "lda"
PCA = "pca"
SVD = "svd"
NMF = "nmf"

def get_pca(vectors, k, **opts):
    #scaling values with standardscaler
    std_scaler = StandardScaler()
    scaled_values = std_scaler.fit_transform(vectors)

    pca = PCA(n_components=k)
    pca_vectors = pca.fit_transform(scaled_values)
    return pca_vectors, pca, std_scaler

def get_lda(vectors, k, **opts):
    #scaling vectors with minmaxscaler (0,1)
    min_max_scaler = MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(vectors)

    lda = LatentDirichletAllocation(n_components=k, verbose=2,random_state=0,learning_method='online',n_jobs=-1)
    lda_vectors = lda.fit_transform(scaled_values)
    return lda_vectors, lda, min_max_scaler

def get_svd(vectors, k, **opts):
    u, s, vh = np.linalg.svd(vectors, full_matrices=False)
    return u[:,:k], s[:k], vh[:,:k]

def get_nmf(vectors, k, **opts):
    model = NMF(n_components=k, init='random', random_state=0)
    u = model.fit_transform(data)
    h = model.components_
    return u, h

reducer_type = {
    "lda": get_lda,
    "pca": get_pca,
    "svd": get_svd,
    "nmf": get_nmf
}

def reducer(vectors, k, r, **opts):
    return reducer_type[r](vectors, k, **opts)
