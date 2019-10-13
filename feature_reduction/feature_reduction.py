import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF

#PCA with standard scaler
def get_pca(vectors, k, **opts):
    std_scaler = StandardScaler()
    scaled_values = std_scaler.fit_transform(vectors)

    pca = PCA(n_components=k)
    pca_vectors = pca.fit_transform(scaled_values)
    return pca_vectors, pca.explained_variance_, pca.components_, pca, std_scaler

#SVD with standard scaler
def get_svd(vectors, k, **opts):
    std_scaler = StandardScaler()
    scaled_values = std_scaler.fit_transform(vectors)

    svd_vectors, eigenvalues, latent_vs_old_features = np.linalg.svd(scaled_values, full_matrices=False)
    return svd_vectors[:,:k], eigenvalues[:k], latent_vs_old_features[:,:k].T, None, std_scaler

#LDA with minmax scaler
def get_lda(vectors, k, **opts):
    min_max_scaler = MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(vectors)

    lda = LatentDirichletAllocation(n_components=k, verbose=2,random_state=0,learning_method='online',n_jobs=-1)
    lda_vectors = lda.fit_transform(scaled_values)
    return lda_vectors, None, lda.components_, lda, None, min_max_scaler

#NMF with minmax scaler
def get_nmf(vectors, k, **opts):
    min_max_scaler = MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(vectors)

    nmf = NMF(n_components=k, init='random', random_state=0)
    nmf_vectors = nmf.fit_transform(scaled_values)
    return nmf_vectors, None, nmf.components_, nmf, min_max_scaler

reducer_type = {
    "lda": get_lda,
    "pca": get_pca,
    "svd": get_svd,
    "nmf": get_nmf
}

def reducer(vectors, k, r, **opts):
    return reducer_type[r](vectors, k, **opts)
