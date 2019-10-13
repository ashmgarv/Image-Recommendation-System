import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation


def get_pca(vectors, k):
    #scaling values with standardscaler
    std_scaler = StandardScaler()
    scaled_values = std_scaler.fit_transform(vectors)

    pca = PCA(n_components=k)
    pca_vectors = pca.fit_transform(scaled_values)
    return pca_vectors, pca, std_scaler

def get_lda(vectors, k):
    #scaling vectors with minmaxscaler (0,1)
    min_max_scaler = MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(vectors)

    lda = LatentDirichletAllocation(n_components=k, verbose=2,random_state=0,learning_method='online',n_jobs=-1)
    lda_vectors = lda.fit_transform(scaled_values)
    return lda_vectors, lda, min_max_scaler


