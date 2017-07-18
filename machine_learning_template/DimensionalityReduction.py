from sklearn.decomposition import PCA
import numpy as np


def pca(X, n_components=None):
    """
    :param n_components: The number of components to be preserved
    :return: datasets with reduced dimensionality if n_components is provided.
    Otherwise returns the variance ratio
    """
    principal = PCA(n_components=n_components)
    X = principal.fit_transform(X)
    explained_variance = principal.explained_variance_ratio_
    if n_components is None:
        return explained_variance
    return X