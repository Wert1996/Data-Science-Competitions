from sklearn.decomposition import PCA
import numpy as np


def pca(X_train, X_test, n_components=None):
    """
    :param n_components: The number of components to be preserved
    :return: datasets with reduced dimensionality if n_components is provided.
    Otherwise returns the variance ratio
    """
    principal = PCA(n_components=n_components)
    X_train = principal.fit_transform(X_train)
    X_test = principal.transform(X_test)
    explained_variance = principal.explained_variance_ratio_
    if n_components is None:
        return explained_variance
    return X_train, X_test