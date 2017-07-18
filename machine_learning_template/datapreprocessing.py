from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split


import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def impute(X):
    print('Providing missing values..')
    X = DataFrameImputer().fit_transform(X)
    return X


def label_encoding(X, list_indices=None):
    print('Encoding with label..')
    if list_indices is None:
        list_indices = []
        for i in range(X.shape[1]):
            if isinstance(X[0][i], str):
                list_indices.append(i)
    for i in list_indices:
        label_encoder = LabelEncoder()
        X[:, i] = label_encoder.fit_transform(X[:, i])
    return X


def oneHotEncoding(X, list_indices):
    """Also takes care of redundant variables
    Removes one column of dummy variable and returns the new matrix"""
    print('Onehotencoding..')
    X = X.astype(float)
    for i in list_indices:
        onehotencoder = OneHotEncoder(categorical_features=[i])
        X = onehotencoder.fit_transform(X)
    return X[1:]


def encode(X, list_indices=None):
    print('Encoding with label..')
    if list_indices is None:
        list_indices = []
        for i in range(X.shape[1]):
            if isinstance(X[0][i], str):
                list_indices.append(i)
    for i in list_indices:
        label_encoder = LabelEncoder()
        X[:, i] = label_encoder.fit_transform(X[:, i])
    print('Onehotencoding..')
    list_indices = [i - X.shape[1] for i in list_indices]
    for i in list_indices:
        onehotencoder = OneHotEncoder(categorical_features=[i])
        X = onehotencoder.fit_transform(X).toarray()
        X = X[:, 1:]
    return X


def split_data_into_train_and_test(X, y, ratio):
    print('Splitting data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=0)
    return X_train, X_test, y_train, y_test


def scaling(X):
    print('Scaling the data..')
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X