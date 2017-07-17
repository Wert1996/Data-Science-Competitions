from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split


def impute(X, list_indices):
    print('Providing missing values..')
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X[:, list_indices] = imputer.fit_transform(X[:, list_indices])
    return X


def label_encoding(X, list_indices):
    print('Encoding with label..')
    for i in list_indices:
        label_encoder = LabelEncoder()
        X[:, i] = label_encoder.fit_transform(X[:, i])
    return X


def oneHotEncoding(X, list_indices):
    print('Onehotencoding..')
    X = X.astype(float)
    for i in list_indices:
        onehotencoder = OneHotEncoder(categorical_features=i)
        X = onehotencoder.fit_transform(X)
    return X


def split_data_into_train_and_test(X, ratio):
    print('Splitting data')
    X_train, X_test = train_test_split(X, test_size=ratio, random_state=0)
    return X_train, X_test


def scaling(X):
    print('Scaling the data..')
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X