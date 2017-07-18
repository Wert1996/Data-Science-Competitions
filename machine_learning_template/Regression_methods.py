from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def linear_regression(X, y, X_test):
    print ('Linear Regression..')
    regressor = LinearRegression()
    regressor.fit(X,y)
    y_pred = regressor.predict(X_test)
    return y_pred


def polynomial_regression(X, y, X_test, degree):
    print('Polynomial Regression')
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X)
    regressor = LinearRegression()
    regressor.fit(X, y)
    y_pred = regressor.predict(X_test)
    return y_pred


def support_vector_regression(X, y, X_test, kernel):
    print('Support Vector Regression')
    regressor = SVR(kernel=kernel)
    regressor.fit(X, y)
    y_pred = regressor.predict(X_test)
    return y_pred


def decision_tree_regressor(X, y, X_test):
    print('Decision tree Regression')
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X, y)
    y_pred = regressor.predict(X_test)
    return y_pred


def random_forest_regressor(X, y, trees, X_test):
    print('Random Forest Regression')
    regressor = RandomForestRegressor(n_estimators=trees, random_state=0)
    regressor.fit(X, y)
    y_pred = regressor.predict(X_test)
    return y_pred


# Regression using Artificial Neural Nets
def ann_regression(first_hidden_layer, X_train, y_train, X_test, second_hidden_layer = None):
    regressor = Sequential()
    regressor.add(Dense(output_dim=first_hidden_layer, input_dim=X_train.shape[1], activation='relu'))
    if second_hidden_layer is not None:
        regressor.add(Dense(output_dim=second_hidden_layer, activation='relu'))
    regressor.add(Dense(output_dim=1, activation='linear'))
    regressor.compile(optimizer='adam', loss='mse')
    regressor.fit(X_train, y_train, epochs=1000, batch_size=20)
    y_pred = regressor.predict(X_test)
    return y_pred
