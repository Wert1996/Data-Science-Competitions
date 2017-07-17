import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


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







