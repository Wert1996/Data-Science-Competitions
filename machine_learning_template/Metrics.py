import numpy as np
from sklearn.metrics import confusion_matrix


def r_squared_metric(y_pred, y_actual):
    y_bar = np.mean(y_actual)
    SS_res = 0
    SS_tot = 0
    for i,j in zip(y_pred, y_actual):
        SS_res += (i - j)**2
        SS_tot += (j-y_bar)**2
    r_squared = 1-(SS_res/SS_tot)
    return r_squared


def adjusted_r_squared_metric(y_pred, y_actual, no_of_regressors):
    r_squared = r_squared_metric(y_pred, y_actual)
    n = len(y_actual)
    adjusted_r_squared = 1-((1-r_squared)*((n-1)/(n-no_of_regressors-1)))
    return adjusted_r_squared


def confusion(y_pred, y_actual):
    cm = confusion_matrix(y_actual, y_pred)
    return cm


def accuracy(cm):
    acc = (cm[0][0] + cm[1][1]) / cm.sum()
    return acc


def precision(cm):
    prec = cm[0][0] / (cm[0][0] + cm[0][1])
    return prec


def recall(cm):
    recal = cm[0][0] / (cm[0][0] + cm[1][0])
    return recal


def f1_score(cm):
    f1 = 2 * precision(cm) * recall(cm) / (precision(cm) + recall(cm))
    return f1

