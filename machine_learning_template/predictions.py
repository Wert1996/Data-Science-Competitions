import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datapreprocessing as dp
import Classification_methods as clam
import Regression_methods as reg
import Clustering_methods as clum
import metrics

# Import your dataset

# dataset = pd.read_csv('./datasets/.csv')
# print(dataset)

# Uncomment and apply preprocessing methods

# y = dataset.iloc[:, ].values
# X = dataset.iloc[:, []].values
# X = dp.impute(X, )
# X = dp.label_encoding(X, )
# X = dp.scaling(X)
# X_train, X_test = dp.split_data_into_train_and_test(X, 0.2)
# y_train, y_test = dp.split_data_into_train_and_test(y, 0.2)

# Apply model to get predictions

# y_test_pred = clam.naive_bayes(X_train, y_train, X_test)

# Metrics

# cm = metrics.confusion(y_test_pred, y_test)
# print(cm)
# print(metrics.accuracy(cm))
# print(metrics.precision(cm))
# print(metrics.recall(cm))
# print(metrics.f1_score(cm))
# print(metrics.r_squared_metric(y_test_pred, y_test))
# print(metrics.adjusted_r_squared_metric())

# Import test dataset

# data_to_predict = pd.read_csv('./datasets/.csv')

# Preprocess test dataset

# X_final = data_to_predict.iloc[:, []].values
# X_final = dp.impute(X_final, [])
# X_final = dp.label_encoding(X_final, [])
# X_final = dp.scaling(X_final)
# y_pred = clam.naive_bayes(X, y, X_final)

# Make dictionary for dataframe

# dicto = {}
# df = pd.DataFrame(dicto)

# Convert prediction dataset to csv file
# df.to_csv('./datasets/_predictions.csv', index=False)
