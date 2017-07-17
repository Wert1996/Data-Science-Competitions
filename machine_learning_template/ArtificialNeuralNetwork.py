from keras.layers import Dense
from keras.models import Sequential


def __init__(input_dim, hidden, output_dim, X_train, y_train, X_test):
    ann_classify(input_dim, hidden, output_dim, X_train, y_train, X_test)


# For classification
def ann_classify(input_dim, hidden, output_dim, X_train, y_train, X_test):
    model = Sequential()
    model.add(Dense(output_dim=hidden, init='uniform', activation='relu', input_dim=input_dim))
    model.add(Dense(output_dim=output_dim, init='uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=10, np_epoch=100)
    y_pred = model.predict(X_test)
    y_pred = (y_pred >= 0.5)
    return y_pred
