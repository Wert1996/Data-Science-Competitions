from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def logistic_classification(X, y, X_test):
    print('Logistic Classification')
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X_test)
    return y_pred


def k_nearest_neighbors_classifier(X, y, X_test, neighbors=5, metric='minkowski', p=2 ):
    print('K nearest neighbors Classification')
    classifier = KNeighborsClassifier(n_neighbors=neighbors, metric=metric, p=p)
    classifier.fit(X, y)
    y_pred = classifier.predict(X_test)
    return y_pred


def support_vector_classifier(X, y, X_test,  kernel):
    print('Support Vector Classification')
    classifier = SVC(kernel=kernel, random_state=0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X_test)
    return y_pred


def naive_bayes(X, y, X_test):
    print('Naive Bayes Classification')
    classifier = GaussianNB()
    classifier.fit(X, y)
    y_pred = classifier.predict(X_test)
    return y_pred


def decision_tree_classification(X, y, X_test):
    print('Decision Tree Classification')
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X_test)
    return y_pred


def random_forest_classification(X, y, X_test, trees=10):
    print('Random Forest Classification')
    classifier = RandomForestClassifier(n_estimators=trees, criterion='entropy', random_state=0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X_test)
    return y_pred

