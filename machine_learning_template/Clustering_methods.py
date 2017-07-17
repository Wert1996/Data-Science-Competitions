from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


def k_means_clustering(X):
    print ("Using the elbow method to perform K means clustering..")
    wcss = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    max_ratio = 0
    optimum_clusters = 0
    for i in range(1,9):
        if (wcss[i] - wcss[i - 1]) / (wcss[i + 1] - wcss[i - 1]) > max_ratio:
            max_ratio = (wcss[i] - wcss[i - 1]) / (wcss[i + 1] - wcss[i - 1])
            optimum_clusters = i + 1
    kmeans = KMeans(n_clusters=optimum_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    return y_kmeans


def hierarchical_clustering(X):
    print('Plotting Dendrogram,.')
    dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
    plt.title('Dendrogram')
    plt.ylabel('Euclidean Distances')
    plt.show()
    optimum_clusters = input('Input the optimum cluster...')
    print('Performing Hierarchical Clustering..')
    hc = AgglomerativeClustering(n_clusters=optimum_clusters, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)
    return y_hc