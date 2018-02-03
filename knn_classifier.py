import numpy as np
import matplotlib.pyplot as plt
import mglearn
from mglearn.make_blobs import make_blobs
from mglearn.plot_helpers import discrete_scatter
from mglearn.datasets import make_forge

from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier

def plot_knn_classification(n_neighbors=1):
    X, y = make_forge()
    x_test = np.arange(8, 12, 0.5)
    y_test = (x_test - 8) * 1.4
    X_test = np.c_[x_test, y_test]

    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    test_points = discrete_scatter(X_test[:, 0], X_test[:, 1], clf.predict(X_test), markers="*")
    training_points = discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(training_points + test_points, ["training class 0", "training class 1","test pred 0", "test pred 1"])

    dist = euclidean_distances(X, X_test)
    closest = np.argsort(dist, axis=0)

    for x, neighbors in zip(X_test, closest.T):
        for neighbor in neighbors[:n_neighbors]:
            plt.arrow(x[0], x[1], X[neighbor, 0] - x[0],
                      X[neighbor, 1] - x[1], head_width=0, fc='k', ec='k')

if __name__ == "__main__":
    plot_knn_classification(n_neighbors=3)
    plt.show()
