import numpy as np
import matplotlib.pyplot as plt
import mglearn

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston

def lasso_fit(alpha, X_train, X_test, y_train, y_test):
    lasso = Lasso(alpha=alpha, max_iter=100000).fit(X_train, y_train)
    return lasso.score(X_train, y_train), lasso.score(X_test, y_test)

def linear_regression_boston():
    X, y = load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    limit = np.arange(0.0001, 1.0001, 0.001)

    training_scores = np.zeros(len(limit), dtype=float)
    test_scores = np.zeros(len(limit), dtype=float)

    best_score = 0.0
    best_alpha = 0.0
    for i, alpha in enumerate(limit):
      training_scores[i], test_scores[i] = lasso_fit(alpha, X_train, X_test, y_train, y_test)
      if best_score < test_scores[i]:
        best_score = test_scores[i]
        best_alpha = alpha

    print("Best Alpha : {}, Best Score : {}".format(best_alpha, best_score))

    plt.plot(limit, test_scores, label="Test Score")
    plt.plot(limit, training_scores, label="Tranining Score")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    linear_regression_boston()
