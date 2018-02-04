import numpy as np
import matplotlib.pyplot as plt
import mglearn

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston

def ridge_fit(alpha, X_train, X_test, y_train, y_test):
    ridge = Ridge(alpha=alpha).fit(X_train, y_train)
    #print("w[0]: %f  b: %f" % (ridge.coef_[0], ridge.intercept_))
    #print("Training set score : {:.2f}". format(ridge.score(X_train, y_train)))
    #print("Test set score ] {:.2f}".format(ridge.score(X_test, y_test)))
    return ridge.score(X_train, y_train), ridge.score(X_test, y_test)

def linear_regression_boston():
    X, y = load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    limit = np.arange(0.01, 0.50, 0.01)

    training_scores = np.zeros(len(limit), dtype=float)
    test_scores = np.zeros(len(limit), dtype=float)

    for i, alpha in enumerate(limit):
      training_scores[i], test_scores[i] = ridge_fit(alpha, X_train, X_test, y_train, y_test)
    
    plt.plot(limit, test_scores, label="Test Score")
    plt.plot(limit, training_scores, label="Tranining Score")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    linear_regression_boston()


