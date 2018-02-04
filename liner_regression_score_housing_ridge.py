import numpy as np
import mglearn

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston

def linear_regression_boston():
    X, y = load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    ridge = Ridge().fit(X_train, y_train)
    print("w[0]: %f  b: %f" % (ridge.coef_[0], ridge.intercept_))
    print("Training set score : {:.2f}". format(ridge.score(X_train, y_train)))
    print("Test set score ] {:.2f}".format(ridge.score(X_test, y_test)))
    
if __name__ == "__main__":
    linear_regression_boston()
