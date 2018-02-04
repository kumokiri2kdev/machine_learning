import numpy as np
import mglearn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mglearn.datasets import make_wave


def linear_regression_housing():
    X, y = make_wave(n_samples=60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    line = np.linspace(-3, 3, 100).reshape(-1, 1)

    lr = LinearRegression().fit(X_train, y_train)
    print("w[0]: %f  b: %f" % (lr.coef_[0], lr.intercept_))
    print("Training set score : {:.2f}". format(lr.score(X_train, y_train)))
    print("Test set score ] {:.2f}".format(lr.score(X_test, y_test)))
    
if __name__ == "__main__":
    linear_regression_housing()

