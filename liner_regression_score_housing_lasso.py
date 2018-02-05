import numpy as np
import mglearn

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston

def linear_regression_boston(alpha):
    X, y = load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    lasso = Lasso(alpha=alpha, max_iter=100000).fit(X_train, y_train)
    print("-------------------")
    print("[alpha : {}]Training set score: {:.2f}".format(alpha, lasso.score(X_train, y_train)))
    print("[alpha : {}]Test set score: {:.2f}".format(alpha, lasso.score(X_test, y_test)))
    print("[alpha : {}]Number of features used: {}".format(alpha, np.sum(lasso.coef_ != 0)))
    
if __name__ == "__main__":
    alpha = [1.0, 0.01, 0.0001]

    for val in alpha:
      linear_regression_boston(val)
