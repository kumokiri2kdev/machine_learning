from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

import matplotlib.pyplot as plt

plt.barh(range(cancer.data.shape[1]), forest.feature_importances_)
plt.yticks(range(cancer.data.shape[1]), cancer.feature_names)

plt.show()
