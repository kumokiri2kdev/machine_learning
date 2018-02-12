from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

depths = range(3, 10)
train_score = np.zeros(len(depths))
test_score = np.zeros(len(depths))

for i, depth in enumerate(depths):
  tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
  tree.fit(X_train, y_train)
  print("Max Depth : {}".format(depth))
  train_score[i] = tree.score(X_train, y_train)
  test_score[i] = tree.score(X_test, y_test)
  print(" Accuracy on training set : {}".format(train_score[i]))
  print(" Accuracy on test set : {}".format(test_score[i] ))


plt.plot(depths, train_score, "o")
plt.plot(depths, test_score, "^")

plt.show()
