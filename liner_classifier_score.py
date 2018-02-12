from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

Cs = [1.0, 100, 0.01]

print("Logistic Regression")
for c in Cs:
  logreg = LogisticRegression(C=c).fit(X_train, y_train)
  print(" C={}".format(c))
  print("  Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
  print("  Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

print("Liner SVC")
for c in Cs:
  logreg = LinearSVC(C=c).fit(X_train, y_train)
  print(" C={}".format(c))
  print("  Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
  print("  Test set score: {:.3f}".format(logreg.score(X_test, y_test)))