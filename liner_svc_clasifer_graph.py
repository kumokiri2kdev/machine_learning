from sklearn.svm import LinearSVC
import mglearn
import numpy as np
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_forge()

clf = LinearSVC().fit(X, y)

x_min, x_max = X[:, 0].min(), X[:, 0].max() 
y_min, y_max = X[:, 1].min(), X[:, 1].max() 

xx = np.linspace(x_min, x_max, 1000)
yy = np.linspace(y_min, y_max, 1000)

X1, X2 = np.meshgrid(xx, yy)
X_grid = np.c_[X1.ravel(), X2.ravel()]

decision_values = clf.decision_function(X_grid)

plt.contour(X1, X2, decision_values.reshape(X1.shape), levels=[0])
plt.plot(X[y==0][:,0], X[y==0][:,1], "o")
plt.plot(X[y==1][:,0], X[y==1][:,1], "*")

plt.show()

