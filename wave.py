import mglearn
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

X, y = mglearn.datasets.make_wave(n_samples=400)

plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")

plt.show()
