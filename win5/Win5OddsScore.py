import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

win5_result = pd.read_csv("data/win5_result.csv",header=1)

lr = joblib.load('lr.pkl') 

new_remaining = 7069088

print("intercept : {}".format(lr.intercept_))
print("coefficient : {}".format(lr.coef_[0]))

new_data_array= [
  [0.788 / 2.6, 1, 13],
  [0.788 / 26.6, 2, 12],
  [0.788 / 1.6, 3, 9],
  [0.788 / 22.3, 4, 18],
  [0.788 / 2.7, 5, 13],
]

diffs = np.zeros(len(win5_result.values))

for i, result in enumerate(win5_result.values):
  if result[22] == 1:
    data_array = np.array(result[1:16]).reshape(-1,3)
    remaining_array = np.array(result[16:22])
    predicteds = lr.predict(data_array)
    predicted = remaining_array[0]
    for j, pred in enumerate (predicteds):
      predicted = predicted * pred

    print("{}, {}, {}, {}".format(result[0], remaining_array[5], predicted, remaining_array[5] - predicted))
    diffs[i] = round(remaining_array[5] - predicted,3)

  else:
    pass

print(np.mean(diffs))
print(np.max(diffs))
print(np.min(diffs))

