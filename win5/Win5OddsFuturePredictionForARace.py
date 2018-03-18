from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

lr = joblib.load('lr.pkl') 

new_remaining = 336

new_data_array= [
  [0.788 / 2.3, 5, 13],
]
print("intercept : {}".format(lr.intercept_))
print("coefficient : {}".format(lr.coef_[0]))

predicteds = lr.predict(new_data_array)
for predicted in predicteds:
  print("remaining : {}, prediction : {} ".format(predicted * new_remaining, predicted))


