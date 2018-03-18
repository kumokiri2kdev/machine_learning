import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

lr = joblib.load('lr.pkl') 

new_remaining = 7161428

new_data_array= [
  [0.788 / 2.7, 1, 11],
]
print("intercept : {}".format(lr.intercept_))
print("coefficient : {}".format(lr.coef_[0]))

predicteds = lr.predict(new_data_array)
for predicted in predicteds:
  new_remaining = predicted * new_remaining
  print("remaining : {}, prediction : {} ".format(new_remaining, predicted))


