import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

win5_df = pd.read_csv("data/win5.csv")

X_data_odds = 0.788 / win5_df['odds'].values.reshape(-1,1)
X_data_index =  win5_df[['index', 'hcount']].values.reshape(-1,2)
X_data = np.c_[X_data_odds, X_data_index]
y_data = win5_df['ratio'].values

lr = LinearRegression().fit(X_data, y_data)
print("intercept : {}".format(lr.intercept_))
print("coefficient : {}".format(lr.coef_[0]))

joblib.dump(lr, 'lr.pkl')

