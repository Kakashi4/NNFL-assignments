## Vectorized gradient descent

import numpy as np
import pandas as pd

df = pd.read_excel('data.xlsx', header = None)
df.iloc[:,:-1]=(df.iloc[:,:-1]-(df.iloc[:,:-1]).mean())/(df.iloc[:,:-1]).std()
df = df.to_numpy()
X2 = df[:, :-1]
temp = np.ones((X2.shape[0], X2.shape[1] + 1))
temp[:, :-1] = X2
X = temp
n = X.shape[0]
y = df[:, -1]

w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))    
y_pred = X[:, 0] * w[0] + X[:, 1] * w[1] + w[2]
error = y_pred - y
print(np.sum(0.5 * (error **2) / n))
print(w)
