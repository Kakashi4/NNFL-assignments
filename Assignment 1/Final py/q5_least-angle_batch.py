##Least Angle regression (Batch)

import numpy as np
import pandas as pd

#Accept and normalize data, convert to numpy array
df = pd.read_excel('data.xlsx', header = None)
df.iloc[:,:-1]=(df.iloc[:,:-1]-(df.iloc[:,:-1]).mean())/(df.iloc[:,:-1]).std()
df = df.to_numpy()

X = df[:, :-1]
n = X.shape[0]
y = df[:, -1]
y_pred = np.zeros_like(y)

num_epochs = 200
w = [0, 0]
b = 0
lr = 2e-2
reg = 0.3
J = []
w1_history = []
w2_history = []

#Cost function
def eval_cost(error) :
    global w
    return np.sum(0.5 * (error **2) / n) + 0.5 * reg * (abs(w[0]) + abs(w[1]))

#Training loop
for i in range(num_epochs) :
    
    error = np.zeros_like(y)
    y_pred = np.sum(w * X, axis = 1) + b
    error = y_pred - y
    w[0] -= lr * (np.sum(error * X[:, 0])/n + reg * np.sign(w[0]))
    w[1] -= lr * (np.sum(error * X[:, 1])/n + reg * np.sign(w[1]))
    b -= lr * np.sum(error)/n
    
    J.append(eval_cost(error))
    w1_history.append(w[0])
    w2_history.append(w[1])
    
    



print(eval_cost(error))
print(w, b)

