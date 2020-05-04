## Least Angle regression (Stochastic)
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
num_epochs = 500
w = [0, 0]
b = 0
lr = 5e-3
reg = 0.3
J = []
w1_history = []
w2_history = []

#Cost function
def eval_cost(X, y, w, b) :
   y_pred = X[:, 0] * w[0] + X[:, 1] * w[1] + b
   error = y_pred - y
   return 0.5 * np.sum(error **2)/n + 0.5 * reg * (abs(w[0]) + abs(w[1]))

#Training loop
for i in range(num_epochs) :
    
        j = np.random.randint(n)
        y_pred = X[j, 0] * w[0] + X[j, 1] * w[1] + b
        error = y_pred - y[j]
        w[0] += -(lr * (error * X[j, 0] + reg * np.sign(w[0])))
        w[1] += -(lr * (error * X[j, 1] + reg * np.sign(w[1])))
        b += -(lr * error)
        J.append(eval_cost(X, y, w, b))
        w1_history.append(w[0])
        w2_history.append(w[0])
    
    


print(J[-1])
print(w, b)

