## Ridge regression (minibatch)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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
batch_size = 16
w1_history = []
w2_history = []

#Cost function
def eval_cost(X, y, w, b) :
   y_pred = X[:, 0] * w[0] + X[:, 1] * w[1] + b
   error = y_pred - y
   return 0.5 * np.sum(error **2) /n + 0.5 * reg * (w[0]**2 + w[1]**2)

#Training function to update weights once
def train(X, y, n) :
    global w, b
    error = np.zeros_like(y)
    y_pred = X[:, 0] * w[0] + X[:, 1] * w[1] + b
    error = y_pred - y
    w[0] -= lr * (np.sum(error * X[:, 0])/n + reg * w[0])
    w[1] -= lr * (np.sum(error * X[:, 1])/n + reg * w[1])
    b -= lr * np.sum(error)/n
    w1_history.append(w[0])
    w2_history.append(w[1])

#Training loop    
for j in range(num_epochs):
    rand_sample = np.random.randint(n, size = batch_size)
    batch_X = X[rand_sample, :]
    batch_y = y[rand_sample]
    train(batch_X, batch_y, batch_size)
    J.append(eval_cost(X, y, w, b))    

#PLots
plt.plot(J)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')

print(J[-1])
print(w, b)

ww1 = np.linspace(min(w1_history)-2,max(w1_history)+2,100)
ww2 = np.linspace(min(w2_history)-2,max(w2_history)+2,100)

J_cont = np.zeros((100, 100))

for i1, w1 in enumerate(ww1):
    for i2, w2 in enumerate(ww2):

        cost = 0
        for i in range(n):

            error = np.zeros_like(y)
            y_pred = w1 * X[:, 0] + w2 * X[:, 1] + b
            error = y_pred - y
            cost = eval_cost(X, y, [w1, w2], b)

        J_cont[i1, i2] = cost
        
www1,www2 = np.meshgrid(ww1,ww2)
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.plot_surface(www1,www2,J_cont, cmap=cm.coolwarm)
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('Cost function')
plt.title("Contour plot: Cost function vs weights")
plt.show()