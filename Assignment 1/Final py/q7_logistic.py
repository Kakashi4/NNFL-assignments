import numpy as np
import pandas as pd
import matplotlib as plt
from math import exp, log


df = pd.read_excel('data3.xlsx', header = None)
df = df.to_numpy()
np.random.shuffle(df)

df = np.insert(df, 0, 1, axis  = 1)
X = df[0:60, :-1]
y = df[0:60, -1]

X_test = df[60:100, :-1]
y_test = df[60:100, -1]

y -= 1
y_test -= 1

w = np.zeros(5, dtype= 'float')
n = X.shape[0]
n_test = X_test.shape[0]
num_epoch = 200

lr = 1e-2
J = []
def sigmoid(x) :
    return 1/(1 + np.exp(-x))

def eval_cost(h, y) :
    return -(y * np.log(h) + (1-y) * np.log(1-h))

for i in range(num_epoch) :
    cost = 0
    h = np.dot(X, w)
    h = sigmoid(h)
    error = h-y
    cost = np.mean(eval_cost(h, y))
    grads = X.T.dot(error)
    w -= lr * grads
    J.append(cost)
    #print(cost)

TN, FN, TP, FP = 0, 0, 0, 0

y_pred = np.dot(X_test, w)
y_pred = sigmoid(y_pred)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0
print(y_pred)
TP = np.sum(y_pred * y_test)
FP = np.sum(y_pred * (1-y_test))
FN = np.sum((1-y_pred) * y_test)
TN = np.sum((1-y_pred) * (1-y_test))
print(y_test)
sensitivity = TP/(TP + FN)
specificity = TN/(TN + FP)
accuracy = (TP + TN)/(TP + TN + FP + FN)
print(sensitivity, specificity, accuracy)
