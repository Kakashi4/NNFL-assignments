## Multiclass logistic regression (One vs All)

import numpy as np 
import pandas as pd 
from math import exp,log
import random

# Define activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Function to calculate gradients for logistic regression
def getGradients(x, weights):
    delta = np.zeros(len(x)-1, dtype=float)
    h = np.dot(weights, x[:-1])
    h = sigmoid(h)
    delta = [(h-x[-1])*x[i] for i in range(len(x)-1)]
    return delta

# Training function
def train(data, num_iter):
    weights = np.zeros(np.size(data,1)-1, dtype=float)
    
    for i in range(num_iter):
    
        x = data[random.randint(0,len(data)-1)] 
        delta = getGradients(x, weights)
        weights = [weights[i] - lr*delta[i] for i in range(len(x)-1)]

    return weights

lr = 0.01

# Read, normalize and shuffle dataset
df = pd.read_excel('data4.xlsx', header=None)
df.iloc[:,:-1]=(df.iloc[:,:-1]-(df.iloc[:,:-1]).mean())/(df.iloc[:,:-1]).std()
df = df.to_numpy()
np.random.shuffle(df)
df = np.insert(df, 0, 1, axis=1)

test = df[int(0.6*len(df)) : ]
X = df[ : int(0.6*len(df))]

weights = np.zeros([3, np.size(df,1)-1])
accuracy = np.zeros(3)
count = [0,0,0]

# Training loop
for i in range(3):  
    data = df.copy()
    X = data[ : int(0.6*len(df))]

    for row in X:
        if row[-1] == i+1:
            row[-1] = 1
        else:
            row[-1] = 0 
    weights[i] = train(X,5000)    

confusion_matrix = np.zeros((3,3))

# Testing loop
for row in test:  

    # Count number of instances per class
    count[int(row[-1])-1] += 1
    h = np.zeros(3)
    
    for i in range(3):
        h_ = sigmoid(np.dot(weights[i], row[:-1]))
        h[i] = h_ 
    
    l = np.argmax(h) + 1
    confusion_matrix[int(row[-1])-1, l-1] += 1

c1 = confusion_matrix[0,0]/np.sum(confusion_matrix[0])
c2 = confusion_matrix[1,1]/np.sum(confusion_matrix[1])
c3 = confusion_matrix[2,2]/np.sum(confusion_matrix[2])
total_accuracy = (confusion_matrix[0,0] + confusion_matrix[1,1] + confusion_matrix[2,2])/np.sum(confusion_matrix)

print("Total accuracy: ", total_accuracy)
print("Class 1 accuracy: ", c1)
print("Class 2 accuracy: ", c2)
print("Class 3 accuracy: ", c3)
print(confusion_matrix)