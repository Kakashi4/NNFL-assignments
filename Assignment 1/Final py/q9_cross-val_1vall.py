## K-fold cross validation (One vs All)

import numpy as np 
import pandas as pd 
from math import exp,log
import random

lr = 0.01

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

# Read, normalize and shuffle dataset
df = pd.read_excel('data4.xlsx', header=None)
df.iloc[:,:-1]=(df.iloc[:,:-1]-(df.iloc[:,:-1]).mean())/(df.iloc[:,:-1]).std()
df = df.to_numpy()
np.random.shuffle(df)
df = np.insert(df, 0, 1, axis=1)

# k-fold loop
for k in range(5):

    data = df.copy()
    test = data[k*int(len(df)/5) : (k+1)*int(len(df)/5)]
    X = np.delete(data, np.s_[k*int(len(df)/5):(k+1)*int(len(df)/5)], 0)

    weights = np.zeros([3, np.size(df,1)-1])
    accuracy = np.zeros(3)
    count = [0,0,0]
   
   # Training loop
    for i in range(3):  

        data = df.copy()
        X = np.delete(data, np.s_[k*int(len(df)/5):(k+1)*int(len(df)/5)], 0)

        for row in X:
            if row[-1] == i+1:
                row[-1] = 1

            else:
                row[-1] = 0 

        weights[i] = train(X,5000)    

    # Testing loop
    for row in test:   

        # Count number of instances per class
        count[int(row[-1])-1] += 1

        h = np.zeros(3)
        for i in range(3):
            h_ = np.dot(weights[i], row[:-1])
            h_ = sigmoid(h_)
            h[i] = h_ 

        predicted = np.argmax(h) + 1
        if(predicted == row[-1]):
            accuracy[int(row[-1])-1] += 1
    
    accuracy = np.divide(accuracy, count)
    print("Fold {}".format(k))
    
    for i in range(3):
        print("Class {} Accuracy: {}".format(i+1,accuracy[i]))
    
    print("Total accuracy {}".format(np.mean(accuracy)))