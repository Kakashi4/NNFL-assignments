## K-fold cross validation (One vs One)

import numpy as np 
import pandas as pd 
from math import exp,log
import random

lr = 0.01

def most_frequent(List): 
    return max(set(List), key = List.count) 

# Define activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Calculate gradients
def getGradients(x, weights):
    
    delta = np.zeros(len(x)-1, dtype=float)
    
    h = np.dot(weights, x[:-1])
    h = sigmoid(h)

    delta = [(h-x[-1])*x[i] for i in range(len(x)-1)]
    return delta

# Training function
def train(data, num_iter):
    '''Returns weights'''

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

overall_accuracy = 0

# k-fold loop
for k in range(5):

    fold_accuracy = [0.0,0.0,0.0]
    count = [0,0,0]

    data = df.copy()
    test = data[k*int(len(df)/5) : (k+1)*int(len(df)/5)]
    X = np.delete(data, np.s_[k*int(len(df)/5):(k+1)*int(len(df)/5)], 0)

    # Initializing weights
    weights = np.zeros([3, np.size(df,1)-1])

    # Preparing training data
    train1, train2, train3 = [], [], [] 

    for row in X:

        if row[-1] == 1:
            train1.append(row)
            train2.append(row)  

        elif row[-1] == 2:
            train1.append(row)
            train3.append(row)  

        elif row[-1] == 3:
            train2.append(row)
            train3.append(row)  

    train1, train2, train3 = np.array(train1), np.array(train2), np.array(train3)   

    for row in train1:
        row[-1] = 0 if row[-1] == 1 else 1  

    for row in train2:
        row[-1] = 0 if row[-1] == 1 else 1  

    for row in train3:
        row[-1] = 0 if row[-1] == 2 else 1  

    # Training weights
    weights[0], weights[1], weights[2] = train(train1, 3000), train(train2, 3000), train(train3, 3000)  

    for row in test:  

        count[int(row[-1])-1] += 1 

        h = []
        for i in range(3):
            h_ = np.dot(row[:-1], weights[i]) 
            h_ = sigmoid(h_)
            h.append(round(h_)) 

        predicted = 0   

        if h[0] == 0 and h[1] == 0:
            predicted = 1

        if h[0] == 1 and h[2] == 0:
            predicted = 2   

        if h[1] == 1 and h[2] == 1:
            predicted = 3

        if(predicted == row[-1]):
            fold_accuracy[int(row[-1]) - 1] += 1

    fold_accuracy = np.divide(fold_accuracy, count)
    print("Fold {}".format(k+1))
    
    for i in range(3):
        print("Class {} accuracy: {}".format(i+1, fold_accuracy[i]))

    print("Total accuracy {}".format(np.mean(fold_accuracy)))