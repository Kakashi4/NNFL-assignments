## Multiclass logistic regression (One vs One)

import numpy as np 
import pandas as pd 
from math import exp,log
import random


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

lr = 0.05

df = pd.read_excel('data4.xlsx', header=None)
df.iloc[:,0:7]=(df.iloc[:,0:7]-(df.iloc[:,0:7]).mean())/(df.iloc[:,0:7]).std()
df = df.to_numpy()
np.random.shuffle(df)
df = np.insert(df, 0, 1, axis=1)

# Train test split
X, test = df[:int(0.6*len(df))], df[int(0.6*len(df)):]

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

# Initializing weights    
weights = np.zeros([3, np.size(df,1)-1])

# Training weights
weights[0], weights[1], weights[2] = train(train1, 4000), train(train2, 4000), train(train3, 4000)

confusion_matrix = np.zeros((3,3))

# Training Loop
for row in test:

    predicted = 0
    h = []
    
    for i in range(3):
        h_ = np.dot(row[:-1], weights[i]) 
        h_ = sigmoid(h_)
        h.append(round(h_))

    if h[0] == 0 and h[1] == 0: predicted = 1
    if h[0] == 1 and h[2] == 0: predicted = 2
    if h[1] == 1 and h[2] == 1: predicted = 3

    confusion_matrix[int(row[-1])-1, predicted-1] += 1

c1 = confusion_matrix[0,0]/np.sum(confusion_matrix[0])
c2 = confusion_matrix[1,1]/np.sum(confusion_matrix[1])
c3 = confusion_matrix[2,2]/np.sum(confusion_matrix[2])
total_accuracy = (confusion_matrix[0,0] + confusion_matrix[1,1] + confusion_matrix[2,2])/np.sum(confusion_matrix)

print("Total accuracy: ", total_accuracy)
print("Class 1 accuracy: ", c1)
print("Class 2 accuracy: ", c2)
print("Class 3 accuracy: ", c3)
print(confusion_matrix)

