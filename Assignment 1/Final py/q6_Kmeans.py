# K-means clustering

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

df = pd.read_excel('data2.xlsx', header = None)
df = df.to_numpy()
X = df
n = X.shape[0]
dist1 = np.zeros(n)
dist2 = np.zeros(n)
class_label = np.zeros(n)


init_mean1 = np.random.random(4)
init_mean2 = np.random.random(4)

def calculate_dist(X_in, mean) :
    distance = np.sqrt(np.sum((X_in - mean)**2, axis = 1))
    return distance

def calculate_mean(X_in, class_label) :
    X_temp = np.reshape(class_label, (n, -1)) * X_in
    X1 = X_temp * (X_temp>0)
    X2 = -1 *(X_temp * (X_temp<0))

    n1 = X1[X1>0].size/4
    n2 = X2[X2>0].size/4

    mean1 = np.sum(X1, axis = 0)/n1
    mean2 = np.sum(X2, axis = 0)/n2
    
    return mean1, mean2
    
init_mean1 = X[np.random.randint(151)]
init_mean2 = X[np.random.randint(151)]
    
mean1 = init_mean1
mean2 = init_mean2

prev_sum_mean = 0
new_sum_mean = 1

while(np.sum(new_sum_mean - prev_sum_mean) > 0.0001) :
    prev_sum_mean = mean1 + mean2
    dist1 = calculate_dist(X, mean1)
    dist2 = calculate_dist(X, mean2)
    class_label = np.sign(dist1 - dist2)
    mean1, mean2 = calculate_mean(X, class_label)
    new_sum_mean = mean1 + mean2
    
    
LABEL_COLOUR_MAP = {-1 : 'r', 1 : 'b'}
label_colour = [LABEL_COLOUR_MAP[l] for l in class_label]

f = plt.figure(figsize = (100, 100))
plt.title('Class labels (Colour) vs pairs of features')
f.add_subplot(3, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c = label_colour)
f.add_subplot(3, 2, 2)
plt.scatter(X[:, 0], X[:, 2], c = label_colour)
f.add_subplot(3, 2, 3)
plt.scatter(X[:, 0], X[:, 3], c = label_colour)
f.add_subplot(3, 2, 4)
plt.scatter(X[:, 1], X[:, 2], c = label_colour)
f.add_subplot(3, 2, 5)
plt.scatter(X[:, 1], X[:, 3], c = label_colour)
f.add_subplot(3, 2, 6)
plt.scatter(X[:, 2], X[:, 3], c = label_colour)
plt.show()