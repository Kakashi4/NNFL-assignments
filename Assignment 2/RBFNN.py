## Radial basis function neural network

import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans

mat_contents = loadmat('data5.mat')
data = mat_contents['x']
np.random.shuffle(data)

def init_data():
    X = np.array(data[:2148, :-1], dtype = float)
    y = np.array(data[:2148, -1], dtype = int)
    X = (X - X.mean(axis = 0))/X.std(axis = 0)
    return X, y

def gaussian(x,center,sigma,beta):
    return np.exp(-beta * (np.linalg.norm(x - center)) ** 2)

def multi_quadric(x, center, sigma, beta):
    return ((np.linalg.norm(x - center)) ** 2 + sigma ** 2) ** 0.5

def linear(x, center, sigma, beta):
    return np.linalg.norm(x - center)

X_tot, y_tot = init_data()

train_X = X_tot[ : 1600]
train_y = y_tot[ : 1600]
test_X = X_tot[1600 : 2148]
test_y = y_tot[1600 : 2148]

def fit_rbf(train_X, train_y, test_X, test_y):
    km = KMeans(n_clusters=550)

    y_km = km.fit_predict(train_X)
    centers = km.cluster_centers_
    labels = km.predict(train_X)

    sigma = np.zeros((len(centers), 1))
    beta = np.zeros((len(centers), 1))
    cluster_size = np.zeros((len(centers), 1))

    for i in range(len(train_X)):
        sigma[labels[i]] += np.linalg.norm(train_X[i] - centers[labels[i]])
        cluster_size[labels[i]] += 1

    sigma /= cluster_size
    beta = 1 / 2 * (sigma * sigma + 1e-6)

    H = np.zeros((len(train_X), len(centers)))

    for i in range(len(train_X)):
        for j in range(len(centers)):
            H[i, j] = linear(train_X[i], centers[j], sigma[j], beta[j])

    W = np.dot(np.linalg.pinv(H), train_y)

    #Test run
    H_test = np.zeros([len(test_X), len(centers)])
    for i in range(len(test_X)):
        for j in range(len(centers)):
            H_test[i, j] = linear(test_X[i], centers[j], sigma[j], beta[j])

    y_pred = np.dot(H_test, W)
    for i in range(len(y_pred)):
        y_pred[i] = 1 if y_pred[i]>=0.5 else 0
        
    accuracy = 0    
    for i in range(len(y_pred)):
        if y_pred[i] == test_y[i]:
            accuracy +=1
    accuracy /= len(y_pred)
    print(accuracy)
    return y_pred, accuracy

y_pred, _ = fit_rbf(train_X, train_y, test_X, test_y)
for i in range(len(y_pred)):
    y_pred[i] = 1 if y_pred[i] > 0.5 else 0

TP, TN, FP, FN = 0,0,0,0 

for i in range(len(test_X)):
    
    if y_pred[i] == 1 and test_y[i] == 1:
        TP += 1
    
    elif y_pred[i] == 0 and test_y[i] == 0:
        TN += 1
        
    elif y_pred[i] == 1 and test_y[i] == 0:
        FP += 1
        
    elif y_pred[i] == 0 and test_y[i] == 1:
        FN += 1
        
accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print("accuracy = ", accuracy, "sensitivity = ", sensitivity, "specificity = ", specificity)
print(TP, FP)
print(FN, TN)
avg_acc = 0

# K - fold cross validation

for k in range(6):
    X = X_tot[0 : 1790]
    y = y_tot[0 : 1790]
    X_val = X_tot[1790 :]
    y_val = y_tot[1790 :]
    _, acc = fit_rbf(X, y, X_val, y_val)
    avg_acc += acc
    X_tot[0 : 358] = X_val
    X_tot[358 : ] = X
    y_tot[0 : 358] = y_val
    y_tot[358 : ] = y

avg_acc /= 6
print(avg_acc)

