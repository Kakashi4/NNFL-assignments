## Extreme learning machine based classifier

import numpy as np 
from scipy.io import loadmat
from sklearn.preprocessing import normalize


#Gaussian activation function
def gaussian(X, a, b):   
    K = np.zeros((X.shape[0], hidden_neurons))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i,j] = np.exp(-b[j] * np.linalg.norm(a[:,j] - X[i,:]))  
    return K

#Tanh activation function
def tanh(X):
    return np.tanh(X)

#Load data, shuffle and normalize
def init_data():
    X = np.array(data[ : , :-1], dtype = float)
    y = np.array(data[ : , -1], dtype = int)
    X = normalize(X, axis = 0)
    return X, y

mat_contents = loadmat('data5.mat')
data = mat_contents['x']
np.random.shuffle(data)

X_tot, y_tot = init_data()
X_tot = np.insert(X_tot, 0, 1, axis=1)

#Generate labels matrix
labels = data[:, -1]
y = np.zeros([len(X_tot), 2])

for i in range(len(labels)):
    if labels[i] == 1:
        y[i,1] = 1.0
    elif labels[i] == 0:
        y[i,0] = 1.0

hidden_neurons = 100
output_neurons = 2

#K fold cross validation
#Gaussian activation function
print("Gaussian activation function fold accuracies: ")
for k in range(6):

    X_train = X_tot[0 : 1790]
    y_train = y[0 : 1790]
    X_val = X_tot[1790 :]
    y_val = y[1790 :]
    
    a = np.random.rand(X_train.shape[1], hidden_neurons) 
    b = np.random.rand(hidden_neurons)
    
    # Training 
    H = gaussian(X_train, a, b)
    H_inv = np.linalg.pinv(H)
    W2 = np.matmul(H_inv, y_train)
    
    # Testing
    H_T = gaussian(X_val, a, b)
    y_pred = np.matmul(H_T, W2)
    
    accuracy = 0

    for p in range(len(y_pred)):
        if np.argmax(y_pred[p]) == np.argmax(y_val[p]):
            accuracy += 1
    accuracy = accuracy / len(y_val)
    print(accuracy)

    X_tot[0 : 358] = X_val
    X_tot[358 : ] = X_train
    y[0 : 358] = y_val
    y[358 : ] = y_train

#Tanh activation function
print("Tanh activation function fold accuracies:")
for k in range(6):
    
    X_train = X_tot[0 : 1790]
    y_train = y[0 : 1790]
    X_val = X_tot[1790 :]
    y_val = y[1790 :]
    
    a = np.random.rand(X_train.shape[1],hidden_neurons) 
    b = np.random.rand(hidden_neurons)
    
    # Training 
    H = tanh(X_train)
    H_inv = np.linalg.pinv(H)
    W2 = np.matmul(H_inv, y_train)
    
    # Testing
    H_T = tanh(X_val)
    y_pred = np.matmul(H_T, W2)
    
    accuracy = 0

    for p in range(len(y_pred)):
        if np.argmax(y_pred[p]) == np.argmax(y_val[p]):
            accuracy += 1
    accuracy = accuracy / len(y_val)
    print(accuracy)

    X_tot[0 : 358] = X_val
    X_tot[358 : ] = X_train
    y[0 : 358] = y_val
    y[358 : ] = y_train

