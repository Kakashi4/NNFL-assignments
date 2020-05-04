##Stacked autoencoder based deep neural network

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize

#Load data, shuffle and normalize
mat_contents = loadmat('data5.mat')
data = mat_contents['x']
np.random.shuffle(data)


def init_data():
    X = np.array(data[ : , :-1], dtype = float)
    y = np.array(data[ : , -1], dtype = int)
    X = (X - X.mean(axis = 0))/X.std(axis = 0)
    return X, y

X, y = init_data()

#Hold out method of model evaluation
X_train, y_train = X[ :int(0.7 * len(X))], y[ :int(0.7 * len(X))]
X_val, y_val = X[ int(0.7 * len(X)): ], y[ int(0.7 * len(X)): ]

alpha = 0.5

#Sigmoid activation function
def sigmoid(x, derivative=False):
        if (derivative == True):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

#Neural network class
class NeuralNetwork(object):
    def __init__(self, sizes):
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.W = {}
        self.a = {}
        self.b = {}
        
        #Initialize Weights
        for i in range(1, self.num_layers):
            self.W[i] = np.random.randn(self.sizes[i-1], self.sizes[i])
            
        #Initialize biases
        for i in range(1, self.num_layers):
            self.b[i] = np.random.randn(self.sizes[i], 1)
        
        #Initialize activations
        for i in range(1, self.num_layers):
            self.a[i] = np.zeros([self.sizes[i], 1])
        
    #Forward pass to compute scores
    def forward_pass(self, X):
        
        self.a[0] = X
        
        for i in range(1, self.num_layers):
            self.a[i] = sigmoid(np.dot(self.W[i].T, self.a[i-1]) + self.b[i])

        return self.a[self.num_layers-1] 
    
    #Backward pass to update weights
    def backward_pass(self, X, Y, output):
        
        self.d = {}
        self.d_output = (Y - output) * sigmoid(output, derivative=True)
        self.d[self.num_layers-1] = self.d_output
        
        #Derivatives of the layers wrt loss
        for i in range(self.num_layers-1, 1, -1):
            self.d[i-1] = np.dot(self.W[i], self.d[i]) * sigmoid(self.a[i-1], derivative=True)
        
        #Updating weights
        for i in range(1, self.num_layers-1):
            self.W[i] += alpha * np.dot(self.a[i-1], self.d[i].T)
            
        #Updating biases
        for i in range(1, self.num_layers-1):
            self.b[i] += alpha * self.d[i]

    #Training helper function   
    def train(self, X, Y):
        X = np.reshape(X, (len(X), 1))
        output = self.forward_pass(X)
        self.backward_pass(X, Y, output)

    #Get weights    
    def get_W(self):
        return self.W
    
    #Load specified weights
    def load_W(self, W):
        self.W = W

    #Scores computation for given input    
    def get_a(self, x):
        x = np.reshape(x, (len(x), 1))
        self.forward_pass(x)
        return self.a
    
    #Helper function for autoencoder chaining
    def load_a(self, a):
        self.a = a

    

#Loss function
def calc_loss(NN,x ,y):
    
    loss = 0
    for i in range(len(x)):
        x_ = np.reshape(x[i], (len(x[i]), 1))
        loss += 0.5 / len(x) * np.sum((y[i] - NN.forward_pass(x_)) ** 2)
    
    return loss

#Network initialization
autoencoder1 = NeuralNetwork([72, 60, 72])
autoencoder2 = NeuralNetwork([60,40,60])
autoencoder3 = NeuralNetwork([40, 30, 40])
NN = NeuralNetwork([72,60,40,30, 1])

#Autoencoder 1 pretraining
for i in range(200):
    for j, row in enumerate(X_train):
        row = np.reshape(row, (72,1))
        autoencoder1.train(row, row)
        
    loss = calc_loss(autoencoder1, X_train, X_train)
    print("Epoch {}, Loss {}".format(i, loss))
    
#Scores computation for autoencoder 1
autoencoder2_input = []

for row in X_train:
    autoencoder2_input.append(autoencoder1.get_a(row)[1])

autoencoder2_input = np.array(autoencoder2_input)


#Autoencoder 2 pretraining
for i in range(200):
    for j, row in enumerate(autoencoder2_input):
        row = np.reshape(row, (60,1))
        autoencoder2.train(row, row)
        
    loss = calc_loss(autoencoder2, autoencoder2_input, autoencoder2_input)
    print("Epoch {}, Loss {}".format(i, loss))


#Scores computation for autoencoder 2
autoencoder3_input = []

for row in autoencoder2_input:
    autoencoder3_input.append(autoencoder2.get_a(row)[1])

autoencoder3_input = np.array(autoencoder3_input)

#Autoencoder 3 pretraining
for i in range(200):
    for j, row in enumerate(autoencoder3_input):
        row = np.reshape(row, (40,1))
        autoencoder3.train(row, row)
        
    loss = calc_loss(autoencoder3, autoencoder3_input, autoencoder3_input)
    print("Epoch {}, Loss {}".format(i, loss))

#Final network weight initialization
W1 = autoencoder1.get_W()[1]
W2 = autoencoder2.get_W()[1]
W3 = autoencoder3.get_W()[1]
W_final = {}
W_final[1] = W1
W_final[2] = W2
W_final[3] = W3
W_final[4] = np.random.randn(30, 1)
NN.load_W(W_final)

#Training loop
for i in range(500):
    print("Epoch: ", i)

    for j in range(len(X_train)):
        NN.train(X_train[j], y_train[j])

TP,TN,FP,FN = 0,0,0,0

for i in range(len(X_val)):

    x = np.reshape(X_val[i], (len(X_val[i]), 1))
    x = NN.forward_pass(x)
    p = 0 if x[0] < 0.5 else 1

    if p == 1 and y_val[i] == 1:
        TP += 1
    elif p == 0 and y_val[i] == 0:
        TN += 1
    elif p == 1 and y_val[i] == 0:
        FP += 1
    elif p == 0 and y_val[i] == 1:
        FN += 1

print(TP, FP)
print(FN, TN)

accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print("accuracy = ", accuracy, "sensitivity = ", sensitivity, "specificity = ", specificity)

