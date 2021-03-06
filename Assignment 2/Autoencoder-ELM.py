## Stacked autoencoder with ELM classifier

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
    X = normalize(X, axis = 0)
    return X, y

#Convert to one-hot
X, y_ = init_data()
y = np.zeros((len(y_), 2))
for i in range(len(y_)):
    if y_[i]==1:
        y[i,1] = 1.0
    elif y_[i]==0:
        y[i,0] = 1.0

#Hold out method of model evaluation
X_train, y_train = X[ :int(0.7 * len(X))], y[ :int(0.7 * len(X))]
X_val, y_val = X[ int(0.7 * len(X)): ], y[ int(0.7 * len(X)): ]

alpha = 0.5

#Sigmoid activation function
def sigmoid(x, derivative=False):
        if (derivative == True):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

#Tanh activation function
def tanh(x):
    return np.tanh(x)

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

#Autoencoder 1 pretraining
for i in range(500):
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
for i in range(500):
    for j, row in enumerate(autoencoder2_input):
        row = np.reshape(row, (60,1))
        autoencoder2.train(row, row)
        
    loss = calc_loss(autoencoder2, autoencoder2_input, autoencoder2_input)
    print("Epoch {}, Loss {}".format(i, loss))

#Inputs to ELM

elm_input = []
for row in autoencoder2_input:
    elm_input.append(autoencoder2.get_a(row)[1])   
elm_input = np.array(elm_input)

#parameters for ELM
elm_neurons = 300
output_neurons = 2
W_elm = np.random.randn(elm_input.shape[1], elm_neurons)

#ELM Training
np.random.seed(1)
elm_input = np.reshape(elm_input, (1503, 40))
H = np.matmul(elm_input, W_elm)
H = tanh(H)
H_inv = np.linalg.pinv(H)
W_final = np.matmul(H_inv, y_train) 

#Testing on validation dataset

#Autoencoder 1 forward pass
layer1_out = []

for i, row in enumerate(X_val):
    act = autoencoder1.get_a(row)[1]
    layer1_out.append(act)
    
layer1_out = np.array(layer1_out)
layer1_out = np.reshape(layer1_out, (645, 60))

#Autoencoder 2 forward pass
layer2_out = []

for i, row in enumerate(layer1_out):
    act = autoencoder2.get_a(row)[1]
    layer2_out.append(act)
    
layer2_out = np.array(layer2_out)
layer2_out = np.reshape(layer2_out, (645, 40))

#ELM forward pass
H_T = np.matmul(layer2_out, W_elm)
H_T = tanh(H_T)
y_pred = np.matmul(H_T, W_final)

TP,TN,FP,FN = 0,0,0,0

for i in range(len(y_pred)):
    
    if np.argmax(y_pred[i]) == 1 and np.argmax(y_val[i]) == 1:
        TP += 1
    elif np.argmax(y_pred[i]) == 0 and np.argmax(y_val[i]) == 0:
        TN += 1
    elif np.argmax(y_pred[i]) == 1 and np.argmax(y_val[i]) == 0:
        FP += 1
    elif np.argmax(y_pred[i]) == 0 and np.argmax(y_val[i]) == 1:
        FN += 1

print(TP, FP)
print(FN, TN)

accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print("accuracy = ", accuracy, "sensitivity = ", sensitivity, "specificity = ", specificity)
