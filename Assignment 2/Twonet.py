## Multilayer perceptron neural network (2 hidden layers)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat_contents = loadmat('data5.mat')
data = mat_contents['x']
np.random.shuffle(data)

def init_data():
    X = np.array(data[:2148, :-1], dtype = float)
    y = np.array(data[:2148, -1], dtype = int)
    X = (X - X.mean(axis = 0))/X.std(axis = 0)
    return X, y

def affine_forward(x, w, b):
    z = x.dot(w) + b
    cache = (x, w, b)
    return z, cache

def relu_forward(x):
    a = x
    a[a<=0] = 0
    cache = x
    return a, cache

def affine_backward(dout, cache):
    x, w, b = cache
    db = np.sum(dout, axis = 0)
    dw = x.T.dot(dout)
    dx = dout.dot(w.T)
    return dx, dw, db

def relu_backward(dout, cache):
    x = cache
    dx = None
    dx = np.ones(x.shape)
    dx[x<=0] = 0
    dx = dx * dout
    return dx

class Twonet(object):

    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, std=1e-4):
        self.W1 = std * np.random.randn(input_size, hidden_size1)
        self.b1 = np.zeros(hidden_size1)
        self.W2 = std * np.random.randn(hidden_size1, hidden_size2)
        self.b2 = np.zeros(hidden_size2)
        self.W3 = std * np.random.randn(hidden_size2, num_classes)
        self.b3 = np.zeros(num_classes)

    def loss(self, X, y = None, reg = 0.0):
        N, D = X.shape
        scores = None
        z1, af_cache1 = affine_forward(X, self.W1, self.b1)
        h1, relu_cache1 = relu_forward(z1)
        z2, af_cache2 = affine_forward(h1, self.W2, self.b2)
        h2, relu_cache2 = relu_forward(z2)
        z3, af_cache3 = affine_forward(h2, self.W3, self.b3)
        scores = z3

        if y is None:
            return scores

        loss = None
        scores -= scores.max()
        scores_exp = np.exp(scores)
        correct_scores = scores[range(N), y]
        correct_scores_exp = np.exp(correct_scores)
        loss = np.sum(-np.log(correct_scores_exp / np.sum(scores_exp, axis = 1))) / N
        loss += 0.5 * reg * (np.sum(self.W1 * self.W1) + \
            np.sum(self.W2 * self.W2) + np.sum(self.W3 * self.W3))

        num = correct_scores_exp
        denom = np.sum(scores_exp, axis = 1)
        mask = (np.exp(z3)/denom.reshape(scores.shape[0],1))
        mask[range(N),y] = -(denom - num)/denom
        mask /= N
        dz3 = mask

        dh2, dw3, db3 = affine_backward(dz3, af_cache3)
        dz2 = relu_backward(dh2, relu_cache2)
        dh1, dw2, db2 = affine_backward(dz2, af_cache2)
        dz1 = relu_backward(dh1, relu_cache1)
        dx, dw1, db1 = affine_backward(dz1, af_cache1)
        
        dw3 = dw3 + reg * self.W3
        dw2 = dw2 + reg * self.W2
        dw1 = dw1 + reg * self.W1

        wgrad = (dw1, dw2, dw3)
        bgrad = (db1, db2, db3)

        return loss, wgrad, bgrad

    def train(self, X, y, X_val, y_val, alpha = 1e-3, alpha_decay = 0.95,\
         reg = 5e-6, num_iters = 100, batch_size = 200):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):

            ind = np.random.choice(num_train, batch_size)
            X_batch = X[ind,:]
            y_batch = y[ind]
            
            loss, wgrad, bgrad = self.loss(X_batch, y = y_batch, reg = reg)
            loss_history.append(loss)

            dw1, dw2, dw3 = wgrad
            db1, db2, db3 = bgrad

            self.W1 -= alpha * dw1
            self.W2 -= alpha * dw2
            self.W3 -= alpha * dw3
            self.b1 -= alpha * db1
            self.b2 -= alpha * db2
            self.b3 -= alpha * db3


            if it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))


            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                alpha *= alpha_decay


        return {'loss_history' : loss_history, 'train_acc_history' : \
            train_acc_history, 'val_acc_history' : val_acc_history}


    def predict(self, X):
        y_pred = np.argmax(self.loss(X), axis = 1)
        return y_pred



input_size = 72
hidden_size1 = 30
hidden_size2 = 30
num_classes = 2
num_inputs = 1790
std = 0.1
alpha = 0.3
batch_size = 1024
reg = 1e-2
num_iters = 5000

X_tot, y_tot = init_data()

train_acc , val_acc = 0, 0
losses = np.empty((6, num_iters))
val_accs = []
train_accs = []

for k in range(6):
    
    X = X_tot[0 : 1790]
    y = y_tot[0 : 1790]
    X_val = X_tot[1790 :]
    y_val = y_tot[1790 :]
    
    Net = Twonet(input_size, hidden_size1, hidden_size2, num_classes, std)
    print("Validation fold : " , k + 1)
    stats = Net.train(X, y, X_val, y_val, num_iters = num_iters,\
         alpha = alpha, batch_size = batch_size, reg = 0.0)
    losses[k] = np.asarray(stats['loss_history'])
    val_accs = np.asarray(stats['val_acc_history'])
    train_accs = np.asarray(stats['train_acc_history'])
    train_acc += train_accs
    val_acc += val_accs


    X_tot[0 : 358] = X_val
    X_tot[358 : ] = X
    y_tot[0 : 358] = y_val
    y_tot[358 : ] = y

train_acc /= 6
val_acc /= 6

print(train_acc[-1], val_acc[-1])
loss_hist = np.mean(losses, axis = 0)

plt.subplot(2, 1, 1)
plt.plot(loss_hist)
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='val')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.tight_layout
plt.show()

y_pred = Net.predict(X_val)
TP, TN, FP, FN = 0, 0, 0, 0
for i in range(len(y_val)):
    if y_pred[i] == 0 and  y_val[i] == 0:
        TN += 1
    elif y_pred[i] == 1 and  y_val[i] == 0:
        FP += 1
    elif y_pred[i] == 0 and  y_val[i] == 1:
        FN += 1
    elif y_pred[i] == 1 and  y_val[i] == 1:
        TP += 1

print(TP, FP)
print(FN, TN)

accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print("accuracy = ", accuracy, "sensitivity = ", sensitivity,\
     "specificity = ", specificity)