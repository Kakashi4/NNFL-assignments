## Convolutional Neural Network classifier
from scipy.io import loadmat
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, AveragePooling1D, Conv1D, Flatten
from keras import optimizers
from keras.utils import to_categorical
from keras import regularizers
from sklearn.preprocessing import normalize


# Load data and shuffle
mat_features = loadmat('data_for_cnn.mat')
mat_labels = loadmat('y.mat')
features = mat_features['ecg_in_window']
labels = mat_labels['label']

data = np.zeros((1000, 1001))
data[:, :-1] = features
data[:, -1] = labels.flatten()

np.random.shuffle(data)

X = data[:, :-1]
y = data[:, -1]
X = X.reshape(X.shape[0], X.shape[1], 1)
y = to_categorical(y, num_classes = 2)

# Hyperparameters
reg = 0.9
epochs = 1000
learning_rate = 1e-3
batch_size = 256
holdout = 0.3

# Define model
model = Sequential()
model.add(Conv1D(50, 30, padding = 'same', input_shape = (1000, 1), kernel_regularizer=regularizers.l2(reg)))
model.add(AveragePooling1D())
model.add(Conv1D(50, 50, padding = 'same', kernel_regularizer=regularizers.l2(reg)))
model.add(AveragePooling1D())
model.add(Flatten())
model.add(Dense(50, activation = 'relu', kernel_regularizer=regularizers.l2(reg)))
model.add(Dense(30, activation = 'relu', kernel_regularizer=regularizers.l2(reg)))
model.add(Dense(2, activation = 'softmax'))

# Train model and check validation accuracy
sgd = optimizers.SGD(lr = learning_rate)
model.compile(loss = 'mean_squared_error', optimizer = sgd, metrics = ['accuracy'])
model.fit(X, y, validation_split = holdout, epochs = epochs, batch_size = batch_size, use_multiprocessing = True)
