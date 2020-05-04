## Convolutional Autoencoder
from scipy.io import loadmat
import numpy as np
from keras import Sequential
from keras import optimizers
from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPooling1D,UpSampling1D,Reshape,Flatten
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Load, shuffle and normalize data
mat_features = loadmat('data_for_cnn.mat')
features = mat_features['ecg_in_window']

data = np.zeros((1000, 1001))

np.random.shuffle(data)

X = data[:, :-1]
X = preprocessing.normalize(X, axis= 0)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Input - Convolution layer - Max-Pooling layer - FC - Upsampling layer - Transpose convolution layer

model = Sequential()
model.add(Conv1D(32, 5, activation= 'relu' ,input_shape = (1000, 1), padding= 'same'))
model.add(MaxPooling1D(4, padding= 'same'))
model.add(Flatten())
model.add(Dense(500, activation = 'sigmoid'))
model.add(UpSampling1D(2))
model.add(Reshape((1000, 1)))
model.add(Conv1D(1, 5, activation='sigmoid', padding='same'))
autoencoder = model
autoencoder.summary()
opt = optimizers.Adam(lr=0.01)
autoencoder.compile(optimizer= opt, loss='mse')

# Train model
history = autoencoder.fit(X, X, epochs=500, batch_size=256, shuffle=True)

# Plot training loss
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.savefig('loss.png')