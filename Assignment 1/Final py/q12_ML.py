##Maximum Likelihood decision rule multiclass classification

import pandas as pd 
import numpy as np

# Preparing training and testing datasets, normalising data
df = pd.read_excel('data4.xlsx', header=None)
df.iloc[:,0:7]=(df.iloc[:,0:7]-(df.iloc[:,0:7]).mean())/(df.iloc[:,0:7]).std()
df = df.to_numpy()
class1, class2, class3 = df[:50], df[50:100], df[100:]
np.random.shuffle(class1)
np.random.shuffle(class2)
np.random.shuffle(class3)
X = np.concatenate((class1[:int(0.7*len(class1))], class2[:int(0.7*len(class2))], class3[:int(0.7*len(class3))]))
test = np.concatenate((class1[int(0.7*len(class1)):], class2[int(0.7*len(class2)):], class3[int(0.7*len(class3)):]))
xlen = len(X)
class_train = [(X[:int(xlen/3), :7]).T, (X[int(xlen/3):2*int(xlen/3), :7]).T,
                (X[2*int(xlen/3):, :7]).T]

# calculate covariance matrices
cov = [np.cov(class_train[0]), np.cov(class_train[1]), np.cov(class_train[2])]

# calculate means for each class 
means = [np.mean(class_train[0], axis=1), np.mean(class_train[1], axis=1), np.mean(class_train[2], axis=1)]

# Likelihood function 
def likelihood(x, c):
    den = np.sqrt(2 * np.pi * np.linalg.det(cov[c])) 
    pow = -0.5 * np.dot(np.dot((x-means[c]).T,np.linalg.inv(cov[c])), (x-means[c]))
    return float(np.exp(pow) / den)

# initialize confusion matrix
confusion_matrix = np.zeros((3,3))

# testing loop
for row in test:

    # calculate predicted class
    l = np.argmax([likelihood(row[:7], i) for i in range(3)]) + 1

    # update confusion matrix
    confusion_matrix[int(row[-1])-1, l-1] += 1

c1 = confusion_matrix[0,0] / np.sum(confusion_matrix[0])
c2 = confusion_matrix[1,1] / np.sum(confusion_matrix[1])
c3 = confusion_matrix[2,2] / np.sum(confusion_matrix[2])
total_accuracy = (confusion_matrix[0,0] + confusion_matrix[1,1] + confusion_matrix[2,2]) / np.sum(confusion_matrix)

print("Total accuracy: ", total_accuracy)
print("Class 1 accuracy: ", c1)
print("Class 2 accuracy: ", c2)
print("Class 3 accuracy: ", c3)