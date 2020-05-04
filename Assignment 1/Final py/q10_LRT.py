##Likelihood Ratio Test binary classification

import numpy as np 
import pandas as pd 

# Preparing training and testing datasets
df = pd.read_excel( 'data3.xlsx', header=None)
df = df.to_numpy()
class1, class2 = df[0:50, :], df[50:100, :]
X = np.zeros([60, 5], dtype=float)
test = np.zeros([40, 5], dtype=float)
X[0:30,:], X[30:60, :] = class1[0:30, :], class2[0:30, :]
test[0:20, :], test[20:40, :] = class1[0:20, :], class2[0:20, :]

# calculating means of feature vectors
means = [np.mean(X[0:30, :-1], axis=0), np.mean(X[30:60, :-1], axis=0)]

# calculation of covariance matrix for each class 
cov = [np.cov((X[0:30, 0:4]).T), np.cov((X[30:60, 0:4]).T)]

# Calculate priors
priors = np.array([0.0, 0.0])
l = np.array(X[:, -1], dtype='int') - 1
priors[l] += 1
priors = priors/len(X)

# likelihood function
def likelihood(x, k):
    m = np.dot((x-means[k-1]).T, np.dot(np.linalg.inv(cov[k-1]), x-means[k-1]))
    p = np.exp(-0.5 * m) / ((2 * np.pi) * np.linalg.det(cov[k-1]))
    return p

ratio = priors[1]/priors[0]
confusion_matrix = np.zeros([2,2])

for row in test:

    l = [likelihood(row[:-1], 1), likelihood(row[:-1], 2)]
    r = l[0]/l[1]
    
    # Calculate predicted class
    p = 1 if r>ratio else 2
    
    # Update confusion matrix
    confusion_matrix[int(row[-1])-1, p-1] += 1
  
sensitivity = confusion_matrix[0,0]/np.sum(confusion_matrix[0])
specificity = confusion_matrix[1,1]/np.sum(confusion_matrix[1])
accuracy = (confusion_matrix[0,0] + confusion_matrix[1,1])/np.sum(confusion_matrix)

print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)
print("Accuracy: ", accuracy)
print(confusion_matrix)