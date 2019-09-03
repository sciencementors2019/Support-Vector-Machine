import itertools
import numpy as np
from scipy import stats
from scipy import linalg
import pylab as pl
from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split
import timeit
from dataread import *


#reading the data from the excel


n_samples = 14970 #this is the number of images in our dataset
n_features = 34 #num of features per feature vector
data = np.empty((n_samples, n_features), dtype=np.int) #creates a 2d array with the first dimension number of samples and second number of features thus the matrix is 502 x 10 
target = np.empty((n_samples), dtype=np.int) #creates a 1d array that is of length sample number ie 502
qid = np.empty((n_samples), dtype=np.int)

trainDat = trainData()
trainLabel = trainLabels()
# print(trainDat)
for y in range(n_samples-1):
    for x in range(1,n_features):
        data[y][x] = trainDat[x][y]# X axis is first index and Y is second!!!!! why numpy...

for i in range(n_samples-1):
    if trainLabel[0][i] == 'triangle':# X axis is first index and Y is second!!!!! why numpy...
        target[i] = 0  
    if trainLabel[0][i] == 'circle':# X axis is first index and Y is second!!!!! why numpy...
        target[i] = 1
    if trainLabel[0][i] == 'square':# X axis is first index and Y is second!!!!! why numpy...
        target[i] = 2  
    if trainLabel[0][i] == 'star':# X axis is first index and Y is second!!!!! why numpy...
        target[i] = 3       
     
print(target)

X_train, X_test, y_train, y_test = train_test_split(data, target, train_size = 0.8)
# split into train and test set
# cv = cross_validate.StratifiedShuffleSplit(target, test_size=.8) #TODO
# train, test = iter(cv).next() #creates a test mask array and a train mask array. these arrays are the indexes of x and y that are assigned to each set

# X_train, y_train = data[train], target[train] #using the train index mask create an copy array of both x and y respectivly that is only filled with the training items
# X_test, y_test, qid_test = data[test], target[test], qid[test] #using the test index mask create an copy array of both x and y respectivly that is only filled with the test items

print("SVM starting")		
#SVM        
#clf = svm.SVC(kernel='linear', C=.1, class_weight='auto', cache_size=1000) #inits the SVM
clf = svm.SVC(kernel='linear', C=.1, cache_size=1000, verbose=True) #inits the SVM
print("SVM created")
clf.fit(X_train, y_train) #fits the ranked data to the vector space of the svm
print("SVM fit")	
coef = clf.coef_.ravel() / linalg.norm(clf.coef_) #finds the coefficient of the seperation of the ranked trained set