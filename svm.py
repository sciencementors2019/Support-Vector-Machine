import itertools
import numpy as np
from scipy import stats
from scipy import linalg
import pylab as pl
from sklearn import svm, linear_model
from sklearn.model_selection import cross_validate
import timeit

# split into train and test set
cv = cross_validation.StratifiedShuffleSplit(target, test_size=.8) #TODO
train, test = iter(cv).next() #creates a test mask array and a train mask array. these arrays are the indexes of x and y that are assigned to each set

X_train, y_train = data[train], target[train] #using the train index mask create an copy array of both x and y respectivly that is only filled with the training items
X_test, y_test, qid_test = data[test], target[test], qid[test] #using the test index mask create an copy array of both x and y respectivly that is only filled with the test items

print("SVM starting")		
#SVM        
#clf = svm.SVC(kernel='linear', C=.1, class_weight='auto', cache_size=1000) #inits the SVM
clf = svm.SVC(kernel='linear', C=.1, cache_size=1000) #inits the SVM
print("SVM created")
clf.fit(X_train, y_train) #fits the ranked data to the vector space of the svm
print("SVM fit")	
coef = clf.coef_.ravel() / linalg.norm(clf.coef_) #finds the coefficient of the seperation of the ranked trained set