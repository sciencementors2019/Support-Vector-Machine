import itertools
import numpy as np
from scipy import stats
from scipy import linalg
import pylab as pl
from sklearn import svm, linear_model, cross_validation
import timeit

start = timeit.default_timer()

print("Starting ----")
print("Note Check reports, samples and features are set in program")

f = open('att.txt', 'r') #loads feature vector read in file for a specfic query
fs = open('class.txt', 'r') #loads in the trained class file for the specific query
fss = open('pairIds.txt', 'r') #loads in a parallel array to att.txt that contains the ids of the pair to have tracability for evaluation 
n_reports = 2503
n_samples = 125150 #253512 #number of individual items in query
n_features = 101 #num of features per feature vector
n_queryPerSample = 50
data = np.empty((n_samples, n_features), dtype=np.int) #creates a 2d array with the first dimension number of samples and second number of features thus the matrix is 502 x 10 
target = np.empty((n_samples), dtype=np.int) #creates a 1d array that is of length sample number ie 502
qid = np.empty((n_samples), dtype=np.int)
ids = []

print("Load complete")
	
#read feature vector into data matrix
count = 0;	
for line in f:
	fields = line.split(';')
	subcount = 0; 
	#qid[count] = fields[0];
	for i in range(1 , len(fields)):
		if fields[i] == '\n':
			continue
		data[count, subcount] = (float(fields[i]))
		subcount = subcount + 1;
	count = count + 1;
	
#read class target into target matrix 
count = 0;
for line in fs:
	target[count] = line[:-1]
	count = count + 1

#read pairIds into ids matrix 
for line in fss:
	ids.append(line)
	
#query mask to hold original id
for x in range(n_samples):
	fields = ids[x].split(',')
	qid[x] = fields[0] #x / (n_queryPerSample)		
	
#close these files
f.close()
fs.close()
fss.close()

print ("Read Done")
# split into train and test set
cv = cross_validation.StratifiedShuffleSplit(target, test_size=.8) #TODO
train, test = iter(cv).next() #creates a test mask array and a train mask array. these arrays are the indexes of x and y that are assigned to each set

X_train, y_train = data[train], target[train] #using the train index mask create an copy array of both x and y respectivly that is only filled with the training items
X_test, y_test, qid_test = data[test], target[test], qid[test] #using the test index mask create an copy array of both x and y respectivly that is only filled with the test items

print ("Split Done")

#Xp, yp, diff = map(np.asanyarray, (Xp, yp, diff)) #maps xp,yp and diff to an numpy array

#to make reading stats easier, not really needed
valset = []
for val in y_test:
    if val == -1:
        valset.append(0)
    else:
        valset.append(1)
		
		
print("SVM starting")		
#SVM        
#clf = svm.SVC(kernel='linear', C=.1, class_weight='auto', cache_size=1000) #inits the SVM
clf = svm.SVC(kernel='linear', C=.1, cache_size=1000) #inits the SVM
print("SVM created")
clf.fit(X_train, y_train) #fits the ranked data to the vector space of the svm
print("SVM fit")	
coef = clf.coef_.ravel() / linalg.norm(clf.coef_) #finds the coefficient of the seperation of the ranked trained set


print("Writing")

#write feature coeefficents
f4 = open('coefficents.txt', 'w')
f4.write("Mean score test:")
f4.write('%s' % clf.score(X_test, valset))
f4.write("coef:\n")
f4.write('%s' % clf.coef_)
f4.write("\nadjusted:\n")
f4.write('%s' % coef)
f4.close()

#Prediction set write, ie binarary classification
predset = clf.predict(X_test)
f1 = open('predset.txt', 'w')
for q in predset:
    f1.write('%s\n' % q)
f1.close()

#truth set write	
f2 = open('truthset.txt', 'w')
for q in valset:
    f2.write('%s\n' % q)
f2.close()

#write decision function for each feature vector
#output: pair ids, post query truth, query mask, decision function (length to margin)
f3 = open('decision.txt', 'w')
decisionSet = clf.decision_function(X_test)
for q in range(decisionSet.shape[0]):
    f3.write('%s;%s;%s\n' % (y_test[q], qid_test[q], decisionSet[q]))
f3.close()

print("Done")

stop = timeit.default_timer()
print("Execution Time: ")
print(stop - start)
#tau, _ = stats.kendalltau(
#        np.dot(X_test, coef), y_test) #utilizing the coefficient, apply the devide to the test data to classify the x_test set
#print('Kendall correlation coefficient: %.5f' % (tau))	
