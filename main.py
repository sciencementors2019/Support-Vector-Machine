from brianthebrain import *
import imageProcessor

imageBank = imageProcessor.newbundle
svm = brian('scale','rbf',1.0)

X = [[2, 2], [1, 1]]
y = [0, 1]
svm.load(X,y)
svm.train()
print(svm.predict([[0, 0]]))
