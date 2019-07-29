from brianthebrain import *

svm = brian('scale','rbf',1.0)

X = [[0, 0], [1, 1]]
y = [0, 1]
svm.load(X,y)
svm.train()
print(svm.predict([[2., 2.]]))