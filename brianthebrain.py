from sklearn import svm
import pickle

class brian:
    def __init__(self, gamma, kern, c):
        self.svm = clf = svm.SVC(C=c, kernel=kern, gamma=gamma)
    def load(self, X, y):
        self.X_train = X
        self.y_train = y
    def train(self):
        self.svm.fit(self.X_train, self.y_train)
        #this is the equivelent of svm.fit(X, y)
    def predict(self, data):
        return self.svm.predict(data)
        #demo data: [[2., 2.]]
    def load():
        self.data = pickle.
#X = [[0, 0], [1, 1]]
#y = [0, 1]