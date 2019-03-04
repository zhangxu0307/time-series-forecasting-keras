import sklearn as sk
from sklearn.svm import SVR


class SVRModel(object):

    def __init__(self, C=1.0, kernel='rbf', epsilon=0.2, shrinking=True):
        self.C = C
        self.kernel = kernel
        self.shrinking = shrinking
        self.epsilon = epsilon
        self.buildModel()

    def buildModel(self,):
        self.model = SVR(C=self.C, cache_size=200, coef0=0.0, degree=3, epsilon=self.epsilon, gamma='auto',
            kernel=self.kernel, max_iter=-1, shrinking=self.shrinking, tol=0.001, verbose=False)

    def train(self, trainX, trainY):
        self.model.fit(trainX, trainY)

    def predict(self, testX):
        pred = self.model.predict(testX)
        return pred
