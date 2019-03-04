import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


class MLP_Model(object):

    def __init__(self, inputDim, hiddenNum, outputDim, lr):

        self.inputDim = inputDim
        self.hiddenNum = hiddenNum
        self.outputDim = outputDim
        self.opt = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-06)
        self.buildModel()

    def buildModel(self):

        self.model = Sequential()
        self.model.add(Dense(self.hiddenNum, input_dim=self.inputDim, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer=self.opt)

    def train(self, trainX, trainY, epoch, batchSize):
        self.model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=1)

    def predict(self, testX):
        pred = self.model.predict(testX)
        return pred