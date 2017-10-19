import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU,SimpleRNN


class ANNModel(object):

    def __init__(self, inputDim, hiddenNum, outputDim):

        self.inputDim = inputDim
        self.hiddenNum = hiddenNum
        self.outputDim = outputDim
        self.buildModel()

    def buildModel(self):

        self.model = Sequential()
        self.model.add(Dense(self.hiddenNum, input_dim=self.inputDim, activation='relu'))
        #self.model.add(Dense(self.hiddenNum//2, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='rmsprop')

    def train(self, trainX, trainY, epoch, batchSize):
        self.model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=1)

    def predict(self,testX):
        pred = self.model.predict(testX)
        return pred.reshape(-1)