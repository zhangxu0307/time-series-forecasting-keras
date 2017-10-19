from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU,SimpleRNN
from keras.layers import Dropout

class statefulRNNsModel(object):

    def __init__(self, inputDim, hiddenNum, outputDim,  batchSize, lag ,unit):

        self.inputDim = inputDim
        self.hiddenNum = hiddenNum
        self.outputDim = outputDim
        self.batchSize = batchSize
        self.lag = lag
        self.buildModel(unit)

    def buildModel(self, unit="GRU"):

        self.model = Sequential()
        if unit == "GRU":
            self.model.add(GRU(self.hiddenNum, batch_input_shape=(self.batchSize, self.lag, self.inputDim), stateful=True))
        elif unit == "LSTM":
            self.model.add(LSTM(self.hiddenNum, batch_input_shape=(self.batchSize, self.lag, self.inputDim), stateful=True))
        elif unit == "RNN":
            self.model.add(SimpleRNN(self.hiddenNum, batch_input_shape=(self.batchSize, self.lag, self.inputDim), stateful=True))
        # self.model.add(Dense(self.hiddenNum/2))
        self.model.add(Dense(self.outputDim))
        self.model.compile(loss='mean_squared_error', optimizer='rmsprop')

    def train(self, trainX, trainY, epoch, batchSize):

        for i in range(epoch):
            self.model.fit(trainX, trainY, nb_epoch=1, batch_size=batchSize, verbose=2, shuffle=False)
            self.model.reset_states()
        #self.model.fit(trainX, trainY, nb_epoch=epoch, batch_size=batchSize, verbose=-1)

    def predict(self, testX, batchSize):

        pred = self.model.predict(testX, batch_size=batchSize)
        self.model.reset_states()
        #pred = self.model.predict(testX)
        return pred.reshape(-1)