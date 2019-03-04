#encoding=utf-8

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU,SimpleRNN
from keras.layers import Dropout
from keras.regularizers import l2
import numpy as np
from keras import optimizers


class RNNsModel(object):

    # 初始化RNN模型参数，包括输入维度、隐藏层维度、输出维度、cell单元类型
    def __init__(self, inputDim, hiddenNum, outputDim, unit, lr):

        self.inputDim = inputDim
        self.hiddenNum = hiddenNum
        self.outputDim = outputDim
        self.opt = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-06)
        self.buildModel(unit)

    # 建立RNN模型
    def buildModel(self, unit="GRU"):

        self.model = Sequential()
        if unit == "GRU":
            self.model.add(GRU(self.hiddenNum, input_shape=(None, self.inputDim)))
        elif unit == "LSTM":
            self.model.add(LSTM(self.hiddenNum, input_shape=(None, self.inputDim)))
        elif unit == "RNN":
            self.model.add(SimpleRNN(self.hiddenNum, input_shape=(None, self.inputDim)))
        self.model.add(Dense(self.outputDim))
        self.model.compile(loss='mean_squared_error', optimizer=self.opt, metrics=["mean_absolute_percentage_error"])

    # 正常训练，可以设置训练轮数、batch大小
    def train(self, trainX, trainY, epoch, batchSize):
        self.model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=1, validation_split=0.0)

    # 预测
    def predict(self,testX):
        pred = self.model.predict(testX)
        return pred

    # 变长度预测，对于同一个数据点，分别预测不同lag下的结果，目前是简单取平均
    def predictVarLen(self, vtestX, minLen, maxLen, step):
        lagNum = (maxLen-minLen) // step + 1
        predAns = []
        pred = self.model.predict(vtestX)
        for i in range(0, len(pred), lagNum):
            predAns.append(np.mean(pred[i:i+lagNum]))
        return np.array(predAns)



