#encoding=utf-8

from models import RNNs
import util
import eval
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv
import numpy as np


def RNN_forecasting(dataset, lookBack, lr, inputDim=1, hiddenNum=64, outputDim=1, unit="GRU", epoch=20,
                    batchSize=30, varFlag=False, minLen=15, maxLen=30, step=5):

    # 归一化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 分割序列为样本,并整理成RNN的输入形式
    train, test = util.divideTrainTest(dataset)

    trainX = None
    trainY = None
    vtrainX = None
    vtrainY = None
    testX = None
    testY = None
    vtestX = None
    vtestY = None

    # 构建模型并训练
    RNNModel = RNNs.RNNsModel(inputDim, hiddenNum, outputDim, unit, lr)
    if varFlag:
        vtrainX, vtrainY = util.createVariableDataset(train, minLen, maxLen, step)
        vtestX, vtestY = util.createVariableDataset(test, minLen, maxLen, step)
        print("trainX shape is", vtrainX.shape)
        print("trainY shape is", vtrainY.shape)
        print("testX shape is", vtestX.shape)
        print("testY shape is", vtestY.shape)
        RNNModel.train(vtrainX, vtrainY, epoch, batchSize)
    else:
        trainX, trainY = util.createSamples(train, lookBack)
        testX, testY = util.createSamples(test, lookBack)
        print("trainX shape is", trainX.shape)
        print("trainY shape is", trainY.shape)
        print("testX shape is", testX.shape)
        print("testY shape is", testY.shape)
        RNNModel.train(trainX, trainY, epoch, batchSize)

    # 预测
    if varFlag:
        trainPred = RNNModel.predictVarLen(vtrainX, minLen, maxLen, step)
        testPred = RNNModel.predictVarLen(vtestX, minLen, maxLen, step)
        trainPred= trainPred.reshape(-1, 1)
    else:
        trainPred = RNNModel.predict(trainX)
        testPred = RNNModel.predict(testX)
        trainPred = trainPred.reshape(-1, 1)

    if varFlag:
        # 转化一下test的label
        testY = util.transform_groundTruth(vtestY, minLen, maxLen, step)
        testY = testY.reshape(-1, 1)
        testPred = testPred.reshape(-1, 1)
        print("testY", testY.shape)
        print("testPred", testPred.shape)

    # 还原数据
    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    # 评估指标
    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(testY, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    #util.plot(trainPred,trainY,testPred,testY)

    return trainPred, testPred, MAE, MRSE, SMAPE


if __name__ == "__main__":

    lag = 24
    batch_size = 32
    epoch = 20
    hidden_dim = 64
    lr = 1e-4
    unit = "GRU"

    # ts, data = util.load_data("./data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/bike_hour.csv", columnName="cnt")
    # ts, data = util.load_data("./data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/traffic_data_in_bits.csv", columnName="value")
    # ts, data = util.load_data("./data/beijing_pm25.csv", columnName="pm2.5")
    # ts, data = util.load_data("./data/pollution.csv", columnName="Ozone")
    ts, data = util.load_data("./data/ali_cloud/m_1955_cpu.csv", columnName="cpu")

    trainPred, testPred, mae, mrse, smape = RNN_forecasting(data, lookBack=lag, epoch=epoch, batchSize=batch_size,
                                            varFlag=False, minLen=24, maxLen=48, step=8, unit=unit, lr=lr)





