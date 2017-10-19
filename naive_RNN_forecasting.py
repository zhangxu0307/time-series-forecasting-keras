#encoding=utf-8

import RNNs
import util
import eval
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv
import numpy as np


def RNNforecasting(dataset,lookBack,inputDim = 1,hiddenNum = 100 ,outputDim = 1 ,unit = "GRU",epoch = 50,batchSize = 30,varFlag=False, minLen = 15, maxLen = 30,step = 5):

    # 归一化数据
    #dataset = dataset.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 分割序列为样本,并整理成RNN的输入形式
    train,test = util.divideTrainTest(dataset)

    trainX = None
    trainY = None
    vtrainX = None
    vtrainY = None
    testX = None
    testY = None
    vtestX = None
    vtestY = None

    # 构建模型并训练
    RNNModel = RNNs.RNNsModel(inputDim, hiddenNum, outputDim, unit)
    if varFlag:
        vtrainX, vtrainY = util.createVariableDataset(train, minLen, maxLen, step)
        vtestX, vtestY = util.createVariableDataset(test, minLen, maxLen, step)
        #vtrainY = vtrainY.reshape(-1, 1)
        RNNModel.train(vtrainX, vtrainY, epoch, batchSize)
    else:
        trainX, trainY = util.createSamples(train, lookBack)
        testX, testY = util.createSamples(test, lookBack)
        # trainY = trainY.reshape(-1,1)
        # testY = testY.reshape(-1,1)
        RNNModel.train(trainX, trainY, epoch, batchSize)

    # 预测
    if varFlag:
        trainPred = RNNModel.predictVarLen(vtrainX, minLen, maxLen, step)
        testPred = RNNModel.predictVarLen(vtestX, minLen, maxLen, step)
    else:
        trainPred = RNNModel.predict(trainX)
        testPred = RNNModel.predict(testX)

    if varFlag:
        testY = util.transformGroundTruth(vtestY, minLen, maxLen, step) # 转化一下test的label
        #testY = testY.reshape(-1, 1)
        print ("testY",testY.shape)
        print ("testPred",testPred.shape)

    # 还原数据
    #trainPred = scaler.inverse_transform(trainPred)
    #trainY = scaler.inverse_transform(trainY)
    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)
    dataset = scaler.inverse_transform(dataset)

    # 评估指标
    # MAE = eval.calcMAE(trainY, trainPred)
    # print ("train MAE",MAE)
    # MRSE = eval.calcRMSE(trainY, trainPred)
    # print ("train MRSE",MRSE)
    # MAPE = eval.calcMAPE(trainY, trainPred)
    # print ("train MAPE",MAPE)


    MAE = eval.calcMAE(testY,testPred)
    print ("test MAE",MAE)
    MRSE = eval.calcRMSE(testY,testPred)
    print ("test RMSE",MRSE)
    MAPE = eval.calcMAPE(testY,testPred)
    print ("test MAPE",MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    #util.plot(trainPred,trainY,testPred,testY)

    return trainPred, testPred, MAE, MRSE, SMAPE

if __name__ == "__main__":

    lag = 24
    # csvfile = open('./result/result_GRU.csv', 'w', newline='')
    # writer = csv.writer(csvfile)
    # writer.writerow(["lag=" + str(lag)])
    # writer.writerow(['tsName', 'MAE', 'MRSE', 'SMAPE'])

    for i in range(1, 2):
        # tsName = "NN5-" + str(i).zfill(3)
        # print(tsName)
        #ts, data = util.load_data_xls("./data/NN5/NN5.xlsx", indexName="date", columnName=tsName)

        #ts, data = util.load_data("./data/AEMO/NSW/nsw.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
        ts, data = util.load_data("./data/bike/hour.csv", indexName="dteday", columnName="registered")
        #ts, data = util.load_data("./data/AEMO/NSW/TAS2016.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
        #ts, data = util.load_data("./data/AEMO/TT30GEN.csv", indexName="TRADING_INTERVAL", columnName="VALUE")
        trainPred, testPred, mae, mrse, smape = RNNforecasting(data,lookBack=lag, epoch=15,
                                                               varFlag=True, minLen=25, maxLen=50, step=3,unit="GRU"
                                                                                                            )
    #     writer.writerow([tsName, str(mae), str(mrse), str(smape)])
    # csvfile.close()



