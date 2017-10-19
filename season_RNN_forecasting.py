#encoding=utf-8

import pandas as pd
import util
import season_decompose
import eval
import numpy as np
import matplotlib.pyplot as plt
import naive_RNN_forecasting as RNNFORECAST
import time
import csv


def seasonRNNForecasting(ts, dataset, freq, lag, unit="GRU"):

    # 序列分解
    #ts.index = pd.date_range(start='19960318',periods=len(ts), freq='Q')
    trend,seasonal,residual = season_decompose.seasonDecompose(ts, freq)
    print (trend.shape)
    print (seasonal.shape)
    print (residual.shape)

    # 分别预测
    trendWin = lag
    resWin = trendWin
    t1 = time.time()
    trTrain, trTest, MAE1, MRSE1, SMAPE1 = RNNFORECAST.RNNforecasting(trend, lookBack=trendWin, epoch=50, unit=unit)
    resTrain,resTest, MAE2, MRSE2, SMAPE2 = RNNFORECAST.RNNforecasting(residual, lookBack=resWin, epoch=60, unit=unit, hiddenNum=100)
    # trTrain, trTest, MAE1, MRSE1, SMAPE1= RNNFORECAST.RNNforecasting(trend, lookBack=resWin, epoch=30, unit=unit,
    #                                                                     varFlag=True, minLen=20, maxLen=lag, step=4,
    #                                                                     hiddenNum=100)
    # resTrain, resTest, MAE2, MRSE2, SMAPE2 = RNNFORECAST.RNNforecasting(residual, lookBack=resWin, epoch=30, unit=unit,
    #                                                                     varFlag=True, minLen=20, maxLen=lag, step=4, hiddenNum=100)
    t2 = time.time()
    print (t2-t1)

    print("trTrain shape is", trTrain.shape)
    print("resTrain shape is", resTrain.shape)

    #'''
    # 数据对齐
    trendPred,resPred = util.align(trTrain,trTest,trendWin,resTrain,resTest,resWin)

    print("trendPred shape is", trendPred.shape)
    print("resPred shape is", resPred.shape)

    # 获取最终预测结果
    finalPred = trendPred+seasonal+resPred

    trainPred = trTrain+seasonal[trendWin:trendWin+trTrain.shape[0]]+resTrain
    testPred = trTest+seasonal[2*resWin+resTrain.shape[0]:]+resTest

    # 获得ground-truth数据
    data = dataset[freq//2:-(freq//2)]
    trainY = data[trendWin:trendWin+trTrain.shape[0]]
    testY = data[2*resWin+resTrain.shape[0]:]

    # 评估指标
    MAE = eval.calcMAE(trainY, trainPred)
    print ("train MAE",MAE)
    MRSE = eval.calcRMSE(trainY, trainPred)
    print ("train MRSE",MRSE)
    MAPE = eval.calcMAPE(trainY, trainPred)
    print ("train MAPE",MAPE)
    MAE = eval.calcMAE(testY,testPred)
    print ("test MAE",MAE)
    MRSE = eval.calcRMSE(testY,testPred)
    print ("test RMSE",MRSE)
    MAPE = eval.calcMAPE(testY,testPred)
    print ("test MAPE",MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    # plt.plot(data)
    # plt.plot(finalPred)
    # plt.show()
    #'''
    return trainPred, testPred, MAE, MRSE, SMAPE

if __name__ == "__main__":
    #ts, data = util.load_data("./data/bike/day.csv", indexName="dteday", columnName="registered")
    lag = 40
    #freq = 24*60//60*1
    freq = 4

    # csvfile = open('./result/result_season_RNN.csv', 'w', newline='')
    # writer = csv.writer(csvfile)
    # writer.writerow(["lag="+str(lag)])
    # writer.writerow(['tsName', 'MAE', 'MRSE', 'SMAPE'])

    for i in range(1, 2):
        # tsName = "NN5-"+str(i).zfill(3)
        # print(tsName)
        # ts, data = util.load_data_xls("./data/NN5/NN5.xlsx", indexName="date", columnName=tsName)
        #ts, data = util.load_data("./data/AEMO/NSW/nsw.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
        ts, data = util.load_data("./data/AEMO/NSW/TAS2016.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
        #ts, data = util.load_data("./data/bike/hour.csv", indexName="dteday", columnName="registered")
        # ts, data = util.load_data("./data/AEMO/TT30GEN.csv", indexName="TRADING_INTERVAL", columnName="VALUE")
        #     ts, data = util.load_data_xls("./data/NN5/NN5.xlsx", indexName="date", columnName=tsName)
        #ts, data = util.load_data_xls("./data/NN5/NN5.xlsx", indexName="date", columnName=tsName)
        trainPred,testPred, mae, mrse, smape = seasonRNNForecasting(ts, data, lag=lag, freq=freq,unit="LSTM")
    #     writer.writerow([tsName, str(mae), str(mrse), str(smape)])
    #
    # csvfile.close()



