#encoding=utf-8

import util
from models import decompose
import eval
from naive_RNN_forecasting import RNN_forecasting
import time
import matplotlib.pyplot as plt


def decompose_RNN_forecasting(ts, dataset, freq, lag, epoch=20, hidden_num=64,
                              batch_size=32, lr=1e-3, unit="GRU", varFlag=False, maxLen=48, minLen=24, step=8):

    # 序列分解
    trend, seasonal, residual = decompose.ts_decompose(ts, freq)
    print("trend shape:", trend.shape)
    print("peroid shape:", seasonal.shape)
    print("residual shape:", residual.shape)

    # 分别预测
    resWin = trendWin = lag
    t1 = time.time()
    trTrain, trTest, MAE1, MRSE1, SMAPE1 = RNN_forecasting(trend, lookBack=lag, epoch=epoch, batchSize=batch_size, hiddenNum=hidden_num,
                                            varFlag=varFlag, minLen=minLen, maxLen=maxLen, step=step, unit=unit, lr=lr)
    resTrain, resTest, MAE2, MRSE2, SMAPE2 = RNN_forecasting(residual, lookBack=lag, epoch=epoch, batchSize=batch_size, hiddenNum=hidden_num,
                                            varFlag=varFlag, minLen=minLen, maxLen=maxLen, step=step, unit=unit, lr=lr)
    t2 = time.time()
    print(t2-t1)

    print("trTrain shape:", trTrain.shape)
    print("resTrain shape:", resTrain.shape)

    # 数据对齐
    trendPred, resPred = util.align(trTrain, trTest, trendWin, resTrain, resTest, resWin)
    print("trendPred shape is", trendPred.shape)
    print("resPred shape is", resPred.shape)

    # 获取最终预测结果
    # finalPred = trendPred+seasonal+resPred

    trainPred = trTrain+seasonal[trendWin:trendWin+trTrain.shape[0]]+resTrain
    testPred = trTest+seasonal[2*resWin+resTrain.shape[0]:]+resTest

    # 获得ground-truth数据
    data = dataset[freq//2:-(freq//2)]
    trainY = data[trendWin:trendWin+trTrain.shape[0]]
    testY = data[2*resWin+resTrain.shape[0]:]

    # 评估指标
    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(testY, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    plt.plot(testY, label='ground-truth')
    plt.plot(testPred, label='prediction')
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("CPU Utilization(%)", fontsize=10)
    plt.legend()
    foo_fig = plt.gcf()
    foo_fig.savefig('M_1955_CPU.eps', format='eps', dpi=1000, bbox_inches='tight')
    plt.show()

    return trainPred, testPred, MAE, MRSE, SMAPE


if __name__ == "__main__":

    lag = 24  # if using varFlag, lag == maxLen
    batch_size = 32
    epoch = 12
    hidden_dim = 64
    unit = "GRU"
    lr = 1e-4
    freq = 8
    varFlag = True
    minLen = 12
    maxLen = 24
    step = 6

    # ts, data = util.load_data("./data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/bike_hour.csv", columnName="cnt")
    # ts, data = util.load_data("./data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/traffic_data_in_bits.csv", columnName="value")
    # ts, data = util.load_data("./data/beijing_pm25.csv", columnName="pm2.5")
    # ts, data = util.load_data("./data/pollution.csv", columnName="Ozone")
    ts, data = util.load_data("./data/ali_cloud/m_1955_cpu.csv", columnName="cpu")

    trainPred, testPred, mae, mrse, smape = decompose_RNN_forecasting(ts, data, lag=lag, freq=freq, unit=unit,
                                                                     varFlag=varFlag, minLen=minLen, maxLen=maxLen,
                                                                     step=step, epoch=epoch, hidden_num=hidden_dim,
                                                                     lr=lr, batch_size=batch_size)




