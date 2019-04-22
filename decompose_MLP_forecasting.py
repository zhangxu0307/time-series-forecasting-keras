import util
from models import decompose
import eval
import naive_MLP_forecasting as ANNFORECAST
import time


def decompose_MLP_forecasting(ts, dataset, freq, lag, epoch=20, hidden_num=64, batch_size=32, lr=1e-3):

    # 序列分解
    trend, seasonal, residual = decompose.ts_decompose(ts, freq = freq)
    print("trend shape:", trend.shape)
    print("peroid shape:", seasonal.shape)
    print("residual shape:", residual.shape)

    # 分别预测
    resWin = trendWin = lag
    t1 = time.time()
    trTrain, trTest, mae1, mrse1, smape1 = \
        ANNFORECAST.MLP_forecasting(trend, inputDim=trendWin, epoch=epoch, hiddenNum=hidden_num,
                                    batchSize=batch_size, lr=lr)
    resTrain, resTest, mae2, mrse2, smape2 = \
        ANNFORECAST.MLP_forecasting(residual, inputDim=trendWin, epoch=epoch, hiddenNum=hidden_num,
                                    batchSize=batch_size, lr=lr)
    t2 = time.time()
    print("time:", t2-t1)

    # 数据对齐
    # trendPred, resPred = util.align(trTrain, trTest, trendWin, resTrain, resTest, resWin)

    # 获取最终预测结果
    # finalPred = trendPred + seasonal + resPred

    trainPred = trTrain + seasonal[trendWin:trendWin+trTrain.shape[0]] + resTrain
    testPred = trTest + seasonal[2 * resWin + resTrain.shape[0]:] + resTest

    # 获得ground-truth数据
    data = dataset[freq // 2 : -(freq//2)]
    trainY = data[trendWin:trendWin+trTrain.shape[0]]
    testY = data[2 * resWin+resTrain.shape[0]:]

    # 评估指标
    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(testY, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    # plt.plot(data)
    # plt.plot(finalPred)
    # plt.show()

    return trainPred, testPred, MAE, MRSE, SMAPE


if __name__ == "__main__":

    lag = 24
    batch_size = 32
    epoch = 20
    hidden_dim = 64
    lr = 1e-4
    freq = 4

    # ts, data = util.load_data("./data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/bike_hour.csv", columnName="cnt")
    # ts, data = util.load_data("./data/TAS2016.csv", columnName="TOTALDEMAND")
    ts, data = util.load_data("./data/traffic_data_in_bits.csv", columnName="value")
    # ts, data = util.load_data("./data/beijing_pm25.csv", columnName="pm2.5")
    # ts, data = util.load_data("./data/pollution.csv", columnName="Ozone")

    trainPred, testPred, mae, mrse, smape = decompose_MLP_forecasting(ts, data, lag=lag, freq=freq,
                                                                      epoch=epoch, hidden_num=hidden_dim,
                                                                      lr=lr, batch_size=batch_size)
