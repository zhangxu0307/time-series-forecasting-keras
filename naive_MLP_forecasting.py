from models import MLP
import util
import eval
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
# np.random.seed(123)


def MLP_forecasting(dataset, inputDim, lr=1e-3, hiddenNum=50, outputDim=1, epoch=20, batchSize=30, plot_flag=False):

    # normalize time series
    # dataset = dataset.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0)).fit(dataset)
    dataset = scaler.transform(dataset)

    # divide the series into training/testing samples
    # NOTE: Not RNN format
    train, test = util.divideTrainTest(dataset)

    trainX, trainY = util.createSamples(train, inputDim, RNN=False)
    testX, testY = util.createSamples(test, inputDim, RNN=False)
    print("trainX shape is", trainX.shape)
    print("trainY shape is", trainY.shape)
    print("testX shape is", testX.shape)
    print("testY shape is", testY.shape)

    # buil model and train
    MLP_model = MLP.MLP_Model(inputDim, hiddenNum, outputDim, lr)
    t1 = time.time()
    MLP_model.train(trainX, trainY, epoch, batchSize)
    t2 = time.time()-t1
    print("train time is", t2)

    # forecasting
    trainPred = MLP_model.predict(trainX)
    testPred = MLP_model.predict(testX)

    # reverse the time series
    trainPred = scaler.inverse_transform(trainPred)
    trainY = scaler.inverse_transform(trainY)
    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    # evaluate
    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    if plot_flag:
        util.plot(trainPred, trainY, testPred, testY)

    return trainPred, testPred, MAE, MRSE, SMAPE


if __name__ == "__main__":

    lag = 40
    batch_size = 32
    epoch = 20
    hidden_dim = 64
    lr = 1e-4

    # ts, data = util.load_data("./data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/bike_hour.csv", columnName="cnt")
    # ts, data = util.load_data("./data/TAS2016.csv", columnName="TOTALDEMAND")
    ts, data = util.load_data("./data/traffic_data_in_bits.csv", columnName="value")
    # ts, data = util.load_data("./data/beijing_pm25.csv", columnName="pm2.5")
    # ts, data = util.load_data("./data/pollution.csv", columnName="Ozone")
    trainPred, testPred, mae, mrse, smape = MLP_forecasting(data, inputDim=lag, hiddenNum=hidden_dim,
                                            lr=lr, epoch=epoch, batchSize=batch_size, plot_flag=True)

