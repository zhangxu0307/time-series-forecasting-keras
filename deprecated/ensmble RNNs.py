from models import RNNs
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler
import eval
import matplotlib.pyplot as plt

def diffLagEnsmble(dataset, minLag, maxLag, step, epoch, batchSize = 10, inputDim = 1, outputDim = 1, hiddenNum = 50, unit="GRU"):


    # 归一化数据
    dataset = dataset.reshape(-1, 1)
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    # 分割序列为样本,并整理成RNN的输入形式
    train, test = util.divideTrainTest(dataset)
    modelList = []

    #在所有lag下独立地训练若干个RNN
    for lag in range(minLag, maxLag+1, step):

        print ("lag is ", lag)

        trainX, trainY = util.createPaddedDataset(train, lag, maxLag)
        #testX, testY = util.createPaddedDataset(test, lag, maxLag)

        RNNModel = RNNs.RNNsModel(inputDim, hiddenNum, outputDim, unit)
        RNNModel.train(trainX, trainY, epoch, batchSize)
        modelList.append(RNNModel)

    print("the number of model is ", len(modelList))

    # 变长分割test数据
    testX, testY = util.createVariableDataset(test, minLag, maxLag, step)
    print ("testX", testX.shape)
    print ("testY", testY.shape)

    lagContainNum = (maxLag - minLag)//step+1
    print ("lag contains num is ", lagContainNum)

    # 对每个数据点上不同的lag用不同的模型做预测，目前是简单取平均
    testNum = len(testX)
    testPred = []
    for i in range(0, testNum, lagContainNum):
        predList = []
        for j in range(0, lagContainNum):
            singlePred = modelList[j].predict(testX[i+j:i+j+1, :, :])
            predList.append(singlePred)
        testPred.append(np.mean(predList))

    # 还原数据
    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    testY = util.transformGroundTruth(testY, minLag, maxLag, step)
    testPred = np.array(testPred)
    print("testY", testY.shape)
    print ("testPred", testPred.shape)

    # 评估
    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(testY, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    # plt.plot(testY)
    # plt.plot(testPred)
    # plt.show()

    return testPred, MAE, MRSE, SMAPE

if __name__ == "__main__":
    for i in range(3, 4):
        # tsName = "NN5-" + str(i).zfill(3)
        # print(tsName)
        # ts, data = util.load_data_xls("./data/NN5/NN5.xlsx", indexName="date", columnName=tsName)
        #ts, data = util.load_data("./data/AEMO/NSW/nsw.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
        #ts, data = util.load_data("./data/bike/hour.csv", indexName="dteday", columnName="registered")
        ts, data = util.load_data("./data/AEMO/TT30GEN.csv", indexName="TRADING_INTERVAL", columnName="VALUE")
        testPred, mae, mrse, smape = diffLagEnsmble(data,minLag=20, maxLag=40, step=5, epoch=10,
                                                               unit="GRU")





