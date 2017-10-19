import ANN
import util
import eval
from sklearn.preprocessing import MinMaxScaler
import csv
import time
import numpy as np
#np.random.seed(123)


def ANNforecasting(dataset, inputDim, hiddenNum = 50 ,outputDim = 1, epoch = 20, batchSize = 30):

    # 归一化数
    #dataset = dataset.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    dataset = scaler.fit_transform(dataset)

    # 分割序列为样本,此处不整理成RNN形式，采用标准形式
    train,test = util.divideTrainTest(dataset)

    trainX, trainY = util.createSamples(train, inputDim, RNN=False)
    testX, testY = util.createSamples(test, inputDim, RNN=False)
    print("trainX shape is", trainX.shape)
    print("trainY shape is", trainY.shape)

    # 构建模型并训练
    ANNModel = ANN.ANNModel(inputDim, hiddenNum, outputDim)
    t1 = time.time()
    ANNModel.train(trainX, trainY, epoch, batchSize)
    t2 = time.time()-t1
    print("train time is", t2)

    # 预测
    trainPred = ANNModel.predict(trainX)
    testPred = ANNModel.predict(testX)


    # 还原数据
    trainPred = scaler.inverse_transform(trainPred)
    trainY = scaler.inverse_transform(trainY)
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
    #MAPE = eval.calcMAPE(testY,testPred)
    #print ("test MAPE",MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print ("test SMAPE",SMAPE)

    util.plot(trainPred,trainY,testPred,testY)

    return trainPred, testPred, MAE, MRSE, SMAPE

if __name__ == "__main__":

    lag = 24
    # csvfile = open('./result/result_ANN.csv', 'w', newline='')
    # writer = csv.writer(csvfile)
    # writer.writerow(["lag="+str(lag)])
    # writer.writerow(['tsName', 'MAE', 'MRSE', 'SMAPE'])

    for i in range(3,4):
        #tsName = "NN5-"+str(i).zfill(3)
        #print(tsName)
        #ts, data = util.load_data_xls("./data/NN5/NN5.xlsx", indexName="date", columnName=tsName)
        #ts, data = util.load_data("./data/AEMO/NSW/nsw.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
        #ts, data = util.load_data("./data/AEMO/NSW/TAS2016.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
        ts, data = util.load_data("./data/bike/hour.csv", indexName="dteday", columnName="cnt")
        #ts, data = util.load_data("./data/AEMO/TT30GEN.csv", indexName="TRADING_INTERVAL", columnName="VALUE")
        trainPred,testPred, mae, mrse, smape = ANNforecasting(data, inputDim=lag, epoch=20)
        #writer.writerow([tsName, str(mae), str(mrse), str(smape)])

    #csvfile.close()

