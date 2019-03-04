from deprecated import stateful_RNNs as staRNNs
import util
import eval
from sklearn.preprocessing import MinMaxScaler


def statefulRNNforecasting(dataset,lookBack,inputDim = 1,hiddenNum = 100 ,outputDim = 1 ,unit = "GRU",epoch = 50,batchSize = 10,varFlag=False, minLen = 15, maxLen = 30,inputNum = 150):

    # 归一化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 分割序列为样本,并整理成RNN的输入形式
    train,test = util.divideTrainTest(dataset)

    trainX,trainY = util.createSamples(train,lookBack)
    testX, testY = util.createSamples(test,lookBack)

    # 构建模型并训练
    RNNModel = staRNNs.statefulRNNsModel(inputDim, hiddenNum, outputDim, unit=unit, batchSize=batchSize, lag=lookBack)
    if varFlag:
        vtrainX, vtrainY = util.createVariableDataset(train, minLen, maxLen, inputNum)
        RNNModel.train(vtrainX, vtrainY, epoch, batchSize)
    else:
        RNNModel.train(trainX, trainY, epoch, batchSize)

    # 预测
    trainPred = RNNModel.predict(trainX, batchSize)
    testPred = RNNModel.predict(testX, batchSize)

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
    MAPE = eval.calcMAPE(testY,testPred)
    print ("test MAPE",MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test MAPE", SMAPE)

    # util.LBtest(testY-testPred)
    # plt.plot(testY-testPred)
    # plt.show()

    #util.plot(trainPred,trainY,testPred,testY)

    return trainPred, testPred, MAE, MRSE, SMAPE

if __name__ == "__main__":
    #ts, data = util.load_data("./data/bike/day.csv", indexName="dteday", columnName="registered")

    lag = 20
    # csvfile = open('./result/result_RNN.csv', 'w', newline='')
    # writer = csv.writer(csvfile)
    # writer.writerow(["lag=" + str(lag)])
    # writer.writerow(['tsName', 'MAE', 'MRSE', 'SMAPE'])

    for i in range(1, 2):
        # tsName = "NN5-" + str(i).zfill(3)
        # print(tsName)
        #ts, data = util.load_data_xls("./data/NN5/NN5.xlsx", indexName="date", columnName=tsName)
        ts, data = util.load_data("./data/AEMO/NSW/nsw.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
        trainPred, testPred, mae, mrse, smape = statefulRNNforecasting(data,lookBack=lag,epoch=10,varFlag=False,minLen=15,maxLen=25, inputNum=20000,unit="GRU")
        #writer.writerow([tsName, str(mae), str(mrse), str(smape)])

    #csvfile.close()
