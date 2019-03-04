#encoding=utf-8

import util
from models import decompose
import eval
import matplotlib.pyplot as plt
import naive_RNN_forecasting as RNNFORECAST
import pandas as pd

# 加载数据
path = "./data/bike/"
filename = path+"day.csv"
ts,dataset = util.load_data(filename,indexName="dteday", columnName="registered")
ts, dataset = util.load_data_xls("./data/NN5/NN5.xlsx", indexName="date", columnName="NN5-003")

# 序列分解
ts.index = pd.date_range(start='19960318',periods=len(ts), freq='Q')
trend,seasonal,residual = decompose.seasonDecompose(ts)
# print trend.shape
# print seasonal.shape
# print residual.shape

# 分别预测
trendWin = 25
resWin = trendWin
trTrain,trTest = RNNFORECAST.RNNforecasting(trend,lookBack=trendWin,epoch=50)

# 残差数据采用变长窗口RNN,采用Bagging方法
resTrain = resTest = None
regNum = 4
for i in range(regNum):
    tmpresTrain, tmpresTest = RNNFORECAST.RNNforecasting(residual,lookBack=resWin,epoch=50,varFlag=True,maxLen=20,inputNum=1500)
    if i == 0:
        resTrain = tmpresTrain
        resTest = tmpresTest
    else:
        resTrain += tmpresTrain
        resTest += tmpresTest
resTrain /= regNum
resTest /= regNum

#'''
# 数据对齐
trendPred,resPred = util.align(trTrain,trTest,trendWin,resTrain,resTest,resWin)

# 获取最终预测结果
finalPred = trendPred+resPred+seasonal

# 分别获得训练集测试集结果
trainPred = trTrain+resTrain+seasonal[trendWin:trendWin+trTrain.shape[0]]
testPred = trTest+resTest+seasonal[2*resWin+resTrain.shape[0]:]

# 获得ground-truth数据
data = dataset[2:-2]
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

plt.plot(data,'r')
plt.plot(finalPred,'g')
plt.show()

#'''



