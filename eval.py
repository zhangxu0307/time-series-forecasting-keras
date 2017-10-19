#encoding=utf-8
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
from keras.metrics import  mean_absolute_percentage_error

# 计算RMSE
def calcRMSE(true,pred):
    return np.sqrt(mean_squared_error(true,pred))

# 计算MAE
def calcMAE(true,pred):
    #pred = pred[:, 0]
    return mean_absolute_error(true,pred)

# 计算MAPE
def calcMAPE(true, pred, epsion = 0.0000000):
    #pred = pred[:,0] # 列转行，便于广播计算误差指标
    # print (true-pred).shape
    # print true.shape
    # print pred.shape
    true += epsion
    return np.sum(np.abs((true-pred)/true))/len(true)*100
    #return mean_absolute_percentage_error(true, pred)

# 计算SMAPE
def calcSMAPE(true, pred):
    delim = (np.abs(true)+np.abs(pred))/2.0
    return np.sum(np.abs((true-pred)/delim))/len(true)*100