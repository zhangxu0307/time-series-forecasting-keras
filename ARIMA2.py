import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as st
import numpy as np
import pyflux as pf
import util
import eval


def stationarity(timeseries):
    timeseries = timeseries.flatten()
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1]


def best_diff(df, maxdiff = 8):
    p_set = {}
    for i in range(0, maxdiff):
        temp = df.copy() #每次循环前，重置
        if i == 0:
            temp['diff'] = temp[temp.columns[1]]
        else:
            temp['diff'] = temp[temp.columns[1]].diff(i)
            temp = temp.drop(temp.iloc[:i].index) #差分后，前几行的数据会变成nan，所以删掉
        pvalue = stationarity(temp['diff'])
        p_set[i] = pvalue
        p_df = pd.DataFrame.from_dict(p_set, orient="index")
        p_df.columns = ['p_value']
    i = 0
    while i < len(p_df):
        if p_df['p_value'][i]<0.01:
            bestdiff = i
            break
        i += 1
    return bestdiff


def produce_diffed_timeseries(df, diffn):
    if diffn != 0:
        df['diff'] = df[df.columns[1]].apply(lambda x:float(x)).diff(diffn)
    else:
        df['diff'] = df[df.columns[1]].apply(lambda x:float(x))
    df.dropna(inplace=True) #差分之后的nan去掉
    return df


def choose_order(ts, maxar, maxma):
    ts = ts.flatten()
    order = st.arma_order_select_ic(ts, maxar, maxma, ic=['aic', 'bic', 'hqic'])
    return order.bic_min_order


def predict_recover(ts, df, diffn):
    if diffn != 0:
        ts.iloc[0] = ts.iloc[0]+df['log'][-diffn]
        ts = ts.cumsum()
    # ts = np.exp(ts)
    print('还原完成')
    return ts


def run_aram(data, maxar, maxma):

    train, test = util.divideTrainTest(data)
    print("train shape is", train.shape)
    print("test shape is", test.shape)

    diffn = 0
    if stationarity(train) < 0.01:
        print('平稳，不需要差分')
    else:
        diffn = best_diff(train, maxdiff=8)
        # train = produce_diffed_timeseries(train, diffn)
        print('差分阶数为'+str(diffn))

    print('开始进行ARMA拟合')
    order = choose_order(train, maxar, maxma)
    print('模型的阶数为：'+ str(order))
    diffn = 1
    _ar = order[0]
    _ma = order[1]
    train = train.flatten()
    model = pf.ARIMA(data=train, ar=_ar, ma=_ma, integ=diffn, family=pf.Normal())
    model.fit("MLE")
    testPred = model.predict(len(test))
    # testPred = predict_recover(test_predict, train, diffn)

    # 评估指标
    MAE = eval.calcMAE(test, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(test, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(test, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(test, testPred)
    print("test SMAPE", SMAPE)


if __name__ == '__main__':

    ts, data = util.load_data("./data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/bike_hour.csv", columnName="cnt")
    # ts, data = util.load_data("./data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/traffic_data_in_bits.csv", columnName="value")
    # ts, data = util.load_data("./data/beijing_pm25.csv", columnName="pm2.5")
    # ts, data = util.load_data("./data/pollution.csv", columnName="Ozone")
    run_aram(data, maxar=12, maxma=12)