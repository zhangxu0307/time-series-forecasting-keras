import util
import eval
import numpy as np
import pyflux as pf


if __name__ == '__main__':

    p = 6
    q = 4
    h_test = 1
    lag = 24

    ts, data = util.load_data("./data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/bike_hour.csv", columnName="cnt")
    # ts, data = util.load_data("./data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/traffic_data_in_bits.csv", columnName="value")
    # ts, data = util.load_data("./data/beijing_pm25.csv", columnName="pm2.5")
    # ts, data = util.load_data("./data/pollution.csv", columnName="Ozone")

    train, test = util.divideTrainTest(data)
    testX, testY = util.createSamples(test, lookBack=24, RNN=False)
    testX = testX[:3]
    print("train shape is", train.shape)
    print("test shape is", test.shape)
    train_data = [x[0] for x in train]
    predictions = []
    realTestY = []

    for t in range(len(testX)):

        add_test = [x for x in testX[t]]
        raw_train_data = train_data
        train_data.extend(add_test)
        model = pf.ARIMA(data=np.array(train_data), ar=p, ma=q, family=pf.Normal())
        model.fit(method="MLE")

        output = model.predict(h_test, intervals=False)

        yhat = output.values[0][0]

        predictions.append(yhat)
        obs = test[t]
        realTestY.append(obs)
        train_data = raw_train_data
        print('t:%d, predicted=%f, expected=%f' % (t,  yhat, obs))

    realTestY = np.array(test).reshape(-1, 1)
    predictions = np.array(predictions).reshape(-1, 1)
    MAE = eval.calcMAE(realTestY, predictions)
    RMSE = eval.calcRMSE(realTestY, predictions)
    MAPE = eval.calcSMAPE(realTestY, predictions)
    print('Test MAE: %.8f' % MAE)
    print('Test RMSE: %.8f' % RMSE)
    print('Test MAPE: %.8f' % MAPE)

    # plot
    # pyplot.plot(test)
    # pyplot.plot(predictions, color='red')
    # pyplot.show()