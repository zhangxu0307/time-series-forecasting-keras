
from pandas import read_csv
import util
import eval
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

ts, data = util.load_data("./data/NSW2013.csv", columnName="TOTALDEMAND")
# ts, data = util.load_data("./data/bike_hour.csv", columnName="cnt")
# ts, data = util.load_data("./data/TAS2016.csv", columnName="TOTALDEMAND")
# ts, data = util.load_data("./data/traffic_data_in_bits.csv", columnName="value")
# ts, data = util.load_data("./data/beijing_pm25.csv", columnName="pm2.5")
# ts, data = util.load_data("./data/pollution.csv", columnName="Ozone")

size = int(len(data) * 0.75)
train, test = data[0:size], data[size:len(data)]
history = [x for x in train]

p = 6
q = 3
d = 1

predictions = []
for t in range(len(test)):
    model = ARIMA(history, order=(p, q, d))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))


test = np.array(test)
predictions = np.array(predictions).reshape(-1, 1)
MAE = eval.calcMAE(test, predictions)
RMSE = eval.calcRMSE(test, predictions)
MAPE = eval.calcMAPE(test, predictions)
print('Test MAE: %.8f' % MAE)
print('Test RMSE: %.8f' % RMSE)
print('Test MAPE: %.8f' % MAPE)

# plot
# pyplot.plot(test)
# pyplot.plot(predictions, color='red')
# pyplot.show()