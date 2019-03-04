
import util
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# ts, data = util.load_data("./data/NSW2013.csv", columnName="TOTALDEMAND")
ts, data = util.load_data("./data/bike_hour.csv", columnName="cnt")
# ts, data = util.load_data("./data/TAS2016.csv", columnName="TOTALDEMAND")
# ts, data = util.load_data("./data/traffic_data_in_bits.csv", columnName="value")
# ts, data = util.load_data("./data/beijing_pm25.csv", columnName="pm2.5")
# ts, data = util.load_data("./data/pollution.csv", columnName="Ozone")

freq = 24*60//30*1
# freq = 8
decomposition = seasonal_decompose(ts.values, model="additive", freq=freq) # 季节分解

# 直接舍弃NA值
trend = pd.DataFrame(decomposition.trend).dropna().values
seasonal = pd.DataFrame(decomposition.seasonal).dropna().values
residual = pd.DataFrame(decomposition.resid).dropna().values
trend = trend.astype('float32')
seasonal = seasonal[freq//2:-(freq//2)].astype('float32')  # 为了三个数据尺度统一，舍弃seanson的前后2个数值
residual = residual.astype('float32')
data = data[freq//2:-(freq//2)].astype('float32')

print("seasonal ts:", len(seasonal))
print("residual ts:", len(residual))
print("trend ts:", len(trend))

seaRange = np.max(seasonal)-np.min(seasonal)
resRange = np.max(residual)-np.min(residual)
trendRange = np.max(trend)-np.min(trend)
totalRange = np.max(data)-np.min(data)

# 作图观察
fig = plt.figure()
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

ax1.plot(data[-1000:], 'k')
ax1.set_ylabel("Original")
ax2.plot(trend[-1000:], 'k')
ax2.set_ylabel("Trend")
ax3.plot(seasonal[-1000:], 'k')
ax3.set_ylabel("Peroid")
ax4.plot(residual[-1000:], 'k')
ax4.set_ylabel("Residual")

fig.tight_layout()

plt.show()