#encoding=utf-8
import util
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm


ts,data = util.load_data("./data/bike/hour.csv", indexName="dteday", columnName="registered")
#ts, data = util.load_data_xls("./data/NN5/NN5.xlsx", indexName="date", columnName="NN5-003")
#ts, data = util.load_data("./data/AEMO/NSW/nsw.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
#ts, data = util.load_data("./data/AEMO/NSW/TAS2016.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
#ts.index = pd.date_range(start='19960318',periods=len(ts))
print(ts)

#freq = 24*60//60*1
freq = 8
decomposition = seasonal_decompose(ts.values, model="additive", freq=freq) # 季节分解

# 填补NA值
# trend = pd.DataFrame(decomposition.trend.fillna(method="pad").fillna(method="bfill")).values
# seasonal = pd.DataFrame(decomposition.seasonal.dropna()).values
# residual = pd.DataFrame(decomposition.resid.fillna(method="pad").fillna(method="bfill")).values

# 直接舍弃NA值
trend = pd.DataFrame(decomposition.trend).dropna().values
seasonal = pd.DataFrame(decomposition.seasonal).dropna().values
residual = pd.DataFrame(decomposition.resid).dropna().values
trend = trend.astype('float32')
seasonal = seasonal[freq//2:-(freq//2)].astype('float32') # 为了三个数据尺度统一，舍弃seanson的前后2个数值，只有season与源数据维度一致
residual = residual.astype('float32')

#new = seasonal+trend+residual


print (len(seasonal))
print (len(residual))
print (len(trend))


seaRange = np.max(seasonal)-np.min(seasonal)
resRange = np.max(residual)-np.min(residual)
trendRange = np.max(trend)-np.min(trend)
totalRange = np.max(data)-np.min(data)

# print np.sum(np.abs(seasonal)/np.abs(new))
# print np.sum(np.abs(residual)/np.abs(new))


#util.LBtest(residual)

#作图观察
ax1 = plt.subplot(411)
ax2 = plt.subplot(412)
ax3 = plt.subplot(413)
ax4 = plt.subplot(414)

ax1.plot(data[-1006:-6],'k')
#ax1.set_title("Original")
ax2.plot(trend[-1000:],'k')
#ax2.set_title("Trend")
ax3.plot(seasonal[-1000:],'k')
#ax3.set_title("Season")
ax4.plot(residual[-1000:],'k')
#ax4.set_title("Residual")
#rec = seasonal+trend+residual
# print rec.shape
# #plt.plot(data[6:-6],'g')
# print rec-data[6:-6]
# plt.plot(rec,'r')

fig = plt.figure()
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(residual,lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(residual,lags=40,ax=ax2)
plt.show()

plt.show()