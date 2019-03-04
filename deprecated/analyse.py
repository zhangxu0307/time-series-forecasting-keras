import util
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import  stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

# 加载数据
path = "./data/bike/"
filename = path+"hour.csv"
#ts,dataset = util.load_data(filename,indexName="dteday", columnName="registered")
#ts, dataset = util.load_data_xls("./data/NN5/NN5.xlsx", indexName="date", columnName="NN5-003")
ts, dataset = util.load_data("./data/AEMO/NSW/nsw.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")

#ts = ts.diff(1)#我们已经知道要使用一阶差分的时间序列，之前判断差分的程序可以注释掉
fig = plt.figure()
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ts,lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ts,lags=40,ax=ax2)
plt.show()