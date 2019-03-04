# time-series-forecasting-keras

The experimental source code of Paper: *Time Series Forecasting using GRU Neural Network with Multi-lag after Decomposition, ICONIP 2017.*
[paper](https://link.springer.com/chapter/10.1007/978-3-319-70139-4_53 "paper"),
[HomePage](http://cs.nju.edu.cn/rinc/student/zhangxu/index.html " HomePage")


## Requirements ##

- python 3.6.3 (Anaconda)
- keras 2.1.2
- tensorflow-gpu 1.13.1
- sklearn 0.19.1
- numpy 1.15.4
- pandas 0.23.4
- statsmodels 0.8.0
- matplotlib 2.1.0

## Model ##

- LSTM
- GRU
- RNN
- MLP
- SVR
- ARIMA
- time series decomposition

## Code ##

- data
- models
	- decompose.py: **time series decomposition**
	- MLP.py: MLP network
	- RNNS.py: RNN faimily network, including RNN, LSTM, GRU
	- SVR.py: SVR model 
- naive\_MLP_forecasting.py
- naive\_RNN_forecasting.py
- naive\_SVR_forecasting.py
- decompose\_MLP_forecasting.py
- decompose\_RNN_forecasting.py
- decompose\_SVR_forecasting.py
- ARIMA.py (to do)
- util.py: load data, pre-processe time series, including **multi-lag** sampling
- eval.py: calculate the metrics
- subseries_plot.py: plot figure of time series decomposition

