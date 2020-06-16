# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:18:50 2020

@author: rugve
"""
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# Now, we will load the data set and look at some initial rows and data types of the columns:
data = pd.read_csv(r"C:\Users\rugve\Desktop\Rucha\3k\Covid19Data\owid-covid-data (2).csv")
country = "India"

#data preprocessing
def data_for_country(country,data):
    data = data[["location","date","total_cases"]]
    data = data[data["location"] == country]
    data = data[data["total_cases"] != 0]
    data.reset_index(inplace = True) 
    data.Timestamp = pd.to_datetime(data.date,format='%Y-%m-%d') 
    data.index = data.Timestamp 
    data=data.drop('index',1)
    data=data.drop('location',1)
    data=data.drop('date',1)
    data = data.resample('D').mean()
    data.total_cases=  data.total_cases.fillna(method='bfill').fillna(method='ffill')
    return data

data = data_for_country(country,data)
ts = data['total_cases']
plt.figure(figsize=(15,8))
plt.xlabel("Dates",fontsize = 10)
plt.ylabel('Total cases',fontsize = 10)
plt.plot(data.index,ts, label='No of cases',linestyle ='dotted',color = 'blue')
plt.legend(loc='best')
plt.title('Daily Cases ' + country)
plt.show() 

#select some last rows for spliting data into train and valid set
#selection based on data size
split = 13
split_index = len(data) - split
train=data[0:split_index] 
test=data[split_index:]

#Plotting data
# train.total_cases.plot(figsize=(15,8), title= 'Daily Cases ' + country, fontsize=14,)
# test.total_cases.plot(figsize=(15,8), title= 'Daily Cases ' +country, fontsize=14)
plt.figure(figsize=(15,8))
plt.plot(train.index,train['total_cases'], label='Train')
plt.plot(test.index,test['total_cases'], label='Test')
plt.legend(loc='best')
plt.title('Daily Cases ' + country)
plt.show() 

#Naive
dd= np.asarray(train.total_cases)
y_hat = test.copy()
y_hat['naive'] = dd[len(dd)-1]
plt.figure(figsize=(12,8))
plt.plot(train.index, train['total_cases'], label='Train')
plt.plot(test.index,test['total_cases'], label='Test')
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()
rms_naive = sqrt(mean_squared_error(test.total_cases, y_hat.naive))
print(rms_naive)

#Simple Average
y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['total_cases'].mean()
plt.figure(figsize=(12,8))
plt.plot(train['total_cases'], label='Train')
plt.plot(test['total_cases'], label='Test')
plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
plt.title("Average Forecast")
plt.legend(loc='best')
plt.show()
rms_sa = sqrt(mean_squared_error(test.total_cases, y_hat_avg.avg_forecast))
print(rms_sa)

#Moving Average
y_hat_avg = test.copy()
y_hat_avg['moving_avg_forecast'] = train['total_cases'].rolling(60).mean().iloc[-1]
plt.figure(figsize=(16,8))
plt.plot(train['total_cases'], label='Train')
plt.plot(test['total_cases'], label='Test')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
plt.title("Moving Average Forecast")
plt.legend(loc='best')
plt.show()
rms_ma = sqrt(mean_squared_error(test.total_cases, y_hat_avg.moving_avg_forecast))
print(rms_ma)

#Simple Exponential Smoothing
y_hat_avg = test.copy()
fit2 = SimpleExpSmoothing(np.asarray(train['total_cases'])).fit(smoothing_level=1.2,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['total_cases'], label='Train')
plt.plot(test['total_cases'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.title("Simple Exponential Smoothing Forecast")
plt.legend(loc='best')
plt.show()
rms_ses = sqrt(mean_squared_error(test.total_cases, y_hat_avg.SES))
print(rms_ses)

#Holt’s Linear Trend method
sm.tsa.seasonal_decompose(train.total_cases).plot()
result = sm.tsa.stattools.adfuller(train.total_cases)
plt.show()
y_hat_avg = test.copy()
fit1 = Holt(np.asarray(train['total_cases'])).fit(smoothing_level = 0.4,smoothing_slope = 0.8)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['total_cases'], label='Train')
plt.plot(test['total_cases'], label='Test')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.title("Holt’s Linear Trend method Forecast")
plt.legend(loc='best')
plt.show()
rms_hl = sqrt(mean_squared_error(test.total_cases, y_hat_avg.Holt_linear))
print(rms_hl)

#Holt’s Winter Trend method -  Triple exponential smoothing - additive
y_hat_avg5 = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['total_cases']) ,trend='add').fit()
y_hat_avg5['Holt_Winter_Additive'] = fit1.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['total_cases'], label='Train')
plt.plot(test['total_cases'], label='Test')
plt.plot(y_hat_avg5['Holt_Winter_Additive'], label='Holt_Winter Additive')
plt.title("Holt’s Winter Trend method additive Forecast")
plt.legend(loc='best')
plt.show()
rms_hw_add = sqrt(mean_squared_error(test.total_cases, y_hat_avg5.Holt_Winter_Additive))
print(rms_hw_add)

#Holt’s Winter Trend method -  Triple exponential smoothing - Multiplicative
y_hat_avg5 = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['total_cases']) ,trend='mul').fit()
y_hat_avg5['Holt_Winter_Multiplicative'] = fit1.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['total_cases'], label='Train')
plt.plot(test['total_cases'], label='Test')
plt.plot(y_hat_avg5['Holt_Winter_Multiplicative'], label='Holt_Winter Multiplicative')
plt.title("Holt’s Winter Trend method Multiplicative Forecast")
plt.legend(loc='best')
plt.show()
rms_hw_mul = sqrt(mean_squared_error(test.total_cases, y_hat_avg5.Holt_Winter_Multiplicative))
print(rms_hw_mul)

#parameter selection for arima
p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(0,0,0,0)]
params = []
rms_arimas =[] 
for param in pdq:
    params.append(param)  
    for param_seasonal in seasonal_pdq:
        try:
            y_hat_avg = test.copy()
            mod = sm.tsa.statespace.SARIMAX(train.total_cases,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            y_hat_avg['SARIMA'] = results.predict(start="2020-05-09", end="2020-05-21", dynamic=True)
            rms_arimas.append(sqrt(mean_squared_error(test.total_cases, y_hat_avg.SARIMA)))
        except:
            continue  

data_tuples = list(zip(params,rms_arimas))
rms = pd.DataFrame(data_tuples, columns=['Parameters','RMS value'])
minimum = int(rms[['RMS value']].idxmin())
parameters = params[minimum]

#SARIMA
y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train.total_cases, order=parameters,seasonal_order=(0,0,0,0),enforce_stationarity=False,
                                            enforce_invertibility=False).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="2020-06-01", end="2020-06-05", dynamic=True).astype(int)
plt.figure(figsize=(16,8))
plt.plot( train['total_cases'], label='Train')
plt.plot(test['total_cases'], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.title("ARIMA Forecast")
plt.legend(loc='best')
plt.show()
rms_arima = sqrt(mean_squared_error(test.total_cases, y_hat_avg.SARIMA))
print(rms_arima)


    
