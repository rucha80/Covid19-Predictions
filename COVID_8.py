# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:15:10 2020

@author: rugve
"""
# import codecs
# import csv
# import urllib
from datetime import date
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
#from statsmodels.tsa.api import ExponentialSmoothing, Holt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


def data_for_country(country,data,column):
    data = data[["location","date",column]]
    data = data[data["location"] == country]
    data = data[data[column] != 0]
    data.reset_index(inplace = True) 
    data.Timestamp = pd.to_datetime(data.date,format='%Y-%m-%d') 
    data.index = data.Timestamp 
    data=data.drop('index',1)
    data=data.drop('location',1)
    data=data.drop('date',1)
    data = data.resample('D').mean()
    data.iloc[:,0]=  data.iloc[:,0].fillna(method='bfill').fillna(method='ffill')
    return data
    
def plot_Data(df):
    ts = df.iloc[:,0]
    ts.plot(figsize=(15,8), title= 'Daily Cases', fontsize=14)
    plt.show()
    
   
# def make_predictions_HL(df):
#     no_of_days = int(input("Please enter the number of days you want to predict for HL:"))
#     ts = df['total_cases']
#     fit1 = Holt(ts).fit(smoothing_level = 0.4,smoothing_slope = 0.8)
#     prediction= fit1.forecast(no_of_days).astype(int)
#     df2 = pd.DataFrame({'prediction': prediction }) 
#     return df2
    
# def make_predictions_HW_add(df):
#     no_of_days = int(input("Please enter the number of days you want to predict for HL:"))
#     ts = df['total_cases']
#     fit1 = ExponentialSmoothing(ts ,trend='add').fit()
#     prediction= fit1.forecast(no_of_days).astype(int)
#     df2 = pd.DataFrame({'prediction': prediction }) 
#     return df2
    
    
# def make_predictions_HW_mul(df):
#     no_of_days = int(input("Please enter the number of days you want to predict for HL:"))
#     ts = df['total_cases']
#     fit1 = ExponentialSmoothing(ts ,trend='mul').fit()
#     prediction= fit1.forecast(no_of_days).astype(int)
#     df2 = pd.DataFrame({'prediction': prediction }) 
#     return df2
    
def select_prams_for_arima(train,test):
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
                mod = sm.tsa.statespace.SARIMAX(train.iloc[:,0],order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                y_hat_avg['SARIMA'] = results.predict(start=test.index[0], 
                                                      end=test.index[-1], dynamic=True)
                rms_arimas.append(sqrt(mean_squared_error(test.iloc[:,0], y_hat_avg.SARIMA)))
            except:
                continue  
    data_tuples = list(zip(params,rms_arimas))
    rms = pd.DataFrame(data_tuples, columns=['Parameters','RMS value'])
    minimum = int(rms[['RMS value']].idxmin())
    parameters = params[minimum]
    return parameters
  
#df = data
def make_predictions_arima(df,parameters):    
    no_of_days = int(input("Please enter the number of days you want to predict :"))
    fit1 = sm.tsa.statespace.SARIMAX(df.iloc[:,0], order=parameters).fit()
    prediction = fit1.forecast(no_of_days).astype(int)
    df2 = pd.DataFrame({'prediction': prediction }) 
    plt.figure(figsize=(16,8))
    plt.plot(df.iloc[:,0], label='Original Data')
    plt.plot(df2['prediction'], label='Predicted data')
    plt.title("Predictions")
    plt.legend(loc='best')
    plt.show()
    print("ARIMAX model prediction")
    print(df2)
    
    #pd.DataFrame(df2, columns=['Date','prediction']).to_csv(r'C:\Users\rugve\Desktop\Rucha\3k\Covid19Data\Result1s.csv',index=True)
    
def train_test_split(data):
    today = date.today()
    today = str(today)
    today = today.replace(today[:8], '')
    today = int(today)
    split_index = len(data) - today
    train=data[0:split_index] 
    test=data[split_index:]
    parameters = select_prams_for_arima(train,test)
    make_predictions_arima(data,parameters)
    
    
# def get_data_from_url(url):
#     ftpstream = urllib.request.urlopen(url)
#     csvfile = csv.reader(codecs.iterdecode(ftpstream, 'utf-8'))
#     data = [ ]
#     for line in csvfile:
#         data.append(line)
#     column_names = data.pop(0)
#     full_data = pd.DataFrame(data,columns=column_names)
#     retrun full_data
    
def main():
    #url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
    url = r"C:\Users\rugve\Desktop\Rucha\3k\Covid19\owid-covid-data (2).csv"
    full_data = pd.read_csv(url)
    coutries = (list(set(full_data["location"])))
    print("Predictions for COVID19")
    print(coutries)
    cont = input("Enter a Country for which You want to make predictions from the above list:")
    columns  = ["total_cases","new_cases","total_deaths","new_deaths"]
    print(columns)
    colummn = input("Print the one value from above list for which  you want to  make prediction: ")
    data = data_for_country(cont,full_data,colummn)
    #plot_Data(data)
    #data = data.iloc[:103]
    train_test_split(data)
    
    
if __name__ == '__main__':
    main()


