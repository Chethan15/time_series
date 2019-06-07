import pandas as pd
from pandas import DataFrame
import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from preprocess import train_test_split
from preprocess import n_series
from datetime import datetime, timedelta 
from sklearn.metrics import mean_absolute_error
from collections import OrderedDict
import os,sys,time

df = pd.read_csv("4200_C005_2019_03_03.tsv", sep=',', header=None)
df.columns = ["kunag", "matnr", "date", "quantity","price"]


def naive(input_df,kunag,matnr,n):
#     path = '/home/rahul/Downloads/bharat/time_series1/arima010'
#     index = str(kunag) +"_"+ str(matnr)

    i = 0
    lst = []
    test1 = train_test_split(df,kunag,matnr,n)[1]
    y_hat = test1.copy()

    for i in range(n,0,-1):
        train,test = train_test_split(df,kunag,matnr,i)
        dd= np.asarray(train["quantity"])
        y_hat['naive'] = int(dd[len(dd)-1])
        pred = y_hat['naive']
        lst.append(pred.iloc[-1])

    
    pd.DataFrame(lst)
    y_hat['pred_column']=lst
    plt.figure(figsize=(12,8))
    plt.plot( train.set_index("date")['quantity'], label='Train',marker = '.')
    plt.plot(test1.set_index("date")['quantity'], label='Test',marker = '.')
    plt.plot(y_hat.set_index("date")['pred_column'], label='Naive Forecast',marker = '.')
    plt.legend(loc='best')
    plt.title("Naive Forecast")
    plt.show()
#     plt.savefig(path + 'Graph_{}.png'.format(index), format="PNG")  

    rms = sqrt(mean_squared_error(test1.quantity, y_hat.pred_column))
    mae = mean_absolute_error(test1.quantity, y_hat.pred_column)
    del y_hat['naive']
    return y_hat,rms,mae


def average_forecast(input_df,kunag,matnr,n):
    i = 0
    lst = []
    test1 = train_test_split(df,kunag,matnr,n)[1]
    y_hat_avg = test1.copy()
    for i in range(n,0,-1):
        train,test = train_test_split(df,kunag,matnr,i)
        dd= np.asarray(train["quantity"])
        y_hat_avg['avg_forecast'] = train['quantity'].mean()
        pred = y_hat_avg['avg_forecast']
        lst.append(pred.iloc[-1])

    pd.DataFrame(lst)
    y_hat_avg['pred_column']=lst
    plt.figure(figsize=(12,8))
    plt.plot( train.set_index("date")['quantity'], label='Train',marker = '.')
    plt.plot(test1.set_index("date")['quantity'], label='Test',marker = '.')
    plt.plot(y_hat_avg.set_index("date")['pred_column'], label='Average Forecast',marker = '.')
    plt.legend(loc='best')
    plt.title("Average forecast")
    plt.show()
    rms = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_column))
    mae = mean_absolute_error(test1.quantity, y_hat_avg.pred_column)
    del y_hat_avg['avg_forecast']
    return y_hat_avg,rms,mae

def moving_average(input_df, kunag,matnr,n,roll):
    i = 0
    lst = []
    test1 = train_test_split(df,kunag,matnr,n)[1]
    y_hat_avg = test1.copy()
    for i in range(n,0,-1):
        train,test = train_test_split(df,kunag,matnr,i)
        dd= np.asarray(train["quantity"])
        y_hat_avg['moving_avg_forecast'] = train['quantity'].rolling(roll).mean().iloc[-1]
        pred = y_hat_avg['moving_avg_forecast']
        lst.append(pred.iloc[-1])

    pd.DataFrame(lst)
    y_hat_avg['pred_column']=lst
    plt.figure(figsize=(12,8))
    plt.plot( train.set_index("date")['quantity'], label='Train',marker = '.')
    plt.plot(test1.set_index("date")['quantity'], label='Test',marker = '.')
    plt.plot(y_hat_avg.set_index("date")['pred_column'], label='moving_avg_forecast',marker = '.')
    plt.legend(loc='best')
    plt.title("moving_avg_forecast")
    plt.show()
    rms = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_column))
    mae = mean_absolute_error(test1.quantity, y_hat_avg.pred_column)
    del y_hat_avg['moving_avg_forecast']
    return y_hat_avg,rms,mae


def ses(input_df, kunag,matnr,n,alpha):
    i = 0
    lst = []
    test1 = train_test_split(df,kunag,matnr,n)[1]
    y_hat_avg = test1.copy()
    for i in range(n,0,-1):
        train,test = train_test_split(df,kunag,matnr,i)
        dd= np.asarray(train["quantity"])
        fit2 = SimpleExpSmoothing(np.asarray(train['quantity'])).fit(smoothing_level=alpha,optimized=False)
        y_hat_avg['SES'] = fit2.forecast(len(test1))
        pred = y_hat_avg['SES']
        lst.append(pred.iloc[-1])

    pd.DataFrame(lst)
    y_hat_avg['pred_column']=lst
    plt.figure(figsize=(12,8))
    plt.plot( train.set_index("date")['quantity'], label='Train',marker = '.')
    plt.plot(test1.set_index("date")['quantity'], label='Test',marker = '.')
    plt.plot(y_hat_avg.set_index("date")['pred_column'], label='SES',marker = '.')
    plt.legend(loc='best')
    plt.title("SES")
    plt.show()
    rms = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_column))
    mae = mean_absolute_error(test1.quantity, y_hat_avg.pred_column)
    del y_hat_avg['SES']
    return y_hat_avg,rms,mae


def holts_linear(input_df, kunag,matnr,n,sl,ss):
    i = 0
    lst = []
    test1 = train_test_split(df,kunag,matnr,n)[1]
    y_hat_avg = test1.copy()
    for i in range(n,0,-1):
        train,test = train_test_split(df,kunag,matnr,i)
        dd= np.asarray(train["quantity"])
        fit1 = Holt(np.asarray(train['quantity'])).fit(smoothing_level = sl,smoothing_slope = ss)
        y_hat_avg['Holt_linear'] = fit1.forecast(len(test1))
        pred = y_hat_avg['Holt_linear']
        lst.append(pred.iloc[-1])

    pd.DataFrame(lst)
    y_hat_avg['pred_column']=lst
    plt.figure(figsize=(12,8))
    plt.plot( train.set_index("date")['quantity'], label='Train',marker = '.')
    plt.plot(test1.set_index("date")['quantity'], label='Test',marker = '.')
    plt.plot(y_hat_avg.set_index("date")['pred_column'], label='Holts linear',marker = '.')
    plt.legend(loc='best')
    plt.title("Holts linear")
    plt.show()
    rms = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_column))
    mae = mean_absolute_error(test1.quantity, y_hat_avg.pred_column)
    del y_hat_avg['Holt_linear']
    return y_hat_avg,rms,mae



def holts_winter(input_df, kunag,matnr,n):
    i = 0
    lst = []
    test1 = train_test_split(df,kunag,matnr,n)[1]
    y_hat_avg = test1.copy()
    for i in range(n,0,-1):
        train,test = train_test_split(df,kunag,matnr,i)
        dd= np.asarray(train["quantity"])
        fit1 = ExponentialSmoothing(np.asarray(train['quantity']) ,seasonal_periods=4 ,trend='add', seasonal='add',).fit()
        y_hat_avg['Holt_Winter'] = fit1.forecast(len(test1))
        pred = y_hat_avg['Holt_Winter']
        lst.append(pred.iloc[-1])

    pd.DataFrame(lst)
    y_hat_avg['pred_column']=lst
    plt.figure(figsize=(12,8))
    plt.plot( train.set_index("date")['quantity'], label='Train',marker = '.')
    plt.plot(test1.set_index("date")['quantity'], label='Test',marker = '.')
    plt.plot(y_hat_avg.set_index("date")['pred_column'], label='Holts Winter',marker = '.')
    plt.legend(loc='best')
    plt.title("Holts Winter")
    plt.show()
    rms = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_column))
    mae = mean_absolute_error(test1.quantity, y_hat_avg.pred_column)
    del y_hat_avg['Holt_Winter']
    return y_hat_avg,rms,mae


def sarima1(input_df, kunag,matnr,n,p,d,q):
    i = 0
    lst = []
    test1 = train_test_split(df,kunag,matnr,n)[1]
    y_hat_avg = test1.copy()
    start_date = str(test1["date"][:1])
    end_date = str(test1["date"][-1:])
    for i in range(n,0,-1):
        train,test = train_test_split(df,kunag,matnr,i)
        dd= np.asarray(train["quantity"])
        fit1 = sm.tsa.statespace.SARIMAX(train["quantity"], order=(p, d,q),enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit()    
        y_hat_avg['SARIMA'] = fit1.predict(1)
        pred = y_hat_avg['SARIMA']
        lst.append(pred.iloc[-1])

    pd.DataFrame(lst)
    y_hat_avg['pred_column']=lst
    y_hat_avg['pred_column'][:1] = lst[1]
    plt.figure(figsize=(12,8))
    plt.plot( train.set_index("date")['quantity'], label='Train',marker = '.')
    plt.plot(test1.set_index("date")['quantity'], label='Test',marker = '.')
    plt.plot(y_hat_avg.set_index("date")['pred_column'], label='SARIMA',marker = '.')
    plt.legend(loc='best')
    plt.title("SARIMA")
    plt.show()
    rms = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_column))
    mae = mean_absolute_error(test1.quantity, y_hat_avg.pred_column)
    del y_hat_avg['SARIMA']
    return y_hat_avg,rms,mae


def sarima(input_df, kunag,matnr,n,p,d,q):
    i = 0
    lst = []
    test1 = train_test_split(df,kunag,matnr,n)[1]
    y_hat_avg = test1.copy()
    start_date = str(test1["date"][:1])
    end_date = str(test1["date"][-1:])
    order = (p, d,q)
    for i in range(n,0,-1):
        train,test = train_test_split(df,kunag,matnr,i)
        dd= np.asarray(train["quantity"])
        fit1 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order,enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit()    
        pred = fit1.predict(1)
        lst.append(pred.iloc[-1])

    pd.DataFrame(lst)
    y_hat_avg['pred_column']=lst
    plt.figure(figsize=(12,8))
    plt.plot( train.set_index("date")['quantity'], label='Train',marker = '.')
    plt.plot(test1.set_index("date")['quantity'], label='Test',marker = '.')
    plt.plot(y_hat_avg.set_index("date")['pred_column'], label='SARIMA' + "_" + str(order),marker = '.')
    plt.legend(loc='best')
    plt.title("SARIMA")
    plt.show()
    rms = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_column))
    mae = mean_absolute_error(test1.quantity, y_hat_avg.pred_column)
    return y_hat_avg,rms,mae


def prophet(input_df, kunag,matnr,n,sps):
#     path = "/home/rahul/Downloads/bharat/time_series1/model_graphs/prophet/"
#     folder = path+ "sps_"+str(sps)
#     if not os.path.exists(folder):
#         os.makedirs(folder)
    index = str(kunag) + "-" + str(matnr)
    i = 0
    lst = []
    test1 = train_test_split(df,kunag,matnr,n)[1]
    y_hat_avg = test1.copy()
    start_date = str(test1["date"][:1])
    end_date = str(test1["date"][-1:])
    for i in range(n,0,-1):
        train,test = train_test_split(df,kunag,matnr,i)
        train['ds'] = train.date
        train['y'] = train.quantity
        data = train[["ds", "y"]]
        ##############################################################################
#         seasonality_prior_scale: Parameter modulating the strength of the
#         seasonality model. Larger values allow the model to fit larger seasonal
#         fluctuations, smaller values dampen the seasonality. Can be specified
#         for individual seasonalities using add_seasonality.
        ######################################################################
        m = Prophet(yearly_seasonality = True, seasonality_prior_scale=sps)
        m.fit(data)
        future_data = m.make_future_dataframe(periods=1, freq="W")
        forecast = m.predict(future_data)
        pred = forecast["yhat"]
        lst.append(pred.iloc[-1])

    pd.DataFrame(lst)
    y_hat_avg['prediction']=lst
    plt.figure(figsize=(12,8))
    plt.plot( train.set_index("date")['quantity'], label='Train',marker = '.')
    plt.plot(test1.set_index("date")['quantity'], label='Test',marker = '.')
    plt.plot(y_hat_avg.set_index("date")['prediction'], label='prophet',marker = '.')
    plt.legend(loc='best')
    plt.title("prophet")
#     plt.savefig(folder +"/" + 'Graph_{}.png'.format(index), format="PNG")  
    plt.show()
    rms = sqrt(mean_squared_error(test1.quantity, y_hat_avg.prediction))
    mae = mean_absolute_error(test1.quantity, y_hat_avg.prediction)
    return y_hat_avg,rms,mae


def croston_tsb(df,kunag,matnr,n,alpha,beta):
    extra_periods=1
    lst = []
    test1 = train_test_split(df,kunag,matnr,n)[1]
    y_hat_avg = test1.copy()
    for i in range(n,0,-1):
        ts,test = train_test_split(df,kunag,matnr,i)
        d = np.array(ts.quantity) # Transform the input into a numpy array
        # Transform the input into a numpy array
        cols = len(d) # Historical p|eriod length
        d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods
        #level (a), probability(p) and forecast (f)
        a,p,f = np.full((3,cols+extra_periods),np.nan)
        # Initialization
        first_occurence = np.argmax(d[:cols]>0)
        a[0] = d[first_occurence]
        p[0] = 1/(1 + first_occurence)
        f[0] = p[0]*a[0]
         # Create all the t+1 forecasts
        for t in range(0,cols): 
            if d[t] > 0:
                a[t+1] = alpha*d[t] + (1-alpha)*a[t] 
                p[t+1] = beta*(1) + (1-beta)*p[t]  
            else:
                a[t+1] = a[t]
                p[t+1] = (1-beta)*p[t]       
            f[t+1] = p[t+1]*a[t+1]
        # Future Forecast
        a[cols+1:cols+extra_periods] = a[cols]
        p[cols+1:cols+extra_periods] = p[cols]
        f[cols+1:cols+extra_periods] = f[cols]
        pred = f
        lst.append(pred[pred.size-1])
        pd.DataFrame(lst)
        df1_ct = pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Period":p,"Level":a,"Error":d-f})
        df1_ct = df1_ct.dropna()
        df1_ct['date'] = ts["date"]
    y_hat_avg['Forecast']=lst
    rms = sqrt(mean_squared_error(test1.quantity, y_hat_avg.Forecast))
    mae = mean_absolute_error(test1.quantity, y_hat_avg.Forecast)

    plt.figure(figsize=(12,8))
    plt.plot( ts.set_index("date")['quantity'], label='Train',marker = '.')
    plt.plot(test1.set_index("date")['quantity'], label='Test',marker = '.')
    plt.plot(y_hat_avg.set_index("date")['Forecast'][-16:], label='Croston TSB',marker = '.')
    plt.legend(loc='best')
    plt.title("Croston TSB")
    plt.show()
    return y_hat_avg,rms,mae