import numpy as np
import pandas as pd
import datetime
import numpy as np
import pandas as pd
import datetime
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
from fbprophet import Prophet
from preprocess_cv import train_test_split
from preprocess_cv import train_cv_split
from preprocess_cv import test

df = pd.read_csv("4200_C005_2019_03_03.tsv", sep=',', header=None)
df.columns = ["kunag", "matnr", "date", "quantity","price"]


def moving_average1(input_df, kunag,matnr,n,l,roll):
    index = str(kunag) + "-" + str(matnr)
    dfw = n_series(df,kunag,matnr)
    test1 = test(df,kunag,matnr)
    mae = []
    for i in range (0,l):
        k = n-i
        lst = []
        train1,cv1 = train_cv_split(df,kunag,matnr,k,l)
        y_hat_avg = cv1.copy()
        for j in range(k,l,-1):        
            train,cv = train_cv_split(df,kunag,matnr,j,l)
            dd= np.asarray(train["quantity"])
            y_hat_avg['moving_avg_forecast'] = train['quantity'].rolling(roll).mean().iloc[-1]
            pred = y_hat_avg['moving_avg_forecast']
            lst.append(pred.iloc[-1])
        pd.DataFrame(lst)
        y_hat_avg['pred_column']=lst
        rms = sqrt(mean_squared_error(cv1.quantity, y_hat_avg.pred_column))
        mae1 = mean_absolute_error(cv1.quantity, y_hat_avg.pred_column)
        mae.append(mae1)
        del y_hat_avg['moving_avg_forecast']
        l=l-1
    return mae


def ses1(input_df, kunag,matnr,n,l,alpha):
    index = str(kunag) + "-" + str(matnr)
    dfw = n_series(df,kunag,matnr)
    test1 = test(df,kunag,matnr)
    mae = []
    for i in range (0,l):
        k = n-i
        lst = []
        train1,cv1 = train_cv_split(df,kunag,matnr,k,l)
        y_hat_avg = cv1.copy()
        for j in range(k,l,-1):        
            train,cv = train_cv_split(df,kunag,matnr,j,l)
            dd= np.asarray(train["quantity"])
            fit2 = SimpleExpSmoothing(np.asarray(train['quantity'])).fit(smoothing_level=alpha,optimized=False)
            y_hat_avg['SES'] = fit2.forecast(len(cv1))
            pred = y_hat_avg['SES']
            lst.append(pred.iloc[-1])
        pd.DataFrame(lst)
        y_hat_avg['pred_column']=lst
        rms = sqrt(mean_squared_error(cv1.quantity, y_hat_avg.pred_column))
        mae1 = mean_absolute_error(cv1.quantity, y_hat_avg.pred_column)
        mae.append(mae1)
        del y_hat_avg['SES']
        l=l-1
    return mae



def sarima1(input_df, kunag,matnr,n,l,p,d,q):
    index = str(kunag) + "-" + str(matnr)
    dfw = n_series(df,kunag,matnr)
    test1 = test(df,kunag,matnr)
    mae = []
    order = (p,d,q)
    for i in range (0,l):
        k = n-i
        lst = []
        train1,cv1 = train_cv_split(df,kunag,matnr,k,l)
        y_hat_avg = cv1.copy()
        for j in range(k,l,-1):        
            train,cv = train_cv_split(df,kunag,matnr,j,l)
            dd= np.asarray(train["quantity"])
            fit1 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order,enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit()    
            pred = fit1.predict(1)
            lst.append(pred.iloc[-1])

        pd.DataFrame(lst)
        y_hat_avg['pred_column']=lst
        rms = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_column))
        mae1 = mean_absolute_error(test1.quantity, y_hat_avg.pred_column)
        mae.append(mae1)
        l=l-1
    return mae


def main1(input_df, kunag,matnr,n,l,roll,alpha,p,d,q):
    index = str(kunag) + "-" + str(matnr)
    dfw = n_series(df,kunag,matnr)
    test1 = test(df,kunag,matnr)
    mae = []
    order = (p,d,q)
    lst = []
#     print("count of predictions done")
    for i in range (0,l):
        k = n-i
        
        lst1 = []
        lst2 = []
        lst3 = []
        nlst = []
        train1,cv1 = train_cv_split(df,kunag,matnr,k,l)
        y_hat_avg = cv1.copy()
        for j in range(k,l,-1):        
            train,cv = train_cv_split(df,kunag,matnr,j,l)
            dd= np.asarray(train["quantity"])
            y_hat_avg['moving_avg_forecast'] = train['quantity'].rolling(roll).mean().iloc[-1]
            pred1 = y_hat_avg['moving_avg_forecast']
            lst1.append(pred1.iloc[-1])

            
            fit2 = SimpleExpSmoothing(np.asarray(train['quantity'])).fit(smoothing_level=alpha,optimized=False)
            y_hat_avg['SES'] = fit2.forecast(len(cv1))
            pred2 = y_hat_avg['SES']
            lst2.append(pred2.iloc[-1])

            
            fit1 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order,enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit()    
            pred3 = fit1.predict(1)
            lst3.append(pred3.iloc[-1])                
        pd.DataFrame(lst1)
        pd.DataFrame(lst2)
        pd.DataFrame(lst3)

        y_hat_avg['pred_ma']=lst1
        y_hat_avg['pred_ses']=lst2
        y_hat_avg['pred_arima']=lst3

        rms1 = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_ma))
        mae1 = mean_absolute_error(test1.quantity, y_hat_avg.pred_ma)
        rms2 = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_ses))
        mae2 = mean_absolute_error(test1.quantity, y_hat_avg.pred_ses)
        rms3 = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_arima))
        mae3 = mean_absolute_error(test1.quantity, y_hat_avg.pred_arima)

        nlst = [mae1,mae2,mae3]
        m = min(nlst)

        if(mae1==m):
            y_hat_avg['moving_avg_forecast'] = train['quantity'].rolling(roll).mean().iloc[-1]
            pred = y_hat_avg['moving_avg_forecast']
            
        elif(mae2==m):
            fit2 = SimpleExpSmoothing(np.asarray(train['quantity'])).fit(smoothing_level=alpha,optimized=False)
            y_hat_avg['SES'] = fit2.forecast(len(cv1))
            pred = y_hat_avg['SES']
            
        elif(mae3==m):
            fit1 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order,enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit()    
            pred = fit1.predict(1)

        lst.append(pred.iloc[-1])
        
#         print(len(lst) )
        pd.DataFrame(lst)
    

        l=l-1
    y_hat_avg['pred_column']=lst
#     plt.figure(figsize=(12,8))
#     plt.plot( train.set_index("date")['quantity'], label='Train',marker = '.')
#     plt.plot(cv1.set_index("date")['quantity'], label='Test',marker = '.')
#     plt.plot(y_hat_avg.set_index("date")['pred_main'], label='MAIN' + "_" + str(order),marker = '.')
#     plt.legend(loc='best')
#     plt.title("MAIN")

#     plt.show()
    rms = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_column))
    mae = mean_absolute_error(test1.quantity, y_hat_avg.pred_column)
    del y_hat_avg["moving_avg_forecast"]
    del y_hat_avg["SES"]
#     print(y_hat_avg)
    print("mae :",mae)
    return y_hat_avg,mae


def main(input_df, kunag,matnr,n,l,roll,alpha,p,d,q):
    index = str(kunag) + "-" + str(matnr)
    dfw = n_series(df,kunag,matnr)
    test1 = test(df,kunag,matnr)
    mae = []
    order = (p,d,q)
    lst = []
    print("count of predictions done")
    for i in range (0,16):
        k = n-i
        m = 16
        c = i+1
        print("n :",c)
        lst1 = []
        lst2 = []
        lst3 = []
        nlst = []

        train1,cv1 = train_cv_split(df,kunag,matnr,k,l)
        y_hat_avg = cv1.copy()
        for j in range(k,l,-1):        
            train,cv = train_cv_split(df,kunag,matnr,j,l)
            dd= np.asarray(train["quantity"])
            
            y_hat_avg['moving_avg_forecast'] = train['quantity'].rolling(roll).mean().iloc[-1]
            pred1 = y_hat_avg['moving_avg_forecast']
            lst1.append(pred1.iloc[-1])

            
            fit2 = SimpleExpSmoothing(np.asarray(train['quantity'])).fit(smoothing_level=alpha,optimized=False)
            y_hat_avg['SES'] = fit2.forecast(len(cv1))
            pred2 = y_hat_avg['SES']
            lst2.append(pred2.iloc[-1])

            
            fit1 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order,enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit()    
            pred3 = fit1.predict(1)
            lst3.append(pred3.iloc[-1])
            m = m-1
            
        pd.DataFrame(lst1)
        pd.DataFrame(lst2)
        pd.DataFrame(lst3)
#         print(lst3)
        y_hat_avg['pred_ma']=lst1
        y_hat_avg['pred_ses']=lst2
        y_hat_avg['pred_arima']=lst3

        rms1 = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_ma))
        mae1 = mean_absolute_error(test1.quantity, y_hat_avg.pred_ma)
        rms2 = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_ses))
        mae2 = mean_absolute_error(test1.quantity, y_hat_avg.pred_ses)
        rms3 = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_arima))
        mae3 = mean_absolute_error(test1.quantity, y_hat_avg.pred_arima)
#         print(mae1,mae2,mae3)
        nlst = [mae1,mae2,mae3]
#         print(nlst)
        m = min(nlst)
#         print(m)
        if(mae1==m):
#             print("mae1:",mae1)
            y_hat_avg['moving_avg_forecast'] = train['quantity'].rolling(roll).mean().iloc[-1]
            pred = y_hat_avg['moving_avg_forecast']
        elif(mae2==m):
#             print("mae2:",mae2)
            fit2 = SimpleExpSmoothing(np.asarray(train['quantity'])).fit(smoothing_level=alpha,optimized=False)
            y_hat_avg['SES'] = fit2.forecast(len(cv1))
            pred = y_hat_avg['SES']
        elif(mae3==m):
#             print("mae3:",mae3)
            fit1 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order,enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit()    
            pred = fit1.predict(1)

        lst.append(pred.iloc[-1])
        pd.DataFrame(lst)
        l=l-1   

    y_hat_avg['pred_column']=lst
#     plt.figure(figsize=(12,8))
#     plt.plot( train.set_index("date")['quantity'], label='Train',marker = '.')
#     plt.plot(cv1.set_index("date")['quantity'], label='Test',marker = '.')
#     plt.plot(y_hat_avg.set_index("date")['pred_column'], label='MAIN' + "_" + str(order),marker = '.')
#     plt.legend(loc='best')
#     plt.title("MAIN")

#     plt.show()
    rms = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_column))
    mae = mean_absolute_error(test1.quantity, y_hat_avg.pred_column)
    del y_hat_avg["moving_avg_forecast"]
    del y_hat_avg["SES"]
#     print(y_hat_avg)
    return y_hat_avg,mae

