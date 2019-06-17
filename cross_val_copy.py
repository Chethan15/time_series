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


def main(input_df, kunag,matnr,n,l,roll,alpha,p,d,q):
    index = str(kunag) + "-" + str(matnr)
    dfw = n_series(df,kunag,matnr)
    test1 = test(df,kunag,matnr)
    mae = []
    order1 = (0,1,1)
    order2 = (0,1,2)
    order3 = (1,1,1)
    lst = []
#     print("count of predictions done")
    for i in range (0,16):
        k = n-i


        lst1 = []
        lst2 = []
        lst3 = []
        lst4 = []
        lst5 = []
        nlst = []

        train1,cv1 = train_cv_split(df,kunag,matnr,k,l)
        y_hat_avg = cv1.copy()
        for j in range(k,l,-1):        
            train,cv = train_cv_split(df,kunag,matnr,j,l)
            dd= np.asarray(train["quantity"])
            
            #model 1 : MOVING AVERAGE
            y_hat_avg['moving_avg_forecast'] = train['quantity'].rolling(roll).mean().iloc[-1]
            pred1 = y_hat_avg['moving_avg_forecast']
            lst1.append(pred1.iloc[-1])

            #model 2 : SES
            fit1 = SimpleExpSmoothing(np.asarray(train['quantity'])).fit(smoothing_level=alpha,optimized=False)
            y_hat_avg['SES'] = fit1.forecast(len(cv1))
            pred2 = y_hat_avg['SES']
            lst2.append(pred2.iloc[-1])

            #model 3 : ARIMA
            fit2 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order1,enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit()    
            pred3 = fit2.predict(1)
            lst3.append(pred3.iloc[-1])
            
            #model 4 : ARIMA
            fit3 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order2,enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit()    
            pred4 = fit3.predict(1)
            lst4.append(pred4.iloc[-1])
            
            #model 5 : ARIMA
            fit4 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order3,enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit()    
            pred5 = fit4.predict(1)
            lst5.append(pred4.iloc[-1])
            
            
        pd.DataFrame(lst1)
        pd.DataFrame(lst2)
        pd.DataFrame(lst3)
        pd.DataFrame(lst4)
        pd.DataFrame(lst5)


        y_hat_avg['pred_ma']=lst1
        y_hat_avg['pred_ses']=lst2
        y_hat_avg['pred_arima011']=lst3
        y_hat_avg['pred_arima012']=lst4
        y_hat_avg['pred_arima111']=lst5

        rms1 = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_ma))
        mae1 = mean_absolute_error(test1.quantity, y_hat_avg.pred_ma)
        rms2 = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_ses))
        mae2 = mean_absolute_error(test1.quantity, y_hat_avg.pred_ses)
        rms3 = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_arima011))
        mae3 = mean_absolute_error(test1.quantity, y_hat_avg.pred_arima011)       
        rms4 = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_arima012))
        mae4 = mean_absolute_error(test1.quantity, y_hat_avg.pred_arima012)
        rms5 = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_arima111))
        mae5 = mean_absolute_error(test1.quantity, y_hat_avg.pred_arima111)
#         print(mae1,mae2,mae3)
        nlst = [mae1,mae2,mae3,mae4,mae5]
#         print(nlst)
        m = min(nlst)
#         print(m)
        if(mae1==m):
#             print("mae1:",mae1)
            y_hat_avg['moving_avg_forecast'] = train['quantity'].rolling(roll).mean().iloc[-1]
            pred = y_hat_avg['moving_avg_forecast']
        elif(mae2==m):
#             print("mae2:",mae2)
            fit1 = SimpleExpSmoothing(np.asarray(train['quantity'])).fit(smoothing_level=alpha,optimized=False)
            y_hat_avg['SES'] = fit1.forecast(len(cv1))
            pred = y_hat_avg['SES']
        elif(mae3==m):
#             print("mae3:",mae3)
            fit2 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order1,enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit()    
            pred = fit2.predict(1)
        elif(mae4==m):
#             print("mae4:",mae4)
            fit3 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order2,enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit()    
            pred = fit3.predict(1)
        elif(mae5==m):
#             print("mae5:",mae5)
            fit4 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order3,enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit()    
            pred = fit4.predict(1)

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

