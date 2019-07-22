from models import *
from params import *
import datetime
import os,sys,time
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


def plot(df,kunag,matnr,n,folder):
        test1 = train_test_split(df,kunag,matnr,16)[1]
        train = train_test_split(df,kunag,matnr,16)[0]
        plt.figure(figsize=(18,10))
        plt.plot(train.set_index("date")['quantity'], label='Train',marker = '.')
        plt.plot(test1.set_index("date")['quantity'], label='Test',marker = '.')
       
        output1,rms1,mae1 = arima(df,kunag,matnr,16,p,d,q)
        output2,mae2 = lstm(df,kunag,matnr,epoch,batch,order,lstm_units,verb,folder, train_=True)

        y_hat_avg1 = output1
        plt.plot(y_hat_avg1.set_index("date")['pred_column'], label='0,1,1' ,marker = '.')
        y_hat_avg2 = output2
        plt.plot(y_hat_avg2[0], label='lstm' ,marker = '.')            
        plt.legend(loc='best')
        plt.title("Arima_lstm Comparision")
        index = str(kunag) + "-" + str(matnr)
        plt.savefig(folder +"/" + 'Graph_{}.png'.format(index), format="PNG")  
        return mae1,mae2
