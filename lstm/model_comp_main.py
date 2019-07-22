# from lstm_pretrained model import *
import datetime
import os,sys,time
import numpy as np
import pandas as pd
from math import sqrt
import dateutil.parser
from pandas import DataFrame
import statsmodels.api as sm
import matplotlib.pyplot as plt
from preprocess import n_series
from collections import OrderedDict
from preprocess import train_test_split
from datetime import datetime, timedelta 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.layers import LSTM
from keras.layers import Dense  
from keras.layers import Dropout
from keras.models import Sequential  
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)





def n_series(input_df,kunag,matnr):
    input_df = input_df[input_df["quantity"] >=0]    
    df = input_df.copy()
    n_df1 = df[(df["kunag"] == kunag) & (df["matnr"] == matnr)]
    n_df1.date = n_df1.date.apply(lambda x : pd.to_datetime(x,format = '%Y%m%d', errors='ignore'))
    n_df1 = n_df1.sort_values('date')
    n_df1.set_index('date',inplace=True)
    weekly_resampled_data = n_df1.quantity.resample('W').sum() 
    weekly_resampled_data = weekly_resampled_data.replace(np.nan, 0)
    individual_series = weekly_resampled_data.to_frame()
    individual_series = individual_series.reset_index()
    return individual_series


def train_test_split(input_df,kunag,matnr,i):
    data= n_series(input_df,kunag,matnr)
    train = data[0:-i]    
    test = data[-i:]
    return train,test


def arima(input_df, kunag,matnr,n,p,d,q):    
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
    rms = sqrt(mean_squared_error(test1.quantity, y_hat_avg.pred_column))
    mae = mean_absolute_error(test1.quantity, y_hat_avg.pred_column)
    return y_hat_avg,rms,mae




def lstm(df,kunag,matnr,epoch,batch,order,lstm_units,verb,folder, train_=True):
    train,test=  train_test_split(df,kunag,matnr,16)
    dataframe = n_series(df,kunag,matnr)
    index = str(kunag)+"_"+str(matnr)
    order = order
    test_points = 16

    
    df_testing_complete = dataframe[-16:]
    test_predictions = []
    df = dataframe[:-test_points]
    df_training_complete = df
    df_training_processed = df_training_complete.iloc[:, 1:2].values  

    scaler = MinMaxScaler(feature_range = (0, 1))
    df_training_scaled = scaler.fit_transform(df_training_processed)  

    for k in range(0, test_points):
        if (k==0) and (train_):
            print("Training model for " + str(index)+ "...")
            df = dataframe[:-test_points + k]
            df_training_complete = df
            df_training_processed = df_training_complete.iloc[:, 1:2].values  

            scaler = MinMaxScaler(feature_range = (0, 1))
            df_training_scaled = scaler.fit_transform(df_training_processed)  

            features_set = []  
            labels = []
            for i in range(order+1 , df.shape[0]):  
                features_set.append(df_training_scaled[i-order:i, 0])
                labels.append(df_training_scaled[i, 0])


            features_set, labels = np.array(features_set), np.array(labels)  
            features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))  
    #         print(features_set.size)
            model = Sequential()  
            model.add(LSTM(units=lstm_units, return_sequences=False, input_shape=(features_set.shape[1], 1)))  
            model.add(Dropout(0.2))  

            model.add(Dense(units = 1))  
            model.compile(optimizer = 'adam', loss = 'mean_squared_error')  

            model.fit(features_set, labels, epochs = epoch, batch_size = batch,verbose = verb)  
            model_json = model.to_json()
            with open("pretrained/weights/model_("+str(index)+").json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("pretrained/weights/model_("+str(index)+").h5")
            print("Saved model to disk")            
            # load json and create model
            json_file = open("pretrained/weights/model_("+str(index)+").json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("pretrained/weights/model_("+str(index)+").h5")
            print("Loaded model from disk")
            loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        elif k==0 and not train_:
               # load json and create model
            json_file = open("pretrained/weights/model_("+str(index)+").json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("pretrained/weights/model_("+str(index)+").h5")
            print("Loaded model from disk")
            # evaluate loaded model on test data
            loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                          
        df_testing_processed = df_testing_complete.iloc[k:k+1, 1:2].values 
        df_total = pd.concat((df_training_complete['quantity'], df_testing_complete['quantity']), axis=0)  
        test_inputs = df_total[len(df_total) - len(df_testing_complete) - order + k:].values    
        test_inputs = test_inputs.reshape(-1,1)  
        test_inputs = scaler.transform(test_inputs) 
        test_features = []   
        test_features.append(test_inputs[1:order+1, 0])
        test_features = np.array(test_features)  
        test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

        predictions = loaded_model.predict(test_features)
        predictions = scaler.inverse_transform(predictions)  
        test_predictions.append(predictions)
   
    test_predictions_1 = [i[0][0] for i in test_predictions]
    df_c = pd.DataFrame(data=test_predictions_1)
    test.reset_index(inplace = True)
    pred = pd.concat([test, df_c], axis=1, join_axes=[test.index])
    pred.set_index("date",inplace = True)

    pred.drop(['index', 'quantity'], axis=1, inplace=True)
    test.drop(['index'], axis=1, inplace=True)


    mae = mean_absolute_error(test.quantity, pred[0])
    rms = sqrt(mean_squared_error(test.quantity, pred[0]))
    print("mae :",mae)
    print("rms :",rms)

    dataframe.set_index('date',inplace=True)
    train.set_index('date',inplace=True)
    test.set_index('date',inplace=True)


#     plt.figure(figsize=(16,8))
#     plt.plot(dataframe, marker='.', color='blue')
#     plt.plot(train, marker='.', color='blue',label = "Train")
#     plt.plot(test, marker='.', color='orange',label = "Test")
#     plt.plot(pred,marker = ".",color = 'green',label = "Prediction")
#     plt.xlabel("time")
#     plt.ylabel('quantity')
#     plt.legend(loc='best')
#     plt.title("batch : " + str(batch)+"_" + "epochs : " + str(epoch))
#     plt.savefig(folder+"/" + 'Graph_{}_{}_{}_{}.png'.format(index,batch,epoch,lstm_units), format="PNG")  
    return pred,mae
    # plt.show()
    

import os
import time
# from weights import *
from params import *
from keras.models import model_from_json


import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

#import collections
#orderedDict = collections.OrderedDict()
from collections import OrderedDict

import pandas as pd
#path to data csvs
bucket = pd.read_csv("bucket.csv")
df = pd.read_csv("4200_C005_2019_03_03.tsv", sep=',', header=None)
df.columns = ["kunag", "matnr", "date", "quantity","price"]

# #no. of series on which model to run
# n = 101

# #how many past values to look back
# order = 52

# #lstm params
# epoch = 200
# batch = 1
# lstm_units = 10

#folder to save graphs
folder = 'pretrained/graphs_weights/EBOL_'+ str(epoch)+"_"+str(batch)+"_"+str(order)+"_"+str(lstm_units)+"/"

#csv filename to save rms and mae
result_csv = "EBOL_"+ str(epoch)+"_"+str(batch)+"_"+str(order)+"_"+str(lstm_units)+".csv"




# epoch = 200
# batch = 1
# n = 101
# folder to save model graphs
if not os.path.exists(folder):
        os.makedirs(folder)


def plot(df,kunag,matnr,n,folder):
        test1 = train_test_split(df,kunag,matnr,16)[1]
        train = train_test_split(df,kunag,matnr,16)[0]
        plt.figure(figsize=(18,10))
        plt.plot(train.set_index("date")['quantity'], label='Train',marker = '.')
        plt.plot(test1.set_index("date")['quantity'], label='Test',marker = '.')

        
        
        output1,rms1,mae1 = arima(df,kunag,matnr,16,p,d,q)
        output2,mae2 = lstm(df,kunag,matnr,epoch,batch,order,lstm_units,verb,folder, train_=True)


        y_hat_avg1 = output1
#         print("y_hat_avg1 :",y_hat_avg1)
        plt.plot(y_hat_avg1.set_index("date")['pred_column'], label='0,1,1' ,marker = '.')
        y_hat_avg2 = output2
        plt.plot(y_hat_avg2[0], label='lstm_200_1' ,marker = '.')
             
        plt.legend(loc='best')
        plt.title("Arima_lstm Comparision")
        index = str(kunag) + "-" + str(matnr)
        plt.savefig(folder +"/" + 'Graph_{}.png'.format(index), format="PNG")  
        return mae1,mae2
#         plt.show()
    
    
    
    
    
    
import os
import time
# from weights import *
from params import *
from keras.models import model_from_json


import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


from collections import OrderedDict



#folder to save graphs
folder = 'pretrained/comp_graphs/EBOL_comp_'+ str(epoch)+"_"+str(batch)+"_"+str(order)+"_"+str(lstm_units)+"/"

#csv filename to save rms and mae
result_csv = "EBOL_comps"+ str(epoch)+"_"+str(batch)+"_"+str(order)+"_"+str(lstm_units)+".csv"


if not os.path.exists(folder):
        os.makedirs(folder)

#dataframe in which results are to be saved
main_df = pd.DataFrame(columns =["kunag","matnr","lstm_mae","lstn_rms"])

import os, sys
import time
start = time.time()
bucket = pd.read_csv("bucket.csv")
df = pd.read_csv("4200_C005_2019_03_03.tsv", sep=',', header=None)
df.columns = ["kunag", "matnr", "date", "quantity","price"]
p,d,q = 0,1,1 # arima parameters

# order = (p,q,d)

cnt = 0

main_df = pd.DataFrame(columns =["kunag","matnr","lstm","arima011"])
filename = "comps"+str((n1,n2))+"_"+str(epoch)+"_"+str(batch)+"_"+str(order)+"_"+str(lstm_units)

for i in range (n1,n2):
            kunag = int(bucket["kunag"].iloc[i])
            matnr = int(bucket["matnr"].iloc[i])
            index = str(kunag) +"_"+ str(matnr)
            cnt+=1
            print("count",cnt)
            print("index : ",index)
            mae1,mae2 = plot(df,kunag,matnr,n,folder)


            result_df = pd.DataFrame(OrderedDict({"kunag" : [kunag],"matnr" : [matnr],
                       "arima011" : [round(mae1,3)],
                       "lstm" : [round(mae2,3)]}))

#             print(result_df)
            main_df = main_df.append(result_df, ignore_index = True) 
            export_csv = main_df.to_csv (r'pretrained/weight_results/'+filename+'_.csv',sep = ",", index = None, header=True)
            end = time.time()
            print("Time Taken : ",end - start)
