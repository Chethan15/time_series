# from lstm_pretrained model import *
import datetime
import os,sys,time
import numpy as np
import pandas as pd
from pandas import DataFrame
from collections import OrderedDict
from params import *
from plot import *

if not os.path.exists(folder):
        os.makedirs(folder)

#dataframe in which results are to be saved
main_df = pd.DataFrame(columns =["kunag","matnr","lstm_mae","lstn_rms"])

cnt = 0

main_df = pd.DataFrame(columns =["kunag","matnr","lstm","arima011"])
filename = "comps"+str((n1,n2))+"_"+str(epoch)+"_"+str(batch)+"_"+str(order)+"_"+str(lstm_units)

for i in range (n1,n2):
            start = time.time()
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
