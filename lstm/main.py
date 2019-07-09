import os
from lstm import *
import time
import logging
import collections
orderedDict = collections.OrderedDict()
from collections import OrderedDict


bucket = pd.read_csv("/home/bharat/proj/time_series1/bucket.csv")
df = pd.read_csv("/home/bharat/proj/time_series1/4200_C005_2019_03_03.tsv", sep=',', header=None)
df.columns = ["kunag", "matnr", "date", "quantity","price"]

main_df = pd.DataFrame(columns =["kunag","matnr","lstm_mae","lstn_rms"])



cnt = 0
for i in range (1,500):
            kunag = int(bucket["kunag"].iloc[i])
            matnr = int(bucket["matnr"].iloc[i])
            print("count",cnt)
            print("index : ",kunag,",",matnr)

            mae,rms = lstm(df,kunag,matnr)
            result_df =pd.DataFrame(OrderedDict({"kunag" : [kunag],"matnr" : [matnr],
                   "lstm_mae" : [round(mae,3)],
                    "lstn_rms" : [round(rms,3)]}))

            export_csv = main_df.to_csv (r'lstm_mae_rms.csv',sep = ",", index = None, header=True)

#                 print(result_df)
            main_df = main_df.append(result_df, ignore_index = True) 
            cnt+=1
            print(cnt)




