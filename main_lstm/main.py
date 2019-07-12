import os
import time
from model import *
from params import *

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

#import collections
#orderedDict = collections.OrderedDict()
from collections import OrderedDict



#folder to save model graphs
if not os.path.exists(folder):
        os.makedirs(folder)

#dataframe in which results are to be saved
main_df = pd.DataFrame(columns =["kunag","matnr","lstm_mae","lstn_rms"])

cnt = 1       
for i in range (1,n):
            s = time.time()
            kunag = int(bucket["kunag"].iloc[i])
            matnr = int(bucket["matnr"].iloc[i])
            print("count",cnt)
            print("index : ",kunag,",",matnr)

            mae,rms = lstm(df,kunag,matnr,lag,epoch,batch,folder)
            result_df =pd.DataFrame(OrderedDict({"kunag" : [kunag],"matnr" : [matnr],
                   "lstm_mae" : [round(mae,3)],
                    "lstn_rms" : [round(rms,3)]}))

            main_df = main_df.append(result_df, ignore_index = True) 
            export_csv = main_df.to_csv (r'results/'+result_csv,sep = ",", index = None, header=True)

            print(result_df)

            cnt+=1
            e = time.time()
            print("Time taken to run " + str(epoch) + " epoch with batch size " + str(batch) + " : ",e-s)
