#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas import DataFrame
import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


df = pd.read_csv("4200_C005_2019_03_03.tsv", sep=',', header=None)
df.columns = ["kunag", "matnr", "date", "quantity","price"]
del df['price']
df = df[df["quantity"] >=0]  


# In[ ]:


from datetime import datetime, timedelta 
freq = 0
cnt = 0
lst = []
bucket = pd.DataFrame({"kunag" : [" "],"matnr" : [" "]})
for index,group in df.groupby(["kunag","matnr"]):
    n_df1 = df.groupby(["kunag","matnr"]).get_group((index[0], index[1]))
    n_df1.date = n_df1.date.apply(lambda x : pd.to_datetime(x,format = '%Y%m%d', errors='ignore'))
    n_df1 = n_df1.sort_values('date')
    delta = (pd.to_datetime(n_df1["date"].iloc[-1], format="%Y%m%d") - pd.to_datetime(n_df1["date"].iloc[0], format="%Y%m%d")).days

    if(delta>730):
        last_1yr = n_df1["date"].iloc[-1] - timedelta(days = 365) 
        lst.append((n_df1["date"]>=last_1yr).sum())
        k = (n_df1["date"]>=last_1yr).sum()
        if(k>=26):
            print("index : ",[index])
            print("Count :",cnt)
            print("Time delta :",delta,"days")
            print("last date :",n_df1["date"].iloc[-1])
            print("last 1 year :",last_1yr)
            print("invoice generated in the last 1 year :",k,"\n")
            bucket1 = pd.DataFrame({"kunag" : [index[0]],"matnr" : [index[1]]})
            bucket = bucket.append(bucket1, ignore_index = True)         
    cnt+=1
print(bucket)


# In[ ]:


export_csv = bucket.to_csv (r'bucket1.csv',sep = ",", index = None, header=True)


# In[ ]:




