import numpy as np
import pandas as pd
import datetime
import numpy as np
import pandas as pd
import datetime

##n_series to find the individual series for a kunag,matnr combination
def n_series(input_df,kunag,matnr):
    #removing the negative value rows
    input_df = input_df[input_df["quantity"] >=0]    
    df = input_df.copy()
    
    #finding a single customer(kunag),material(matnr) combination on the basis of input provided
    n_df1 = df[(df["kunag"] == kunag) & (df["matnr"] == matnr)]
    
    #parsing the date column in a datetime format
    n_df1.date = n_df1.date.apply(lambda x : pd.to_datetime(x,format = '%Y%m%d', errors='ignore'))
    #sort date start to end
    n_df1 = n_df1.sort_values('date')
    n_df1.set_index('date',inplace=True)
    
    #sampling the data on weekly basis (index is set to date first in the above step to do weekly sampling
    weekly_resampled_data = n_df1.quantity.resample('W').sum() 
    weekly_resampled_data = weekly_resampled_data.replace(np.nan, 0)
    individual_series = weekly_resampled_data.to_frame() 
    
    #resetting index so that date can be used as column
    individual_series = individual_series.reset_index()
    return individual_series


#splitting the individual series basesd on the number of weeks (i, is for dyanamic point forecasts)
def train_test_split(input_df,kunag,matnr,i):
    data= n_series(input_df,kunag,matnr)
    train = data.iloc[0:-i]    
    test = data.iloc[-i:]
    return train,test