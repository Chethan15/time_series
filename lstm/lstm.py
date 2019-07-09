import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import mean_squared_error

# file_path = "/home/bharat/proj/time_series1/4200_C005_2019_03_03.tsv"
# df = pd.read_csv(file_path, sep=',', header=None)
# df.columns = ["kunag", "matnr", "date", "quantity","price"]

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



# main_df = pd.DataFrame(columns =["kunag","matnr","111","022"])




def lstm(df,kunag,matnr):
    train,test=  train_test_split(df,kunag,matnr,16)
    dataframe = n_series(df,kunag,matnr)
    index = str(kunag)+"_"+str(matnr)
    order = 16
    test_points = 16
    df_testing_complete = dataframe[-16:]
    test_predictions = []
    for k in range(0, test_points):
        df = dataframe[:-test_points + k]
        df_training_complete = df
        print(df_training_complete.shape)

        df_training_processed = df_training_complete.iloc[:, 1:2].values  

        scaler = MinMaxScaler(feature_range = (0, 1))
        df_training_scaled = scaler.fit_transform(df_training_processed)  

        features_set = []  
        labels = []
        for i in range(order+1 , df.shape[0]):  
            features_set.append(df_training_scaled[i-order+1:i+1, 0])
            labels.append(df_training_scaled[i, 0])


        features_set, labels = np.array(features_set), np.array(labels)  
        features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))  

        model = Sequential()  
        model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))  
        model.add(Dropout(0.2))  
        model.add(LSTM(units=50, return_sequences=True))  
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))  
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))  
        model.add(Dropout(0.2))
        model.add(Dense(units = 1))  
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')  

        model.fit(features_set, labels, epochs = 5, batch_size = 32)  
        df_testing_processed = df_testing_complete.iloc[k:k+1, 1:2].values 

        df_total = pd.concat((df_training_complete['quantity'], df_testing_complete['quantity']), axis=0)  
        test_inputs = df_total[len(df_total) - len(df_testing_complete) - order + k:].values  
        test_inputs = test_inputs.reshape(-1,1)  
        test_inputs = scaler.transform(test_inputs) 

        test_features = []   

        test_features.append(test_inputs[1:order+1, 0])

        test_features = np.array(test_features)  
        test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
        predictions = model.predict(test_features)  
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


    plt.figure(figsize=(16,8))
    plt.plot( dataframe, marker='.', color='blue')
    plt.plot( train, marker='.', color='blue')
    plt.plot( test, marker='.', color='orange')
    plt.plot(pred,marker = ".",color = 'green')
    plt.xlabel("time")
    plt.ylabel('quantity')
    plt.savefig("/home/bharat/proj/time_series1/lstm/graphs/" + 'Graph_{}.png'.format(index), format="PNG")  
    return mae,rms
    # plt.show()