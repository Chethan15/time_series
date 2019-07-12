import sys
import datetime
import numpy as np  
import pandas as pd 
from math import sqrt 
from preprocess import *
import matplotlib.pyplot as plt  
from keras.layers import LSTM  
from keras.layers import Dense  
from keras.layers import Dropout
from keras.models import Sequential  
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


def lstm(df,kunag,matnr,lag,epoch,batch,folder):
    train,test=  train_test_split(df,kunag,matnr,16)
    dataframe = n_series(df,kunag,matnr)
    index = str(kunag)+"_"+str(matnr)
    order = lag
    test_points = 16
    df_testing_complete = dataframe[-16:]
    test_predictions = []
    for k in range(0, test_points):
        print("predicting point " +str(k+1) + "...")
        df = dataframe[:-test_points + k]
        df_training_complete = df
        print(df_training_complete.shape)

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

        model = Sequential()  
        model.add(LSTM(units=50, return_sequences=False, input_shape=(features_set.shape[1], 1)))  
        model.add(Dropout(0.2))  
#         model.add(LSTM(units=50, return_sequences=True))  
#         model.add(Dropout(0.2))
#         model.add(LSTM(units=50, return_sequences=True))  
#         model.add(Dropout(0.2))
#         model.add(LSTM(units=50))  
#         model.add(Dropout(0.2))
        model.add(Dense(1))  
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')  

        model.fit(features_set, labels, epochs = epoch, batch_size = 1)  
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
    plt.plot(dataframe, marker='.', color='blue')
    plt.plot(train, marker='.', color='blue',label = "Train")
    plt.plot(test, marker='.', color='orange',label = "Test")
    plt.plot(pred,marker = ".",color = 'green',label = "Prediction")
    plt.xlabel("time")
    plt.ylabel('quantity')
    plt.legend(loc='best')
    plt.title("batch : " + str(batch)+"_" + "epochs : " + str(epoch))
    plt.savefig("/home/bharat/proj/time_series1/lstm/graphs/" + 'Graph_{}.png'.format(index), format="PNG")  
    return mae,rms
    # plt.show()

if __name__ == "__main__":
#     file_path = "/home/bharat/proj/time_series1/4200_C005_2019_03_03.tsv"
#     df = pd.read_csv(file_path, sep=',', header=None)
#     df.columns = ["kunag", "matnr", "date", "quantity","price"]
#     kunag,matnr = 500056565,100278
#     epoch = int(sys.argv[1])
#     batch = int(sys.argv[2])
    lstm(df,kunag,matnr,lag,epoch,batch,folder)
    
