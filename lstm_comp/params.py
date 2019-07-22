import pandas as pd
#path to data csvs
bucket = pd.read_csv("/home/bharat/proj/time_series1/bucket.csv")
df = pd.read_csv("/home/bharat/proj/time_series1/4200_C005_2019_03_03.tsv", sep=',', header=None)
df.columns = ["kunag", "matnr", "date", "quantity","price"]

#arima parameters
p,d,q = 0,1,1

#no. of series on which model to run
n = 16
n1 = 1
n2 = 31

#how many past values to look back
order = 52

#lstm params
epoch = 100
batch = 64
verb = 2
lstm_units = 5

#folder to save graphs
folder = 'pretrained/comp_graphs/EBOL_comp_'+ str(epoch)+"_"+str(batch)+"_"+str(order)+"_"+str(lstm_units)+"/"

#csv filename to save rms and mae
result_csv = "EBOL_comps"+ str(epoch)+"_"+str(batch)+"_"+str(order)+"_"+str(lstm_units)+".csv"
