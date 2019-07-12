import pandas as pd
#path to data csvs
bucket = pd.read_csv("bucket.csv")
df = pd.read_csv("4200_C005_2019_03_03.tsv", sep=',', header=None)
df.columns = ["kunag", "matnr", "date", "quantity","price"]

#no. of series on which model to run
n = 2

#how many past values to look back
lag = 16

#lstm params
epoch = 5
batch = 1
lstm_units = 10

#folder to save graphs
folder = 'graphs/'+ str(epoch)+"_"+str(batch)+"_one_layer"

#csv filename to save rms and mae
result_csv = "lstm_mae_rms_1-101.csv"
