from models import *

bucket = pd.read_csv("bucket.csv")
df = pd.read_csv("4200_C005_2019_03_03.tsv", sep=',', header=None)
df.columns = ["kunag", "matnr", "date", "quantity","price"]

main_df = pd.DataFrame(OrderedDict({"kunag" : ["kunag"],
                        "matnr" : ["matnr"],
                        "Naive" : ["-"],
                       "Average" : ["-"],
                       "Moving Average" : ["-"],
                       "SES" : ["-"],
                       "Holts Linear" : ["-"],
                       "Holts Winter" : ["-"],
                       "Arima" : ["-"]}))

cnt = 0
for i in range (1,50):
            kunag = int(bucket["kunag"].iloc[i])
            matnr = int(bucket["matnr"].iloc[i])
            print("count",cnt)
            print("index : ",kunag,",",matnr)
            output1,rms1,mae1 = naive(df,kunag,matnr,16)
            output2,rms2,mae2 = average_forecast(df,kunag,matnr,16)
            output3,rms3,mae3 = moving_average(df,kunag,matnr,16,30)
            output4,rms4,mae4 = ses(df,kunag,matnr,16,0.2)
            output5,rms5,mae5 = holts_linear(df,kunag,matnr,16,0.3,0.1)
            output6,rms6,mae6 = holts_winter(df,kunag,matnr,16)
            output7,rms7,mae7 = sarima(df,kunag,matnr,16,2,1,2)
            result_df =pd.DataFrame(OrderedDict({"kunag" : [kunag],
                    "matnr" : [matnr],
                    "Naive" : [round(rms1,3)],
                   "Average" : [round(rms2,3)],
                   "Moving Average" : [round(rms3,3)],
                   "SES" : [round(rms4,3)],
                   "Holts Linear" : [round(rms5,3)],
                   "Holts Winter" : [round(rms6,3)],
                   "Arima" : [round(rms7,3)]}))
            print(result_df)
            main_df = main_df.append(result_df, ignore_index = True) 
            cnt+=1
print(main_df)
