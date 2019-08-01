import pandas as pd
import numpy as np
main_df = pd.read_csv("4200_C005_2019_03_03.tsv", sep = ",", header = None)
main_df.columns  = ["kunag","matnr","date","quantity","price"]
del main_df["price"]

#taking only first 1000 rows
main_df = main_df.head(1000)

#removing negative quantities
main_df = main_df[main_df.quantity > 0]

#groupby customer and material number
main_df_groups = main_df.groupby(["kunag","matnr"])

#changing datetime fromat
main_df["date"] = main_df["date"].apply(lambda x: pd.to_datetime(x, format="%Y%m%d"))

#minimum and maximum date
main_df_date = main_df_groups["date"].agg([min,max]).reset_index()

#no of days in a customer material series, (history of series)
main_df_date["diff_in_days"] = main_df_date["max"]- main_df_date["min"]

#changing days in a numerical format
main_df_date["diff_in_days"] = main_df_date["diff_in_days"].apply(lambda x : x.days)

current_date = "20190725"
current_date = pd.to_datetime(curr_date , format = "%Y%m%d")


#
main_df_frequency = main_df.groupby(["kunag", "matnr"]).filter(lambda x: x["date"].between(current_date, current_date-pd.to_timedelta(92, unit='d')).sum()>0)


# In[ ]:


main_df_frequency.shape


# In[ ]:


main_df_frequency = main_df.groupby(["kunag", "matnr"]).filter(lambda x: x["date"].between(current_date, current_date-pd.to_timedelta(365, unit='d')).sum()>4)


# In[ ]:


main_df_static_frequency = main_df.groupby(["kunag", "matnr"]).filter(lambda x: (x["date"] < (current_date-pd.to_timedelta(365, unit='d'))).sum() > 0)


# In[ ]:


main_df_static_frequency.head(10)


# In[ ]:


main_df_static_frequency = main_df_static_frequency.loc[main_df_static_frequency["date"] > current_date - pd.to_timedelta(365,unit = "d")]
main_df_static_frequency = main_df_static_frequency.groupby(["kunag","matnr"])["quantity"].count().reset_index()
main_df_static_frequency = main_df_static_frequency.rename({"quantity" : 'frequency'},axis = 1)


# In[ ]:


main_df_static_frequency.head()


# In[ ]:


main_df_dynamic_factor = main_df.groupby(["kunag","matnr"]).filter(lambda x : (x["date"] < (current_date - pd.to_timedelta(365,unit = "d"))).sum()==0)
main_df_dynamic_factor = main_df_dynamic_factor.groupby(["kunag","matnr"])["date"].min().reset_index()
main_df_dynamic_factor["factor"] = main_df_dynamic_factor.date.apply(lambda x : 365/(current_date - x).days)
main_df_dynamic_factor.head()


# In[ ]:


main_df_dynamic_frequency = main_df.groupby(["kunag", "matnr"]).filter(lambda x: (x["date"] < (current_date-pd.to_timedelta(365, unit='d'))).sum() == 0)
main_df_dynamic_frequency = main_df_dynamic_frequency.groupby(["kunag", "matnr"])["quantity"].count().reset_index()
main_df_dynamic_frequency = main_df_dynamic_frequency.rename({"quantity": "frequency"}, axis=1)
main_df_dynamic_frequency.head()


# In[ ]:


main_df_dynamic_frequency["frequency"] = main_df_dynamic_frequency["frequency"]*main_df_dynamic_factor["factor"]


# In[ ]:


main_df_dynamic_frequency


# In[ ]:


main_df_comp_frequency = pd.concat([main_df_static_frequency, main_df_dynamic_frequency], axis =0).reset_index(drop=True)


# In[ ]:


main_df_comp_frequency[:5]


# In[ ]:


main_df_final = pd.merge(main_df_comp_frequency, main_df_date, on=["kunag", "matnr"], how="inner")


# In[ ]:


main_df_final.head()


# In[ ]:


def ass_cat(frequency, days):
    if days>250 and frequency <26:
        return "cat I"
    else:
        return "None" 


# In[ ]:


main_df_final["category"] = main_df_final[['frequency','diff_in_days']].apply(lambda x: ass_cat(x.frequency,x.diff_in_days), axis=1)
main_df_final


# In[ ]:



empty_dict = dict(zip(range(5), [[]]*5 ))
print(empty_dict)
lst = [0,2,1,3,4,2,3,4,2,1]
key=3
empty_dict[3].append("h")
print(empty_dict)


# In[ ]:


myDict = {} 
# Adding list as value 
for i in range(0,len(lst)):
    
myDict[lst[0]] = ["h"] 
myDict[lst[1]] = ["e"]  
  
print(myDict) 


# In[ ]:


class my_dictionary(dict): 
  
    # __init__ function 
    def __init__(self): 
        self = dict() 
          
    # Function to add key:value 
    def add(self, key, value): 
        self[key] = value 
  
# Main Function 
dict_obj = my_dictionary() 
print(dict_obj)
dict_obj.add(1, 'Geeks') 
dict_obj.add(2, 'forGeeks') 
  
print(dict_obj) 


# In[ ]:


[[]]*8


# In[ ]:


list(range(5))


# In[ ]:


zip(range(5), [[]]*5 )


# In[ ]:


dict(zip(range(5), [[]]*5 ))


# In[ ]:


empty_dict = dict(zip(range(5), [[]]*5 ))
print(empty_dict)


# In[ ]:


empty_dict = {}

lst = [0,2,1,3,4,2,3,4,2,1]
for x,i in enumerate(list('helloworld')):
    key=lst[x]
    
    if key not in empty_dict.keys():
        empty_dict[key] = []

    empty_dict[key].append(i)

print(empty_dict)


# In[ ]:




