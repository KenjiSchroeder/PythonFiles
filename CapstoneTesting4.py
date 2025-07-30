from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo

# fetch dataset 
drug_consumption_quantified = fetch_ucirepo(id=373) 
  
# data (as pandas dataframes) 
X = drug_consumption_quantified.data.features 
y = drug_consumption_quantified.data.targets 
DrugData = drug_consumption_quantified.data.original
#DrugData = pd.DataFrame(drug_consumption_quantified)
# metadata 
print(drug_consumption_quantified.metadata) 
  
# variable information 
print(drug_consumption_quantified.variables) 
print("break")
print(DrugData)
#filter out fraud (semeron positive)
FilterSemer = DrugData["semer"]
Listtofilter = []
for i in range(0,len(DrugData)):
    if FilterSemer[i]== "CL0":
        Listtofilter.append(i)
    else:
        print("I did it!")

cleanData = DrugData.iloc[Listtofilter,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31]]
print(cleanData)
print(len(DrugData))
print(len(cleanData))
print (cleanData.shape[1])
cleanData = cleanData.replace('CL0',0)
cleanData = cleanData.replace('CL1',1)
cleanData = cleanData.replace('CL2',2)
cleanData = cleanData.replace('CL3',3)
cleanData = cleanData.replace('CL4',4)
cleanData = cleanData.replace('CL5',5)
cleanData = cleanData.replace('CL6',6)
print(cleanData)

def findS(con, tar):
   for i, val in enumerate(tar):
       if val == 'Yes':
           specific_h = con[i].copy()
           break
          
   for i, val in enumerate(con):
       if tar[i] == 'Yes':
           for x in range(len(specific_h)):
               if val[x] != specific_h[x]:
                   specific_h[x] = '?'
               else:
                   pass
   return specific_h
# from https://github.com/kevinadhiguna/find-S-algorithm/blob/master/find-s.ipynb

concepts = cleanData.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
target = cleanData.iloc[:,23]
print(findS(concepts,target))