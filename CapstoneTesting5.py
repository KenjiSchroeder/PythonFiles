from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import random as rand

drug_consumption_quantified = fetch_ucirepo(id=373) 
  
# data (as pandas dataframes) 
X = drug_consumption_quantified.data.features 
y = drug_consumption_quantified.data.targets 
DrugData = drug_consumption_quantified.data.original

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
from ucimlrepo import fetch_ucirepo 
from sklearn import datasets
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import re


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
print("break")

from sklearn.utils import resample
class_0 = cleanData[cleanData['heroin'] == 0]
class_1 = cleanData[cleanData['heroin'] > 0]
class_0_downsampled = class_0.sample(n=len(class_1), random_state=1)
cleanDataHeroinFilter = pd.concat([class_0_downsampled, class_1]).sample(frac=1, random_state=1).reset_index(drop=True)
print(cleanDataHeroinFilter)
print(np.mean(cleanData.iloc[:,22]))
print(np.mean(cleanDataHeroinFilter.iloc[:,22]))
print(np.mean(cleanData.iloc[:,23]))
print(np.mean(cleanDataHeroinFilter.iloc[:,23]))
#End cleanData 
print(cleanData.columns.tolist()[18])
print((cleanData["cannabis"] == 0).sum())
print((cleanData["cannabis"] == 1).sum())
print((cleanData["cannabis"] == 2).sum())
print((cleanData["cannabis"] == 3).sum())
print((cleanData["cannabis"] == 4).sum())
print((cleanData["cannabis"] == 5).sum())
print((cleanData["cannabis"] == 6).sum())