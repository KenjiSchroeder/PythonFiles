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

target_names 
feature_names =
#filter out fraud (semeron positive)
FilterSemer = DrugData["semer"]
Listtofilter = []
for i in range(0,len(FilterSemer)):
    if FilterSemer[i]== "CL0":
        Listtofilter.append(i)
    else:
        print("I did it!")

cleanData = DrugData.iloc[Listtofilter,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31]]
print(cleanData)
for i in range (13,cleanData.shape[1]):
    for j in range (0,len(cleanData)):
        re1 = re.findall(r'\d+',cleanData.iloc[j,i])
        cleanData.iloc[j,i] = int(re1[0])
print(cleanData)
#End cleanData




plt.figure(figsize=(6, 4)) 
for i in range(len(target_names)):
   plt.scatter(X[y == i, 0], X[y == i, 1], label=target_names[i])
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("Drug dataset: First two features")
plt.legend()
plt.tight_layout()
plt.show() 
plt.savefig("First2 features dataset")

#C:\Users\ICSSA-student\AppData\Local\Microsoft\WindowsApps\python3.13.exe C:/Users/ICSSA-student/Downloads/PythonFiles/CapstoneTesting2.py