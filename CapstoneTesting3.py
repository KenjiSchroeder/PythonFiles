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
import re

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
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

columns = drug_consumption_quantified.data.ids

print(type(drug_consumption_quantified.data ))

cleanData = pd.DataFrame(cleanData)

'''encoders = {}
for col in df.columns:
   le = LabelEncoder()
   df[col] = le.fit_transform(df[col])
   encoders[col] = le'''

X3 = cleanData.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
y3 = cleanData.iloc[:,23]
print("break0")
print(y3)
y3 = y3.squeeze() #this is just to make the y values into a one-dimensional list.
print("break")
print(y3)
print("aljrhfb asdjlfh askdjcfnzlduih valur")
print(type(y3))
print(type(X3))
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=1)
print("break2")
print(X3)
print(y3)
print("break3")
y3test = drug_consumption_quantified.data.targets
print(y3test)
X3test = drug_consumption_quantified.data.features
print(X3test)
clf = DecisionTreeClassifier(criterion='entropy', random_state=1,)
print(X3_train)
print(y3_train)
clf.fit(X3_train, y3_train)
y3_pred = clf.predict(X3_test)
print("Accuracy:", accuracy_score(y3_test, y3_pred))
plt.figure(figsize=(12, 8))
plot_tree(
clf,
feature_names=drug_consumption_quantified.feature_names,
class_names=drug_consumption_quantified.target_names,
filled=True
)
#plt.show()
#plt.savefig("MaxDepth.png")
for i in range(1,10):
    clf = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=i)
    clf.fit(X3_train, y3_train)
    y3_pred = clf.predict(X3_test)
    print("Accuracy:", accuracy_score(y3_test, y3_pred))
    plt.figure(figsize=(12, 8))
    plot_tree(
    clf,
    feature_names=drug_consumption_quantified.feature_names,
    class_names=drug_consumption_quantified.target_names,
    filled=True
    )
    #plt.show()
    #plt.savefig("MaxDepth"+str(i)+".png")




clf = RandomForestClassifier(n_estimators=100, random_state=1)
clf.fit(X3_train, y3_train)




y3_pred = clf.predict(X3_test)
print("RF Accuracy:", accuracy_score(y3_test, y3_pred))

importances = clf.feature_importances_
features = drug_consumption_quantified.feature_names

plt.figure(figsize=(8, 5))
plt.barh(range(len(importances)), importances, align='center')
plt.yticks(range(len(importances)), features)
plt.xlabel("Feature Importance")
plt.title("Random Forest - Drug Feature Importances")
plt.tight_layout()
plt.show()






#C:\Users\ICSSA-student\AppData\Local\Microsoft\WindowsApps\python3.13.exe C:/Users/ICSSA-student/Downloads/PythonFiles/CapstoneTesting3.py