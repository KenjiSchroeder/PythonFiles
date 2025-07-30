from ucimlrepo import fetch_ucirepo 
from sklearn import datasets
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# fetch dataset 
drug_consumption_quantified = fetch_ucirepo(id=373) 
  
# data (as pandas dataframes) 
X = drug_consumption_quantified.data.features 
y = drug_consumption_quantified.data.targets 
DrugData = drug_consumption_quantified.data.original

# filter out fraud (semeron positive)
FilterSemer = DrugData["semer"]
Listtofilter = []
for i in range(0, len(DrugData)):
    if FilterSemer[i] == "CL0":
        Listtofilter.append(i)
    else:
        print("I did it!")

cleanData = DrugData.iloc[Listtofilter, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31]]
print(cleanData)
print(len(DrugData))
print(len(cleanData))
print(cleanData.shape[1])
cleanData = cleanData.replace('CL0', 0)
cleanData = cleanData.replace('CL1', 1)
cleanData = cleanData.replace('CL2', 2)
cleanData = cleanData.replace('CL3', 3)
cleanData = cleanData.replace('CL4', 4)
cleanData = cleanData.replace('CL5', 5)
cleanData = cleanData.replace('CL6', 6)
print(cleanData)
# End cleanData

columns = drug_consumption_quantified.data.ids
print("columns length:", len(columns))
print("cleanData shape:", cleanData.shape)

# Assign columns by slicing to match shape
if len(columns) == cleanData.shape[1]:
    cleanData.columns = columns
else:
    print("Column count mismatch. Using generic names.")
    cleanData.columns = [f"col_{i}" for i in range(cleanData.shape[1])]

print(cleanData)
print(cleanData.columns)

# Set the target column index (e.g., last column)
target_col_idx = 30  # Change this to the correct index for your target

# Use the generic column name for the target
target_col = f"col_{target_col_idx}"

X_clean = cleanData.drop(columns=[target_col])
y_clean = cleanData[target_col]

# Ensure all features are numeric
X_clean = X_clean.apply(pd.to_numeric, errors='coerce')
y_clean = pd.to_numeric(y_clean, errors='coerce')

# Drop rows with missing values (if any)
X_clean = X_clean.dropna()
y_clean = y_clean.loc[X_clean.index]

# Standardize features
scaler = StandardScaler()
X_clean_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index)

# Find the best k using cross-validation
best_k = 1
best_score = 0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_clean_scaled, y_clean, cv=5)
    mean_score = scores.mean()
    if mean_score > best_score:
        best_score = mean_score
        best_k = k
print(f"Best k: {best_k} with CV accuracy: {best_score:.4f}")

# Train/test split and KNN with best k
X_train, X_test, y_train, y_test = train_test_split(X_clean_scaled, y_clean, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Test Accuracy (k={best_k}): {accuracy * 100:.2f}%")

# Cross-validation
scores = cross_val_score(knn, X_clean, y_clean, cv=5)
print("KNN Cross-Validation Scores:", scores)
print(f"Mean CV Accuracy: {scores.mean():.4f}")


#C:\Users\ICSSA-student\AppData\Local\Microsoft\WindowsApps\python3.13.exe C:/Users/ICSSA-student/Downloads/PythonFiles/CapstoneTesting2.py