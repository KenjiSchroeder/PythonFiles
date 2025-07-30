from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


zoo = fetch_ucirepo(id=111)


X3 = zoo.data.features

y3 = zoo.data.targets
print("aoidhfbsdkfb")
print(y3)
y3 = y3.squeeze() #this is just to make the y values into a one-dimensional list.
print("blsdaojlsakd")
print(y3)
print(X3)
print(type(X3))
print(type(y3))
test = zoo.target_names
print(test)
wine = load_wine()
print(wine.target_names)
drug_consumption_quantified = fetch_ucirepo(id=373) 
print(drug_consumption_quantified.target_names)