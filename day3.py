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


weather_data = [
   ['Sunny', 'Warm', 'Low', 'Yes'],      # Go out
   ['Sunny', 'Hot', 'Low', 'Yes'],       # Go out 
   ['Sunny', 'Warm', 'Normal', 'Yes'],   # Go out
   ['Rainy', 'Cold', 'High', 'No'],      # Don't go out
   ['Cloudy', 'Cold', 'High', 'No'],     # Don't go out
   ['Rainy', 'Warm', 'High', 'No']       # Don't go out
]


columns = ['Weather', 'Temperature', 'Humidity', 'Go Out']


df = pd.DataFrame(weather_data, columns=columns)

# this is to encode the categorical data into numerical values,
# so that we can input them into the decision tree classifier.
encoders = {}
for col in df.columns:
   le = LabelEncoder()
   df[col] = le.fit_transform(df[col])
   encoders[col] = le

X = df.drop(columns=['Go Out'])
y = df['Go Out']

clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X, y)

plt.figure(figsize=(10,6))
plot_tree(clf, feature_names=X.columns, class_names=encoders['Go Out'].classes_, filled=True)
plt.show()


test_cases = [
   ['Sunny', 'Warm', 'Low'],
   ['Rainy', 'Cold', 'High'],
   ['Cloudy', 'Warm', 'Normal']
]


for case in test_cases:
   encoded = [encoders[c].transform([val])[0] for c, val in zip(X.columns, case)]
   pred = clf.predict([encoded])
   label = encoders['Go Out'].inverse_transform(pred)[0]
   print(f"{case} => Go Out? {label}")
#Yes, this one cares about all 3 variations of Weather, while the finds only cared if it's sunny or not


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
clf = DecisionTreeClassifier(criterion='entropy', random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
plt.figure(figsize=(12, 8))
plot_tree(
   clf,
   feature_names=iris.feature_names,
   class_names=iris.target_names,
   filled=True
)
plt.show()
#slightly less accurate, but probably more for other data

wine = load_wine()
X2 = wine.data
y2 = wine.target

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=1)
clf = DecisionTreeClassifier(criterion='entropy', random_state=1)
clf.fit(X2_train, y2_train)
y2_pred = clf.predict(X2_test)
print("Accuracy:", accuracy_score(y2_test, y2_pred))
plt.figure(figsize=(12, 8))
plot_tree(
   clf,
   feature_names=wine.feature_names,
   class_names=wine.target_names,
   filled=True
)
plt.show()

importances2 = clf.feature_importances_
features2 = wine.feature_names

plt.figure(figsize=(8, 5))
plt.barh(range(len(importances2)), importances2, align='center')
plt.yticks(range(len(importances2)), features2)
plt.xlabel("Feature Importance")
plt.title("Random Forest - Wine Feature Importances")
plt.tight_layout()
plt.show()

#More accurate than the basic knn

zoo = fetch_ucirepo(id=111)


X3 = zoo.data.features
y3 = zoo.data.targets
print("aoidhfbsdkfb")
print(y3)
y3 = y3.squeeze() #this is just to make the y values into a one-dimensional list.
print("blsdaojlsakd")
print(y3)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=1)
clf = DecisionTreeClassifier(criterion='entropy', random_state=1,)
print("ZOOOOO BREAK")
print(X3_train)
print(y3_train)
clf.fit(X3_train, y3_train)
y3_pred = clf.predict(X3_test)
print("Accuracy:", accuracy_score(y3_test, y3_pred))
plt.figure(figsize=(12, 8))
plot_tree(
clf,
feature_names=zoo.feature_names,
class_names=zoo.target_names,
filled=True
)
plt.show()

for i in range(1,6):
    clf = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=i)
    clf.fit(X3_train, y3_train)
    y3_pred = clf.predict(X3_test)
    print("Accuracy:", accuracy_score(y3_test, y3_pred))
    plt.figure(figsize=(12, 8))
    plot_tree(
    clf,
    feature_names=zoo.feature_names,
    class_names=zoo.target_names,
    filled=True
    )
    plt.show()

mushroom = fetch_ucirepo(id=73)


X4 = mushroom.data.features
y4 = mushroom.data.targets


X4_encoded = X4.copy()
encoders = {}


for col in X4.columns:
   le = LabelEncoder()
   X4_encoded[col] = le.fit_transform(X4[col])
   encoders[col] = le




X4_train, X4_test, y4_train, y4_test = train_test_split(X4_encoded, y4, test_size=0.2, random_state=1)
clf = DecisionTreeClassifier(criterion='entropy', random_state=1,)
clf.fit(X4_train, y4_train)
y4_pred = clf.predict(X4_test)
print("Accuracy:", accuracy_score(y4_test, y4_pred))
plt.figure(figsize=(12, 8))
plot_tree(
clf,
feature_names=zoo.feature_names,
class_names=zoo.target_names,
filled=True
)
plt.show()
#Yes, 100%


iris = load_iris()
X5 = iris.data
y5 = iris.target


X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.2, random_state=1)

clf = RandomForestClassifier(n_estimators=100, random_state=1)
clf.fit(X5_train, y5_train)




y5_pred = clf.predict(X5_test)
print("RF Accuracy:", accuracy_score(y5_test, y5_pred))

importances = clf.feature_importances_
features = iris.feature_names

plt.figure(figsize=(8, 5))
plt.barh(range(len(importances)), importances, align='center')
plt.yticks(range(len(importances)), features)
plt.xlabel("Feature Importance")
plt.title("Random Forest - Iris Feature Importances")
plt.tight_layout()
plt.show()


