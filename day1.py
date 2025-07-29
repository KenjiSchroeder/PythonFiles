print("Hello, World")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
X = iris.data
feature_names = iris.feature_names
print("Feature names:", feature_names)
y = iris.target
target_names = iris.target_names
print("Target classes:", target_names)
print("first 5")
print("First 5 samples:\n", X[:5])
print("First 5 labels:\n", y[:5])
print("last 5 ")
print("Dataset shape:", X.shape)
print("Last 5 samples:\n", X[145:])
print("Last 5 labels:\n", y[145:])

plt.figure(figsize=(6, 4))
for i in range(len(target_names)):
   plt.scatter(X[y == i, 0], X[y == i, 1], label=target_names[i])
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("Iris dataset: First two features")
plt.legend()
plt.tight_layout()
#plt.show()

plt.figure(figsize=(6, 4))
for i in range(len(target_names)):
   plt.scatter(X[y == i, 1], X[y == i, 2], label=target_names[i])
plt.xlabel(feature_names[1])
plt.ylabel(feature_names[2])
plt.title("Iris dataset: Second and Third features")
plt.legend()
plt.tight_layout()
#plt.show()

plt.savefig('my_plot.png')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, X, y, cv=5)
print(scores)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
X = df[["sepal length (cm)", "sepal width (cm)"]]

from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, X, y, cv=5)
print(scores)

