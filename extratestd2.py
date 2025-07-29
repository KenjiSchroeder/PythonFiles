from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo

from sklearn.datasets import load_wine
X = load_wine().data
y = load_wine().target
wine = load_wine()
feature_names = wine.feature_names
target_names = wine.target_names
print("Dataset shape:", X.shape)
print("Feature names:", feature_names)
print("Target classes:", target_names)
'''
plt.figure(figsize=(6, 4))
for i in range(len(target_names)):
   plt.scatter(X[y == i, 0], X[y == i, 1], label=target_names[i])
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("Wine dataset: First two features")
plt.legend()
plt.tight_layout()
#plt.show()

plt.figure(figsize=(6, 4))
for i in range(len(target_names)):
   plt.scatter(X[y == i, 1], X[y == i, 2], label=target_names[i])
plt.xlabel(feature_names[1])
plt.ylabel(feature_names[2])
plt.title("Wine dataset: Second and Third features")
plt.legend()
plt.tight_layout()
#plt.show()

#plt.savefig('my_plot.png')
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

for i in [14,18]:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Accuracy:" + str(i), accuracy_score(y_test, y_pred))

    from sklearn.model_selection import cross_val_score
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, X, y, cv=5)
    print(scores)

from sklearn.model_selection import cross_val_score
df = pd.DataFrame(wine.data, columns=wine.feature_names)
X = df[["proline", "flavanoids"]]
for i in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, X, y, cv=5)
    print(np.mean(scores))
    print(scores)

