from sklearn.datasets import load_wine
import itertools
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


wine = load_wine()
X = load_wine().data
y = load_wine().target
feature_names = wine.feature_names
target_names = wine.target_names
print(feature_names)
grp1 = feature_names
grp2 = list(itertools.combinations(feature_names,2))

for i in range(1, 14):
    ev = "grp" + str(i) + " = list(itertools.combinations(feature_names,"+str(i)+"))"
    exec(ev)
    prnt =  "print('group"+ str(i) +": '+" + " str(grp"+str(i)+"))"
    exec(prnt)

df = pd.DataFrame(wine.data, columns=wine.feature_names)
X = df[list(grp2[63])]
for i in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, X, y, cv=5)
    print(np.mean(scores))
    print(scores)

print(grp2[63])
print(len(grp2))