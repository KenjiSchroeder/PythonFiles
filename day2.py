from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo

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


concepts = np.array(df)[:,:-1]
target = np.array(df)[:,-1]

findshypoweather = findS(weather_data, np.array(weather_data)[:, -1])
print("Find-S Hypothesis Weather:", findshypoweather)
#yes


restaurant_data = [
   ['Yes', 'No', 'No', 'Yes', 'Some', '$$$', 'No', 'French', 'Yes'],
   ['Yes', 'No', 'No', 'Yes', 'Full', '$', 'No', 'Thai', 'No'],
   ['No', 'Yes', 'No', 'No', 'Some', '$', 'No', 'Burger', 'Yes'],
   ['Yes', 'No', 'Yes', 'Yes', 'Full', '$', 'Yes', 'Thai', 'Yes'],
   ['Yes', 'No', 'Yes', 'No', 'Full', '$$$', 'No', 'French', 'Yes'],
   ['No', 'Yes', 'No', 'Yes', 'Some', '$$', 'Yes', 'Italian', 'No'],
   ['No', 'No', 'No', 'Yes', 'None', '$', 'Yes', 'Burger', 'Yes'],
   ['No', 'No', 'Yes', 'No', 'Some', '$$', 'Yes', 'Thai', 'No'],
   ['No', 'Yes', 'Yes', 'No', 'Full', '$', 'Yes', 'Burger', 'No'],
   ['Yes', 'Yes', 'Yes', 'Yes', 'Full', '$$$', 'No', 'Italian', 'Yes'],
   ['No', 'No', 'No', 'No', 'None', '$', 'No', 'Thai', 'No'],
   ['Yes', 'Yes', 'No', 'Yes', 'Some', '$', 'No', 'Burger', 'Yes'],
   ['Yes', 'No', 'No', 'Yes', 'Some', '$$', 'No', 'Italian', 'Yes'],
   ['No', 'Yes', 'No', 'No', 'Some', '$$$', 'No', 'French', 'No'],
   ['No', 'No', 'Yes', 'Yes', 'Some', '$', 'Yes', 'Italian', 'Yes'],
   ['Yes', 'No', 'Yes', 'No', 'Some', '$$', 'No', 'Thai', 'No'],
   ['Yes', 'Yes', 'No', 'No', 'Full', '$$', 'Yes', 'French', 'No'],
   ['No', 'No', 'No', 'Yes', 'Full', '$$$', 'Yes', 'Burger', 'No'],
   ['Yes', 'No', 'Yes', 'Yes', 'None', '$$', 'No', 'Italian', 'Yes'],
   ['No', 'Yes', 'Yes', 'No', 'None', '$$$', 'Yes', 'French', 'No']
]

columns = ['Alternate', 'Bar', 'FriSat', 'Hungry', 'Patrons', 'Price', 'Rain', 'Type', 'WillWait']

findshyporestaurant = findS(restaurant_data, np.array(restaurant_data)[:, -1])
print("Find-S Hypothesis Weather:", findshyporestaurant)


def candidate_eliminate(concepts, target):
   specific_h = concepts[0].copy() 
   print("initialization of specific_h \n",specific_h) 
   general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]    
   print("initialization of general_h \n", general_h) 


   for i, h in enumerate(concepts):
       if target[i] == 'Yes':
           print("If instance is Positive ")
           for x in range(len(specific_h)):
               if h[x]!= specific_h[x]:                   
                   specific_h[x] ='?'                    
                   general_h[x][x] ='?'
                 
       if target[i] == 'No':           
           print("If instance is Negative ")
           for x in range(len(specific_h)):
               if h[x]!= specific_h[x]:                   
                   general_h[x][x] = specific_h[x]               
               else:                   
                   general_h[x][x] = '?'       


       print(" step {}".format(i+1))
       print(specific_h)        
       print(general_h)
       print("\n")
       print("\n")


   indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?']]   
   for i in indices:  
       general_h.remove(['?', '?', '?'])
   return specific_h, general_h
# from https://github.com/DeepakDVallur/Candidate_Elimination

weathersfinal, weathergfinal = candidate_eliminate(concepts,target)
print("Candidate Elimination Hypothesis:", weathersfinal, weathergfinal)

zoo = fetch_ucirepo(id=111)


X = zoo.data.features
y = zoo.data.targets




target = y.iloc[:, 0].apply(lambda x: 'Yes' if x == 4 else 'No').tolist()


concepts = X.astype(str).values.tolist()

reptilefinds = findS(concepts, target)
print("Find-S Hypothesis Zoo:", reptilefinds)

for i in range(1,7):
    throw = "target" + str(i) + " = y.iloc[:, 0].apply(lambda x: 'Yes' if x == "+str(i)+" else 'No').tolist()"
    exec(throw)
    throw2 = "finds" + str(i) + " = findS(concepts, target" + str(i) + ")"
    exec(throw2)
    print(f"Find-S Hypothesis for target {i}:", eval(f"finds{i}"))



mushroom = fetch_ucirepo(id=73)


X2 = mushroom.data.features
y2 = mushroom.data.targets


target2 = y2.iloc[:, 0].apply(lambda x: 'Yes' if x == 'e' else 'No').tolist()
concepts2 = X2.astype(str).values.tolist()
mushroomfinds = findS(concepts2, target2)
print("Find-S Hypothesis Mushroom:", mushroomfinds)

from sklearn.datasets import load_wine
X = load_wine().data
y = load_wine().target
wine = load_wine()
feature_names = wine.feature_names
target_names = wine.target_names
print("Dataset shape:", X.shape)
print("Feature names:", feature_names)
print("Target classes:", target_names)
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

'''
df = pd.DataFrame(wine.data, columns=wine.feature_names)
X = df[["variable", "variable2"]]

from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, X, y, cv=5)
print(scores)
'''
