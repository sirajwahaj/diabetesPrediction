import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import datasets
class KNN:
    def __init__(self, k=2):
        self.k = k

    def value(self,knn_values):
        most_common = Counter(knn_values).most_common()
        return most_common[0][0]
    
    def fit(self, X, Y):
        #X_train
        self.X = X
        #y_train
        self.Y = Y
        
    def euclidean_distance(self, point_a, point_b):
        return np.linalg.norm(point_a-point_b)
    

    def predict(self,X):
        #assign each distance to a label and then pick the most popular
        predictions = []
        for test_point in X:
            distances = []
            train_data = zip(self.X, self.Y)
            for train_point, train_label in train_data:
                distance = self.euclidean_distance(test_point, train_point)
                distances.append((distance, train_label))
            distances.sort()
            dists = distances[:self.k]
            y = [y_ for _, y_ in dists]
            predictions.append(self.value(y))
        return predictions
    


diabetes = pd.read_csv('diabetes.csv')
diabetes = diabetes.drop(columns=['BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

X = diabetes.iloc[:,:-1].values
y = diabetes.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

knearest =  KNN()
knearest.fit(X_train,y_train)

y_pred = knearest.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)
