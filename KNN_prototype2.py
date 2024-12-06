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
        self.X = X
        self.Y = Y
        
    def euclidean_distance(self, train, test):
        #calculate the distance between every feature in train vs test row
        dist = train-test
        #return the norm representation for the array (the euclidean distance)
        return np.linalg.norm(dist)
    
    
    def predict(self, X_test):
        return [self.predict_y(X_row) for X_row in X_test]


    def predict_y(self,X_test):

        #get the euclidean distances, sort it and have a majority vote of k number of labels to predict the y value of the test point
        dist_arr = []
         
        distances = [self.euclidean_distance(x_train, X_test) for x_train in self.X]
        dist_arr = np.array(distances) 
 
       
        cluster = np.argsort(dist_arr)[:self.k] 

        y = self.Y[cluster]
 
        return self.value(y)


diabetes = pd.read_csv('diabetes.csv')
#diabetes = diabetes.drop(columns=['BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
diabetes = diabetes.iloc[:10]
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
