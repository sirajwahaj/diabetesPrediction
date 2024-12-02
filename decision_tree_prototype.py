import numpy as np
import pandas as pd
from collections import Counter
import types
import collections
import importlib
from sklearn.model_selection import train_test_split
from sklearn import datasets

#helper class to keep data
class Node():
    def __init__(self, column_index=None, question_split=None, left=None, right=None):

        self.column_index = column_index
        self.question_split = question_split
        self.left = left
        self.right = right
   

class Leaf():
    def __init__(self,value):
        self.value = value
    
#decision tree class
class DecisionTree():
    def __init__(self,allowed_depth=2):

        #first node
        self.root = None

        #tree limit
        self.allowed_depth = allowed_depth
    
    def build(self, X,Y, depth=0):

        #number of rows and columns   
        n_rows, n_Y = np.shape(X)
        #check tree limit
        if depth>=self.allowed_depth:
            val = self.value(Y)
            return(Leaf(val))
        
        if depth<self.allowed_depth:
            #get split
            column_index,right,left,qst = self.accurate_split(X,Y, n_rows, n_Y)
            #if we have a split
            if column_index is not None:
                left_side = self.build(left[:, :-1],left[:, -1], depth+1)
                right_side = self.build(right[:, :-1],right[:, -1], depth+1)
                return Node(column_index, qst,left_side, right_side)
            else:
                val = self.value(Y)
                return(Leaf(val))
 
            
    
    def accurate_split(self, X,Y, n_x, n_y):
    
        column_index = None
        left = None
        right = None
        question_split = None
        total_ig = -99999999
        
        for column in range(n_y):
            x_values = X[:, column]
            
            for question in x_values:
                #calculate information gain
                dataset = np.concatenate((X, Y.reshape(1, -1).T), axis=1)
                data_left = np.array([row for row in dataset if row[column] <= question])
                data_right = np.array([row for row in dataset if row[column] > question])
                
                if(len(data_left)>0 and len(data_right>0)):
                    info_gain= self.calc_ig(dataset,data_left,data_right,column,question)
                    #replace information gain
                    if info_gain>total_ig and info_gain >0:
                        column_index = column
                        right = data_right
                        left = data_left
                        question_split = question
                              
        return column_index,right,left,question_split
    

    def calc_entropy(self,y,base = None):

        q = np.bincount(np.array(y, dtype=np.int64))
        ps = q / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])   

    def calc_ig(self,dataset,dataset_left,dataset_right,col,qst):

        data_left = dataset_left[:, -1]
        data_right = dataset_right[:, -1]
        data = dataset[:, -1]
        #calculate left probability
        left_weight = len(data_left) / len(data)
        #calculate right probability
        right_weight = len(data_right) / len(data)
        #get the parent entropy
        parent_entropy = self.calc_entropy(data)
        #calculate entropy for the left and right side
        left_child_entropy = left_weight*self.calc_entropy(data_left)
        right_child_entropy = right_weight*self.calc_entropy(data_right)
        #calculate gain
        gain = parent_entropy- (left_child_entropy + right_child_entropy)
        return gain

    def find_leaves(self, X):

        leaves = [self.find_leaf(x, self.root) for x in X]
        return np.array(leaves)
    
    def find_leaf(self, x, node):
        #return if leaf
        if isinstance(node, Leaf):
            return node.value
        feature_index = x[node.column_index]
        
        if feature_index<=node.question_split:
            return self.find_leaf(x, node.left)
        else:
            return self.find_leaf(x, node.right)

 
    #leaf node gets the most common dependent variable
    def value(self, Y):
        counter = Counter(Y)
        value = counter.most_common(1)[0][0]
        return value
    
    #starter method
    def fit(self, X,Y):
        self.root = self.build(X,Y)
    #calculate how many predictions were accurate
    def acc(self,label,pred_label):
        return np.sum(np.equal(label, pred_label)) / len(label)
    #print number of nodes


diabetes = pd.read_csv('diabetes.csv')
diabetes = diabetes.drop(columns=['BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

X = diabetes.iloc[:,:-1].values
y = diabetes.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

decisiontree =  DecisionTree()
decisiontree.fit(X_train,y_train)
y_pred = decisiontree.find_leaves(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = decisiontree.acc(y_pred, y_test)
print(acc)



