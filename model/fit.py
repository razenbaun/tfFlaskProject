import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle

iris = pd.read_csv('data (2).csv')
iris.drop_duplicates(inplace=True)
X = iris.drop(["beloved_color"], axis=1)
Y = iris["beloved_color"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
model.fit(X_train, y_train)



with open('Iris_pickle_file_tree', 'wb') as pkl:
    pickle.dump(model, pkl)