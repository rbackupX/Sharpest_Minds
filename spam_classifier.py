# Script attempts to perform spam classification using Logistic Regression and Decision Trees
# HW assignment for Data Analytics course
# Author: Ryan Kingery
# Date: Oct 16, 2017

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

PATH = '/Users/ryankingery/Desktop/'

# load data from internet and set up X and y
df = pd.read_csv("http://www.apps.stat.vt.edu/leman/VTCourses/spam.data.txt",sep=' ')

X = df.values[:,:-1]
y = df.values[:,-1].reshape((len(df.values[:,-1]),1))


# performing 10-fold cross validation to evaluate decision tree and logistic regression models
avg_scores = []
for i in range(50):
    kf = KFold(n_splits=10,shuffle=True)
    kf.get_n_splits(X)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = DecisionTreeClassifier(min_samples_leaf=1,max_depth=10)
        #model = LogisticRegressionCV(penalty='l1',max_iter=500,solver='liblinear')
        model.fit(X_train,y_train)
        scores += [model.score(X_test,y_test)]
    print "Iter "+str(i)+": "+str(np.mean(scores))
    avg_scores += [np.mean(scores)]
    
print np.min(avg_scores), np.mean(avg_scores), np.max(avg_scores)
print 1-np.max(avg_scores), 1-np.mean(avg_scores), 1-np.min(avg_scores)

# generate and save decision tree image
export_graphviz(model,out_file=PATH+'tree.dot')
    

# testing accuracy of various models for the sake of comparison
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)

# model = LogisticRegressionCV(penalty='l1',max_iter=500,solver='liblinear')
# model.fit(X_train,y_train)
# print 'Logistic Regression Score: ', model.score(X_test,y_test)

# model = MLPClassifier(hidden_layer_sizes=(),max_iter=1000,alpha=0.1)
# model.fit(X_train,y_train)
# print 'Neural Network Score: ', model.score(X_test,y_test)

# model = DecisionTreeClassifier(min_samples_leaf=10)
# model.fit(X_train,y_train)
# print 'Decision Tree Score: ', model.score(X_test,y_test)

