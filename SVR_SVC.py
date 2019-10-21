#(1101 homework)SVC參數調整
import pandas as pd
#import numpy as np
#from sklearn import svm
#from sklearn.svm import SVR #Support Vector Regression 回歸
from sklearn.svm import SVC #Support Vector Classification 分析
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import metrics
#載入數據集
avocado=pd.read_csv("avocado_2.csv")

x = pd.DataFrame(avocado,columns=["Total Volume","Total Bags","AveragePrice","Small Bags","Large Bags","XLarge Bags"])

#資料預處理
label_Encoder=preprocessing.LabelEncoder()
y=label_Encoder.fit_transform(avocado["type"])

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
"""
tuned_parameters = [{'kernel': ['rbf'],'gamma':  [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]
clf = GridSearchCV(SVC(C=1), tuned_parameters)
grid_result=clf.fit(train_x, train_y)
"""
"""   
best_parameters = clf.best_estimator_.get_params()
b=clf.best_score_
print("\nbest parameters:\n",best_parameters,'\n',b)   
"""
"""
for params, mean_score, scores in grid_result.grid_scores_:
    clf= SVC(kernel=params['kernel'], C=params['C'], gamma=params['gamma'], probability=True)
    clf.fit(train_x, train_y)
    predict = clf.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params),"accuracy: %.2f%%" % (100 * accuracy))
"""

def evaluate_svm(train_data, train_labels, test_data, test_labels):
    parameters = [{'kernel': ['rbf'],'gamma':  [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]
 #   model = svm.SVC(SVC(C=1))
    clf = GridSearchCV(SVC(C=1), parameters)
    clf.fit(train_data, train_labels)
   # lin_svm_test = clf.score(test_data, test_labels)
    return clf

grid_result=evaluate_svm(train_x,train_y,test_x,test_y)

for params, mean_score, scores in grid_result.grid_scores_:
    clf= SVC(kernel=params['kernel'], C=params['C'],gamma=params['gamma'],probability=True)
    clf.fit(train_x, train_y)
    predict = clf.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    print((mean_score, scores.std() / 2, params),"accuracy: %.2f%%" % (100 * accuracy))
