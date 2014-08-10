 
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm #lin reg
import pylab as py
import matplotlib as mp
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
 
#%pylab qt 
#create graphs
 
filepath = '/Documents/Data Science/Kaggle/Bike Sharing Demand/'
 
def read_file(path):
    data = pd.read_csv(path, parse_dates = ['datetime'], index_col = 'datetime')
    return data
    
def describe_data(bk, test):
    print bk.head()
    print test.head()
 
def clean_data_bk(bk):
    
    bk = bk.drop('casual', 1)
    bk = bk.drop('registered',1)
    #Adding a weekday variable
    bk['weekday'] = bk.index.weekday
    return bk
 
def clean_data_test(test):
    test['weekday'] = test.index.weekday
    return test
 
 
def decision_tree_regressor_fit(bk_columns, bk):
 
    clf = DecisionTreeRegressor()
    X = bk[bk_columns]
    y = bk['count']
    clf = clf.fit(X, y)
    return clf
 
def decision_tree_prediction(bk_columns, clf, test):
    clf_pred_1 = clf.predict(test[bk_columns])
    print clf_pred_1
    return clf_pred_1
 
def main():
    bk = read_file(filepath + 'train.csv')
    test = read_file(filepath + 'test.csv')
    bk = clean_data_bk(bk)
    test = clean_data_test(test)
    describe_data(bk, test)
    bk_columns = bk.columns.tolist()
    bk_columns.remove('count')
    clf = decision_tree_regressor_fit(bk_columns, bk)
    clf_pred_1 = decision_tree_prediction(bk_columns, clf, test)
 
    clf_pred_1 = pd.DataFrame(clf_pred_1, index = test.index, columns = ['count'])
 
    results_tree_file = filepath + 'results_tree1.csv'
    clf_pred_1.to_csv(results_tree_file, index_label = ['datetime'])
 
    
if __name__ == '__main__':
    main()
