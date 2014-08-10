 
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm #lin reg
import pylab as py
import matplotlib as mp
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.grid_search import GridSearchCV
 
#%pylab qt 
#create graphs
 
filepath = '/Kaggle/Bike Sharing Demand/'
 
def read_file(path):
    data = pd.read_csv(path, parse_dates = ['datetime'], index_col = 'datetime')
    return data
    
def describe_data(bk, test):
    print bk.head()
    print test.head()
 
def get_season(date):
    '''from stack overflow'''
    # convert date to month and day as integer (md), e.g. 4/21 = 421, 11/17 = 1117, etc.
    season = []
    for d in date:
        mon = d.month * 100
        da = d.day
        md = mon + da
        
        if ((md > 320) and (md < 621)):
            season.append(0) #spring
        elif ((md > 620) and (md < 923)):
            season.append(1) #summer
        elif ((md > 922) and (md < 1223)):
            season.append(2) #fall
        else:
            season.append(3) #winter
    return season
 
def clean_data_bk(bk):
    
    bk = bk.drop('casual', 1)
    bk = bk.drop('registered',1)
    #Adding a weekday variable
    bk['month'] = bk.index.month
    bk['weekday'] = bk.index.weekday
    bk['day'] = bk.index.day
    bk['hour'] = bk.index.hour
    bk['season'] = get_season(bk.index)
    return bk
 
def clean_data_test(test):
    test['month'] = test.index.month
    test['weekday'] = test.index.weekday
    test['day'] = test.index.day
    test['hour'] = test.index.hour
    test['season'] = get_season(test.index)
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
 
def split_dataset(df):
    #split the data set
    test_idx = np.random.uniform(0, 1, len(df)) <= 0.3
    train = df[test_idx == True]
    test = df[test_idx == False]
    return(train, test)
 
def sle(actual, predicted):
    """
    Taken from benhamner's Metrics library.
    Computes the squared log error.
 
    This function computes the squared log error between two numbers,
    or for element between a pair of lists or numpy arrays.
 
    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
 
    Returns
    -------
    score : double or list of doubles
            The squared log error between actual and predicted
 
    """
    return (np.power(np.log(np.array(actual)+1) - 
            np.log(np.array(predicted)+1), 2))
 
def rmsle(targets, predictions):
    return np.sqrt((sle(targets, predictions)**2).mean())
 
def normalize(bk):
    bk = (bk - bk.mean()) / (bk.max() - bk.min())
    return bk
 
def crossval_GBRT(X_train, y_train, X_test, y_test):
    learning_rate = range(1,11)
    #learning_rate = [x/10.0 for x in learning_rate] #best is 0.1
    #learning_rate = [x/100.0 for x in learning_rate] #best is 0.001
    learning_rate = [x/1000.0 for x in learning_rate] #best is 0.00001
    
    param_grid = {'learning_rate': learning_rate,
                }
            
    est = GradientBoostingRegressor(n_estimators = 3000, max_depth = 1)
    #gs_cv = GridSearchCV(est, param_grid).fit(X_train, y_train)
    #gs_cv.best_params_
 
 
#def main():
bk = read_file(filepath + 'train.csv')
test = read_file(filepath + 'test.csv')
bk = clean_data_bk(bk)
test = clean_data_test(test)
describe_data(bk, test)
 
#Normalize the data
bk_not_norm = bk.copy()
bk_columns = bk.columns.tolist()
bk_columns.remove('count')
 
bk[bk_columns] = normalize(bk[bk_columns])
test = normalize(test)
 
 
bk_train = split_dataset(bk)[0]
bk_test = split_dataset(bk)[1]
 
 
 
X_train = bk_train[bk_columns]
X_test = bk_test[bk_columns]
y_train = bk_train['count']
y_test = bk_test['count']
 
 
#tune hyperparameters via grid serach
best_params = crossval_GBRT(X_train, y_train, X_test, y_test)
print best_params
 
#Train GBRT with the best params
clf = GradientBoostingRegressor(n_estimators = 2500, max_depth = 4, learning_rate = 0.00001, random_state = 0, loss = 'huber')
clf.fit(X_train, y_train)
clf_pred_1 = clf.predict(X_test)
clf_pred_1 = pd.DataFrame(clf_pred_1, index = bk_test.index, columns = ['count'])
clf_pred_1.describe()
#there are negative values
clf_pred_1[clf_pred_1 < 0] = 0
clf_pred_1.describe()
 
#Performance metrics
print 'Grad Tree R^2 score:'
print r2_score(y_test, clf_pred_1)
#0.296
print 'Grad Tree Mean Squared Error:'
print mean_squared_error(y_test, clf_pred_1)
#22938
print 'Grad Tree Root Mean Squared Log Error:'
print rmsle(y_test, clf_pred_1)
#5.8
 
 
#Now make predictions for actual testing set
 
clf_pred_test = clf.predict(test)
 
clf_pred_test = pd.DataFrame(clf_pred_test, index = test.index, columns = ['count'])
clf_pred_test[clf_pred_test < 0] = 0
 
#Print to file
results_tree_file = filepath + 'results_tree2.csv'
clf_pred_test.to_csv(results_tree_file, index_label = ['datetime'])
