import numpy as np
import pandas as pd
import statsmodels.formula.api as sm #lin reg
import pylab as plt
import matplotlib as mp
 
 
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import lasso_path, enet_path
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
 
 
#Lasso
 
 
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
 
bk = read_file(filepath + 'train.csv')
test = read_file(filepath + 'test.csv')
bk = clean_data_bk(bk)
test = clean_data_test(test)
describe_data(bk, test)
 
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
 
def make_formula(df_col, depvar):
    '''list1, list2 -> str
    takes a list of all column names and a list of 
    the dependent variable name, returns
    a formula in Y ~ X1 + X2 + ... + Xn form
    >>> make_formula(tt_train.columns, ["Survived"])
    'Survived~Age+Cabin+Embarked+Fare+Name+Parch+
    PassengerId+Pclass+Sex+SibSp+Ticket'
 
    '''
    all_columns = "+".join(df_col - depvar)
    return str(depvar[0]) + "~" + all_columns
 
 
#Splitting data into 2 groups 
 
is_test = np.random.uniform(0,1,len(bk)) > 0.75
train = bk[is_test == False]
test = bk[is_test==True]
len(train), len(test)
 
 
bk_columns = bk.columns.tolist()
 
y_train = np.array(train['count'])
X_train = train.drop('count', 1)
y_test = np.array(test['count'])
X_test = test.drop('count',1)
 
#Compute paths
#eps = 5e-3 #the smaller, the longer the path
 
 
#print "Computing regularization path using the lasso..."
#models = lasso_path(X, y, eps=eps)
#alphas_lasso = np.array([model.alpha for model in models])
#coefs_lasso = np.array([model.coef_ for model in models])
 
 
#Linear Regression
formula1 = make_formula(bk.columns, ['count'])
lm1 = sm.ols(formula = formula1, data = test).fit()
 
lm1_pred = lm1.predict(X_test)
lm1_pred[lm1_pred < 0] = 0
 
print 'Lin Reg R^2 score:'
print r2_score(y_test, lm1_pred)
#0.265
print 'Lin Reg Mean Squared Error:'
print mean_squared_error(y_test, lm1_pred)
#24068
print 'Lin Reg Root Mean Squared Log Error:'
print rmsle(y_test, lm1_pred)
#4.39
 
    
            
#Lasso
alpha = 0.1
lasso1 = Lasso(alpha = alpha)
y_pred_lasso1 = lasso1.fit(X_train, y_train).predict(X_test)
print lasso1
print 'Lasso R^2 score:'
print r2_score(y_test, y_pred_lasso1)
#0.2599
print 'Lasso Mean Squared Error:'
print mean_squared_error(y_test, y_pred_lasso1)
#23939
print 'Lasso Root Mean Squared Log Error:'
print rmsle(y_test, y_pred_lasso1)
 
 
y_pred_lasso1 = pd.DataFrame(y_pred_lasso1, index = test.index, columns = ['count'])
y_pred_lasso1.describe()
#Getting rid of the negative values
y_pred_lasso1[y_pred_lasso1< 0] = 0
y_pred_lasso1.describe()
print lasso1
print 'Lasso R^2 score:'
print r2_score(y_test, y_pred_lasso1)
#0.2604
print 'Lasso Mean Squared Error:'
print mean_squared_error(y_test, y_pred_lasso1)
#24232
print 'Lasso Root Mean Squared Log Error:'
print rmsle(y_test, y_pred_lasso1)
#6.089
 
#Cross-validate the LASSO-penalized linear regression
lasso2 = LassoCV(cv = 15) #cv specifies the number of cross-validation folds to 
lasso2_fit = lasso2.fit(X_train, y_train)
lasso2_path = lasso2.score(X_train, y_train)
#run on each penalty-parameter value
 
 
plt.plot(-np.log(lasso2_fit.alphas_),
np.sqrt(lasso2_fit.mse_path_).mean(axis = 1))
plt.ylabel('RMSE (avg. across folds)')
plt.xlabel(r'\$-\\log(\\lambda)\$')
# Indicate the lasso parameter that minimizes the average MSE across
#folds
plt.axvline(-np.log(lasso2_fit.alpha_), color = 'red')
 
 
alpha = lasso2_fit.alpha_
 
lasso3 = Lasso(alpha = alpha)
y_pred_lasso3 = lasso3.fit(X_train, y_train).predict(X_test)
print lasso3
print 'Lasso R^2 score:'
print r2_score(y_test, y_pred_lasso3)
#0.2599
print 'Lasso Mean Squared Error:'
print mean_squared_error(y_test, y_pred_lasso3)
#23939
print 'Lasso Root Mean Squared Log Error:'
print rmsle(y_test, y_pred_lasso3)
 
 
y_pred_lasso3 = pd.DataFrame(y_pred_lasso3, index = test.index, columns = ['count'])
y_pred_lasso3.describe()
#Getting rid of the negative values
y_pred_lasso3[y_pred_lasso3< 0] = 0
y_pred_lasso3.describe()
print lasso3
print 'Lasso 3 R^2 score:'
print r2_score(y_test, y_pred_lasso3)
#0.236
print 'Lasso 3 Mean Squared Error:'
print mean_squared_error(y_test, y_pred_lasso3)
#25175
print 'Lasso 3 Root Mean Squared Log Error:'
print rmsle(y_test, y_pred_lasso3)
#6.36
