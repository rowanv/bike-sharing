#Kaggle competition
 
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm #lin reg
import pylab as py
 
%pylab qt 
#create graphs
 
filepath = '/Documents/Data Science/Kaggle/Bike Sharing Demand/'
 
def read_file(path):
    data = pd.read_csv(path, parse_dates = ['datetime'], index_col = 'datetime')
    return data
    
def clean_data(bk, test):
    
    bk = bk.drop('casual', 1)
    bk = bk.drop('registered',1)
    #Adding a weekday variable
    bk['weekday'] = bk.index.weekday
    test['weekday'] = test.index.weekday
 
    #correlation
    corr_matrix = bk.corr()
    py.pcolor(corr_matrix)
    colorbar()
 
    #temp and atemp nearly 1
    #dropping temp
    bk = bk.drop(bk.columns[4])
    test = test.drop(test.columns[4])
 
def descriptive_plots(bk):
#bike use spikes on saturdays
#Exploratory -- looking at bike use over the week
    weekday_counts = bk.groupby('weekday').aggregate(sum)
    weekday_counts.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts
    weekday_counts[['count']].plot()
    bk.describe()
 
    bk['weather'].plot()
    bk['temp'].plot()
    bk['humidity'].plot()
 
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
 
def lin_reg(bk):
    formula1 = make_formula(bk.columns, ['count'])
 
    lm1 = sm.ols(formula = formula1, data = bk).fit()
 
    return lm1
 
def pred_lin_reg(lm1, test):
 
    lm1_pred = lm1.predict(test)
    lm1_pred[lm1_pred < 0] = 0
    lm1_pred.index = test.index
 
 
    lm1_pred = pd.DataFrame(lm1_pred, index = lm1_pred.index.values, columns = ['count'])
 
    results2_file = filepath + 'results2.csv'
    lm1_pred.to_csv(results2_file, index_label = ['datetime'])
 
    
def splitDatetime(data):
    sub = pd.DataFrame(data.datetime.str.split(' ').tolist(), columns = "date time".split())
    date = pd.DataFrame(sub.date.str.split('-').tolist(), columns="year month day".split())
    time = pd.DataFrame(sub.time.str.split(':').tolist(), columns = "hour minute second".split())
    data['year'] = date['year']
    data['month'] = date['month']
    data['day'] = date['day']
    data['hour'] = time['hour'].astype(int)
    return data
 
def main():
    bk = read_file(filepath + 'train.csv')
    test = read_file(filepath + 'test.csv')
    clean_data(bk, test)
    descriptive_plots(bk)
    lm1 = lin_reg(bk)
    print lm1.summary()
    pred_lin_reg(lm1, test)
    
if __name__ == '__main__':
    main()
 
    
 
#Improved R score by 0.00355
