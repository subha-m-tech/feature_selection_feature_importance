# Imports all the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split 
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import pickle
import matplotlib.pyplot as plt

def permutation_feature_importance(indep_X,dep_Y,n, feature_names):
    """This method helps to find best features based on n value using feature_importance technique
    with the help of permulation importance"""
    
    from sklearn.linear_model import LinearRegression
    lin = LinearRegression()
    
    from sklearn.svm import SVR
    SVRl = SVR(kernel = 'linear')
    
    from sklearn.svm import SVR
    #SVRnl = SVR(kernel = 'rbf')
    
    from sklearn.tree import DecisionTreeRegressor
    dec = DecisionTreeRegressor(random_state = 0)
    
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators = 10, random_state = 0)
    
    fimodellist=[lin,SVRl,dec,rf] 
    feature_list = []
    for i in fimodellist:
        best_feature_name_list = []
        print(i)
        results_fit = i.fit(indep_X, dep_Y)
        results = permutation_importance(results_fit, indep_X, dep_Y, n_repeats=10, scoring='r2')
        importance = results.importances_mean
        # print index of features based on value from low to hign mean
        sorted_idx = importance.argsort()[::-1][:n]
        print('mean::: ', results.importances_mean)
        print('mean arg sort::: ', results.importances_mean.argsort())
        print(f'Selected top {n} features index {sorted_idx} for {i}')
        for j in sorted_idx:
            best_feature_name_list.append(feature_names[j])
            print(f"{feature_names[j]:<8}: {results.importances_mean[j]:.3f} +/- {results.importances_std[j]:.3f}")
        print(f'Selected top {n} features {best_feature_name_list} for {i}')
        feature_list.extend(best_feature_name_list)
        sorted_importance=np.sort(importance)
        print("sorted_importance :: ", sorted_importance)
        print("Total no of features ::", len(importance))
        plt.bar([x for x in range(len(importance))] ,importance)
        plt.show()
    print("feature_list:: ",feature_list)
    ensemble_feature_list = list(set(feature_list))
    
    return ensemble_feature_list

def split_scalar(indep_X,dep_Y):
    """This method takes independent and dependent varaibles and split the dataset into training
    and test data"""
    
    X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)    
    return X_train, X_test, y_train, y_test
    
def r2_prediction(regressor,X_test,y_test):
    """This method gives r2 score values based on the  test data and model"""
    
    y_pred = regressor.predict(X_test)
    from sklearn.metrics import r2_score
    r2=r2_score(y_test,y_pred)
    return r2
 
def Linear(X_train,y_train,X_test, y_test):       
   """This method takes training data and input test data, create Linear models
    and finally calculate R2 score and returns r2 score"""
   from sklearn.linear_model import LinearRegression
   regressor = LinearRegression()
   regressor.fit(X_train, y_train)
   r2=r2_prediction(regressor,X_test,y_test)
   return  r2   
    
def svm_linear(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create svm_linear models
    and finally calculate R2 score and returns r2 score"""
    
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'linear')
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2  
    
def svm_NL(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create svm_NL models
    and finally calculate R2 score and returns r2 score"""
    
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2  
 

def Decision(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create svm_NL models
    and finally calculate R2 score and returns r2 score"""
    
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2  
 

def random_forest(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create random forest models
    and finally calculate R2 score and returns r2 score"""
    
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2 
    
    
def feature_importance_regression(acclin,accsvml,accsvmnl,accdes,accrf): 
    """This method returns dataframe with accuracy of different alogorithms"""
    
    dataframe=pd.DataFrame(index=['R2 score'],columns=['Linear','SVMl','SVM_NL', 'Decision','Random'])

    for number,idex in enumerate(dataframe.index):
        
        dataframe['Linear'][idex]=acclin[number]       
        dataframe['SVMl'][idex]=accsvml[number]
        dataframe['SVM_NL'][idex]=accsvmnl[number]
        dataframe['Decision'][idex]=accdes[number]
        dataframe['Random'][idex]=accrf[number]
    return dataframe
