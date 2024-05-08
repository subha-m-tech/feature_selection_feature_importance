# import all the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split 
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def permutation_feature_importance(indep_X,dep_Y,n, feature_names):
    """This method helps to find best features based on n value using feature_importance technique
    with the help of permulation importance"""
    log_model = LogisticRegression(solver='lbfgs')
    RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    DT= DecisionTreeClassifier(criterion = 'gini', max_features='sqrt',splitter='best',random_state = 0)
    svc_model = SVC(kernel = 'linear', random_state = 0)
    fimodellist=[log_model,svc_model,RF,DT]
    feature_list = []
    for i in fimodellist:
        best_feature_name_list = []
        print(i)
        results_fit = i.fit(indep_X, dep_Y)
        results = permutation_importance(results_fit, indep_X, dep_Y, n_repeats=10, scoring='accuracy')
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
    
    #Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test
    
def cm_prediction(classifier,X_test, y_test):
    """This method gives conflusion matrix values based on the  test data and model"""

    y_pred = classifier.predict(X_test)
        
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
        
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import classification_report
    Accuracy=accuracy_score(y_test, y_pred )
        
    report=classification_report(y_test, y_pred)
    return  classifier,Accuracy,report,X_test,y_test,cm

def logistic(X_train,y_train,X_test, y_test):       
    """This method takes training data and input test data, create logistic models
    and finally calculate confusion matrix and returns model object with metrics"""
    
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test,y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm      
    
def svm_linear(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create svm_linear models
    and finally calculate confusion matrix and returns model object with metrics"""
    
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test,y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm
    
def svm_NL(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create svm_NL models
    and finally calculate confusion matrix and returns model object with metrics"""
    
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test,y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm
   
def Navie(X_train,y_train,X_test, y_test):       
    """This method takes training data and input test data, create Navie models
    and finally calculate confusion matrix and returns model object with metrics"""
    
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test,y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm         
    
    
def knn(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create knn models
    and finally calculate confusion matrix and returns model object with metrics"""

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test,y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm
    
def Decision(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create Decision models
    and finally calculate confusion matrix and returns model object with metrics"""

    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test,y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm      


def random_forest(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create random forest models
    and finally calculate confusion matrix and returns model object with metrics"""
    
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test,y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm
    

def fi_classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf): 
    """This method returns dataframe with accuracy of different algorithms"""
    
    fi_dataframe=pd.DataFrame(index=['Accuracy'],columns=['Logistic','SVMl','SVMnl','KNN','Navie','Decision','Random'])

    for number,idex in enumerate(fi_dataframe.index):
        
        fi_dataframe['Logistic'][idex]=acclog[number]       
        fi_dataframe['SVMl'][idex]=accsvml[number]
        fi_dataframe['SVMnl'][idex]=accsvmnl[number]
        fi_dataframe['KNN'][idex]=accknn[number]
        fi_dataframe['Navie'][idex]=accnav[number]
        fi_dataframe['Decision'][idex]=accdes[number]
        fi_dataframe['Random'][idex]=accrf[number]
    return fi_dataframe