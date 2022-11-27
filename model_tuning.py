import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 

import preprocessing
import training

# Scaling

def standardize_using_scalar(X_train,X_test):
    """
    Standardise data using standard scalar
    """
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)
    return X_train, X_test

def standardize_using_robust(X_train,X_test):
    """
    Standardise data using robust scalar
    """ 
    robust_scaler = RobustScaler()
    X_train = robust_scaler.fit_transform(X_train)
    X_test = robust_scaler.transform(X_test)
    return X_train, X_test
    

def standardize_using_minmax(X_train,X_test):
    """
    Standardise data using minmax scalar
    """
    minmax_scaler = MinMaxScaler()
    X_train = minmax_scaler.fit_transform(X_train)
    X_test= minmax_scaler.transform(X_test)
    return X_train, X_test
    

def standardize_using_normalizer(X_train,X_test):
    """
    Standardise data using normalizer scalar
    """
    normalizer = Normalizer()
    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.transform(X_test)
    return X_train, X_test

# Under and Oversampling

def check_imbalance(df):
    """
    Checks data imbalance by presenting it on a bar chart
    """
    sal_imbalance = df['salary'].value_counts().plot(kind='bar')
    sal_imbalance.set_title('Frequency of salary data \n salary: 0:<=50k , 1:>=50k')
    sal_imbalance.set_xlabel('Frequency')
    sal_imbalance.set_ylabel('Salary')

def oversampling(X,y):
    """
    Use oversamping to fix the imbalance in data
    """
    X, y = make_classification(n_samples=32561, n_features=15, n_informative=2,
                           weights= [10] )
    ros = RandomOverSampler()
    ros.fit(X, y)
    X, y = ros.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y))  
    return X,y


def undersampling(X,y):
    """
    Use undersampling to fix the imbalance in data
    """
    X, y = make_classification(n_samples=32561, n_features=15, n_informative=2,
                        weights=  [10] )
    rus = RandomOverSampler()
    rus.fit(X, y)
    X, y = rus.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y))  
    return X,y

def fix_imbalanced_data_with_smote(X,y):
    """
    Use Synthetic Minority Over-sampling Technique (SMOTE) to fix the 
    imbalance in data
    """
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y))  
    return X,y

def GridSearch(model,X_train,y_train,X_test,y_test,params):
    """
    Performs hypertuning of params to find most optimal parameters to get the 
    best accuracy for the chosen model
    """
    grid=GridSearchCV(model,params,verbose=3)
    grid.fit(X_train,y_train)
    new_model = grid.best_estimator_
    new_model.fit(X_train,y_train)
    y_pred = new_model.predict(X_test)

    print(grid.best_score_)
    print(training.classification_report(y_test,y_pred))
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    ax = plt.subplot()

    cm_plt =  sns.heatmap(cm/np.sum(cm), fmt='.2%', annot=True, cmap='Blues',ax=ax)
    cm_plt.set_title('Confusion Matrix using ' + str(model))
    cm_plt.xaxis.set_ticklabels(['Negative','Posotive'])
    cm_plt.yaxis.set_ticklabels(['Negative','Posotive'])
    plt.show()

def find_best_num_of_neighbours(X_train, y_train, X_test, y_test):
    """
    Return optimal number of neighbours for knn with the lowest error rate
    """
    error_rate = []
    for i in range(1,30):
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        y_pred = knn.predict(X_test)
        error_rate.append(np.mean(y_pred != y_test))
    
    #Plot graphs showing error rate when there is x number of neighbours

    # plt.figure(figsize=(20,10))
    # plt.plot(range(1,30),error_rate,linestyle='dashed',marker='o',markerfacecolor='red')
    # plt.title('Error Rate vs. K Value')
    # plt.xlabel('K')
    # plt.ylabel('Error Rate')

    req_k_value = error_rate.index(min(error_rate))+1
    print("Minimum error:-",min(error_rate),"at K =",req_k_value)
    return req_k_value
