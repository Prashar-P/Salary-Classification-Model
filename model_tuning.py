import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
import preprocessing


# Scaling

def standardize_using_scalar(X_train,X_test):
    """
    Trains  data in order to make predictions based on the data provided.
    """
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)
    return X_train, X_test

def standardize_using_robust(X_train,X_test):
    """
    Trains  data in order to make predictions based on the data provided.
    """ 
    robust_scaler = RobustScaler()
    X_train = robust_scaler.fit_transform(X_train)
    X_test = robust_scaler.transform(X_test)
    return X_train, X_test
    

def standardize_using_minmax(X_train,X_test):
    minmax_scaler = MinMaxScaler()
    X_train = minmax_scaler.fit_transform(X_train)
    X_test= minmax_scaler.transform(X_test)
    return X_train, X_test
    

def standardize_using_normalizer(X_train,X_test):
    """
    Trains  data in order to make predictions based on the data provided.
    """
    normalizer = Normalizer()
    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.transform(X_test)
    return X_train, X_test

# Under and Oversampling

def check_imbalance(df):
    """
    Trains  data in order to make predictions based on the data provided.
    """
    sal_imbalance = df['salary'].value_counts().plot(kind='bar')
    sal_imbalance.set_title('Frequency of salary data \n salary: 0:<=50k , 1:>=50k')
    sal_imbalance.set_xlabel('Frequency')
    sal_imbalance.set_ylabel('Salary')

def oversampling(X,y):
    """
    Trains  data in order to make predictions based on the data provided.
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
    Trains  data in order to make predictions based on the data provided.
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
    Trains  data in order to make predictions based on the data provided.
    """
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y))  
    return X,y


def find_best_num_of_neighbours(X_train, y_train, X_test, y_test):
    """
    Trains  data in order to make predictions based on the data provided.
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
