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
import preprocessing


# Scaling

def standardize_using_scalar(X_train,X_test):
    """
    Trains  data in order to make predictions based on the data provided.
    """
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_train = standard_scaler.transform(X_test)
    return X_train

def standardize_using_robust(X_train,X_test):
    """
    Trains  data in order to make predictions based on the data provided.
    """ 
    robust_scaler = RobustScaler()
    X_train = robust_scaler.fit_transform(X_train)
    X_train = robust_scaler.transform(X_test)
    return X_train

def standardize_using_minmax(X_train,X_test):
    """
    Trains  data in order to make predictions based on the data provided.
    """
    minmax_scaler = MinMaxScaler()
    X_train = minmax_scaler.fit_transform(X_train)
    X_test= minmax_scaler.transform(X_test)
    return X_train

def standardize_using_normalizer(X_train,X_test):
    """
    Trains  data in order to make predictions based on the data provided.
    """
    normalizer = Normalizer()
    X_train = normalizer.fit_transform(X_train)
    X_train = normalizer.transform(X_test)
    return X_train

# Under and Oversampling

def oversampling(X,y):
    """
    Trains  data in order to make predictions based on the data provided.
    """
    X, y = make_classification(n_samples=32561, n_features=15, n_informative=12,
                           weights= None )
    ros = RandomOverSampler()
    ros.fit(X, y)
    X_ros, y_ros = ros.fit_resample(X, y)


def undersampling(X,y):
    """
    Trains  data in order to make predictions based on the data provided.
    """
    X, y = make_classification(n_samples=32561, n_features=15, n_informative=12,
                        weights= None )
    rus = RandomOverSampler()
    rus.fit(X, y)
    X_rus, y_rus = rus.fit_resample(X, y)


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
