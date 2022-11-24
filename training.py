from turtle import mode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 

import preprocessing
import model_tuning

def train_model(df):
    
    """
    Converts all categorical data to numerical
    """

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    #model_tuning.check_imbalance(df)
    # X, y = model_tuning.fix_imbalanced_data_with_smote(X,y)
    # X, y = model_tuning.oversampling(X,y)
    # X, y = model_tuning.undersampling(X,y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # fig, axs = plt.subplots(figsize=(30,5),ncols=4)
    # X_train, X_test = model_tuning.standardize_using_scalar(X_train,X_test)
    # print("KNN standardize using scalar accuracy:")
    # model = k_nearest_algorithm(X_train, y_train, X_test, y_test,X,y)
    # plot_confusion_matrix(X_test, y_test, model,"Standard Scalar",axs,0)

    # X_train, X_test = model_tuning.standardize_using_robust(X_train,X_test)
    # print("KNN standardize using robust accuracy:")
    # model = k_nearest_algorithm(X_train, y_train, X_test, y_test,X,y)
    # plot_confusion_matrix(X_test, y_test, model,"Robust Scalar", axs,1)

    # X_train, X_test = model_tuning.standardize_using_minmax(X_train,X_test)
    # print("KNN standardize using minmax accuracy:")
    # model = k_nearest_algorithm(X_train, y_train, X_test, y_test,X,y)
    # plot_confusion_matrix(X_test, y_test, model,"MinMax Scalar",axs,2)

    # X_train, X_test = model_tuning.standardize_using_normalizer(X_train,X_test)
    # print("KNN standardize using normalizer accuracy:")
    # model = k_nearest_algorithm(X_train, y_train, X_test, y_test,X,y)
    # plot_confusion_matrix(X_test, y_test, model,"Normalizer", axs,3)

    fig, axs = plt.subplots(figsize=(30,5),ncols=4)
    
    X_train, X_test = model_tuning.standardize_using_scalar(X_train,X_test)
    print("Logistic Regression using scalar accuracy:")
    model = Logistic_Regression(X_train, y_train, X_test, y_test,X,y)
    plot_confusion_matrix(X_test, y_test, model,"Standard Scalar",axs,0)

    X_train, X_test = model_tuning.standardize_using_robust(X_train,X_test)
    print("Logistic Regression using robust accuracy:")
    model = Logistic_Regression(X_train, y_train, X_test, y_test,X,y)
    plot_confusion_matrix(X_test, y_test, model,"Robust Scalar", axs,1)

    X_train, X_test = model_tuning.standardize_using_minmax(X_train,X_test)
    print("Logistic Regression standardize using minmax accuracy:")
    model = Logistic_Regression(X_train, y_train, X_test, y_test,X,y)
    plot_confusion_matrix(X_test, y_test, model,"MinMax Scalar",axs,2)

    X_train, X_test = model_tuning.standardize_using_normalizer(X_train,X_test)
    print("Logistic Regression using normalizer accuracy:")
    model = Logistic_Regression(X_train, y_train, X_test, y_test,X,y)
    plot_confusion_matrix(X_test, y_test, model,"Normalizer", axs,3)


# K-Nearest Algorithm 

def k_nearest_algorithm(X_train, y_train, X_test, y_test,X,y):
    """
    Converts all categorical data to numerical
    """   
    from sklearn.neighbors import KNeighborsClassifier
    # range = list(range(1, 31))
    neighbours = model_tuning.find_best_num_of_neighbours(X_train, y_train, X_test, y_test)
    # set n_neighbors=1 as 1 to see accuracy at 1 neighbour
    knn = KNeighborsClassifier(n_neighbors=neighbours)
    accuracy(X,y,X_train,y_train,X_test,y_test,knn)
    # print(model_tuning.GridSearch(knn,X_train,y_train,X_test,y_test,range))
    return knn

# Logistic Regression Algorithm 

def Logistic_Regression(X_train, y_train, X_test, y_test,X,y):
    """
    Converts all categorical data to numerical
    """   
    from sklearn.linear_model import LogisticRegression
    logistic = LogisticRegression(max_iter=10000)
    accuracy(X,y,X_train,y_train,X_test,y_test,logistic)
    return logistic

def accuracy(X,y,X_train,y_train,X_test,y_test,model):
    """
    Converts all categorical data to numerical
    """   
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    target = ["<=50K",">=50k"]
    score = model.score(X_train, y_train)
    print("Training score: {:.3f}".format(score))
    score = model.score(X_test, y_test)
    print("Test score: {:.3f}".format(score))
    # print("Balanced Accuracy Score:", balanced_accuracy_score(y_test,y_pred))
    print(classification_report(y_test , y_pred,target_names=target))
    cross_validation(model,X,y)   

# Confusion Matrix

def plot_confusion_matrix(X_test, y_test, model, scalar, axs,axis):
    """
    Converts all categorical data to numerical
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    cm_plt =  sns.heatmap(cm/np.sum(cm), fmt='.2%', annot=True, cmap='Blues',ax=axs[axis])
    cm_plt.set_title('Confusion Matrix using ' + scalar)
    cm_plt.xaxis.set_ticklabels(['True','False'])
    cm_plt.yaxis.set_ticklabels(['Negative','Posotive'])

# Cross-Validation

def cross_validation(model,X,y):
    """
    Trains  data in order to make predictions based on the data provided.
    """
    scores = cross_val_score(model,X, y, cv=5, scoring="accuracy")
    print("Scores: ", scores)
    print("Average model accuracy: {:.2f}".format(scores.mean()))

