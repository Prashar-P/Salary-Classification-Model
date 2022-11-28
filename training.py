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
    train data model using KNN and Logistic Regression
    Scalars, Confusion Matrix plotting and under/oversampling techiniques
    to be used can be set in this function
    """

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    #SET UNDER/OVER/SMOTE SAMPLING HERE

    # model_tuning.check_imbalance(df)
    # X, y = model_tuning.fix_imbalanced_data_with_smote(X,y)
    # X, y = model_tuning.oversampling(X,y)
    # X, y = model_tuning.undersampling(X,y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # #LOGISTIC REGRESSION  & KNN WITH NO SCALARS

    # fig, axs = plt.subplots(figsize=(30,5),ncols=2)

    # model = Logistic_Regression(X_train, y_train, X_test, y_test,X,y)
    # plot_confusion_matrix(X_test, y_test, model,"Logistic Regression", axs,0)

  
    # model = k_nearest_algorithm(X_train, y_train, X_test, y_test,X,y)
    # plot_confusion_matrix(X_test, y_test, model,"KNN", axs,1)

    #KNN REGRESSION WITH SCALARS

    fig, axs = plt.subplots(figsize=(30,5),ncols=4)
    
    X_train, X_test = model_tuning.standardize_using_scalar(X_train,X_test)
    print("KNN standardize using scalar accuracy:")
    model = k_nearest_algorithm(X_train, y_train, X_test, y_test,X,y)
    plot_confusion_matrix(X_test, y_test, model,"Standard Scalar",axs,0)

    X_train, X_test = model_tuning.standardize_using_robust(X_train,X_test)
    print("KNN standardize using robust accuracy:")
    model = k_nearest_algorithm(X_train, y_train, X_test, y_test,X,y)
    plot_confusion_matrix(X_test, y_test, model,"Robust Scalar", axs,1)

    X_train, X_test = model_tuning.standardize_using_minmax(X_train,X_test)
    print("KNN standardize using minmax accuracy:")
    model = k_nearest_algorithm(X_train, y_train, X_test, y_test,X,y)
    plot_confusion_matrix(X_test, y_test, model,"MinMax Scalar",axs,2)

    X_train, X_test = model_tuning.standardize_using_normalizer(X_train,X_test)
    print("KNN standardize using normalizer accuracy:")
    model = k_nearest_algorithm(X_train, y_train, X_test, y_test,X,y)
    plot_confusion_matrix(X_test, y_test, model,"Normalizer", axs,3)

     #LOGISTIC REGRESSION WITH SCALARS

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
    Classify data using KNN algorithm
    Note: GridSearch can be used here
    """   
    
    from sklearn.neighbors import KNeighborsClassifier
    # neighbours = model_tuning.find_best_num_of_neighbours(X_train, y_train, X_test, y_test)
    params  = { 
        'n_neighbors' : range(1,30),
        'weights' : ['uniform'],
        'metric' : ['euclidean']
        }
    # set n_neighbors=1 as 1 to see accuracy at 1 neighbour
    knn = KNeighborsClassifier(n_neighbors=1)
    accuracy(X,y,X_train,y_train,X_test,y_test,knn)
    ##Set when using grid search:
    # knn = KNeighborsClassifier()
    # model_tuning.GridSearch(knn,X_train,y_train,X_test,y_test,params)
    
    return knn

# Logistic Regression Algorithm 

def Logistic_Regression(X_train, y_train, X_test, y_test,X,y):
    """
    Classify data using Logistic Regression algorithm
    Note GridSearch can be used here
    """     
    from sklearn.linear_model import LogisticRegression
    params  = {
        "solver": ["liblinear"],
        'penalty' : ['l1', 'l2'],
        'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }
    logistic = LogisticRegression(max_iter=10000)
    accuracy(X,y,X_train,y_train,X_test,y_test,logistic)
    ##Set when using grid search:
    # model_tuning.GridSearch(logistic,X_train,y_train,X_test,y_test,params)
    
    return logistic

def accuracy(X,y,X_train,y_train,X_test,y_test,model):
    """
    Print accuracy scores for test and train data and produce classification result
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
    Plot accuracy onto a confusion matrix
    See:https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}' for v1, v2 in
          zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    cm_plt =  sns.heatmap(cm, fmt='', annot=labels, cmap='Blues',ax=axs[axis])
    cm_plt.set_title('Confusion Matrix using ' + scalar)
    

# Cross-Validation

def cross_validation(model,X,y):
    """
    Apply cross-validation to classification
    """
    scores = cross_val_score(model,X, y, cv=5, scoring="accuracy")
    print("Scores: ", scores)
    print("Average model accuracy: {:.2f}".format(scores.mean()))

