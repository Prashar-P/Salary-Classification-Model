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
import model_tuning

def train_model(df):
    
    """
    Converts all categorical data to numerical
    """
    # standardize(df)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # model_tuning.oversampling(X,y)
    # model_tuning.undersampling(X,y)
    
    # fig, axs = plt.subplots(figsize=(30,5),ncols=4)
    
    # standardize_using_scalar(X_train,X_test)
    # print("KNN standardize using scalar accuracy:")
    # model = k_nearest_algorithm(X_train, y_train, X_test, y_test)
    # # plot_confusion_matrix(X_test, y_test, model,"Standard Scalar",axs,0)

    # standardize_using_robust(X_train,X_test)
    # print("KNN standardize using robust accuracy:")
    # model = k_nearest_algorithm(X_train, y_train, X_test, y_test)
    # # plot_confusion_matrix(X_test, y_test, model,"Robust Scalar", axs,1)

    # standardize_using_minmax(X_train,X_test)
    # print("KNN standardize using minmax accuracy:")
    # model = k_nearest_algorithm(X_train, y_train, X_test, y_test)
    # # plot_confusion_matrix(X_test, y_test, model,"MinMax Scalar",axs,2)

    # standardize_using_normalizer(X_train,X_test)
    # print("KNN standardize using normalizer accuracy:")
    # model = k_nearest_algorithm(X_train, y_train, X_test, y_test)
    # # plot_confusion_matrix(X_test, y_test, model,"Normalizer", axs,3)

    fig, axs = plt.subplots(figsize=(30,5),ncols=4)
    
    model_tuning.standardize_using_scalar(X_train,X_test)
    print("Logistic Regression using scalar accuracy:")
    model = Logistic_Regression(X_train, y_train, X_test, y_test,X,y)
    # plot_confusion_matrix(X_test, y_test, model,"Standard Scalar",axs,0)

    model_tuning.standardize_using_robust(X_train,X_test)
    print("Logistic Regression using robust accuracy:")
    model = Logistic_Regression(X_train, y_train, X_test, y_test,X,y)
    # plot_confusion_matrix(X_test, y_test, model,"Robust Scalar", axs,1)

    model_tuning.standardize_using_minmax(X_train,X_test)
    print("Logistic Regression standardize using minmax accuracy:")
    model = Logistic_Regression(X_train, y_train, X_test, y_test,X,y)
    # plot_confusion_matrix(X_test, y_test, model,"MinMax Scalar",axs,2)

    model_tuning.standardize_using_normalizer(X_train,X_test)
    print("Logistic Regression using normalizer accuracy:")
    model = Logistic_Regression(X_train, y_train, X_test, y_test,X,y)
    # plot_confusion_matrix(X_test, y_test, model,"Normalizer", axs,3)



def k_nearest_algorithm(X_train, y_train, X_test, y_test,X,y):
    """
    Converts all categorical data to numerical
    """   
    from sklearn.neighbors import KNeighborsClassifier
    neighbours = model_tuning.find_best_num_of_neighbours(X_train, y_train, X_test, y_test)
    # set n_neighbors=1 as 1 to see accuracy at 1 neighbour
    knn = KNeighborsClassifier(n_neighbors=neighbours)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    target = ["<50k",">50k"]
    print(classification_report(y_test , y_pred,target_names=target))
    cross_validation(knn,X,y)
    return knn


def Logistic_Regression(X_train, y_train, X_test, y_test,X,y):
    """
    Converts all categorical data to numerical
    """   
    from sklearn.linear_model import LogisticRegression
    logistic = LogisticRegression(max_iter=10000)
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)
    target = ["<50k",">50k"]
    print(classification_report(y_test , y_pred,target_names=target))
    cross_validation(logistic,X,y)
    return logistic

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
