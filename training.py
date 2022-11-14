import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler
from sklearn.metrics import classification_report
import preprocessing


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
    
    fig, axs = plt.subplots(figsize=(30,5),ncols=4)
    
    standardize_using_scalar(X_train,X_test)
    print("KNN standardize using scalar accuracy:")
    model = k_nearest_algorithm(X_train, y_train, X_test, y_test)
    plot_confusion_matrix(X_test, y_test, model,"Standard Scalar",axs,0)

    standardize_using_robust(X_train,X_test)
    print("KNN standardize using robust accuracy:")
    model = k_nearest_algorithm(X_train, y_train, X_test, y_test)
    plot_confusion_matrix(X_test, y_test, model,"Robust Scalar", axs,1)

    standardize_using_minmax(X_train,X_test)
    print("KNN standardize using minmax accuracy:")
    model = k_nearest_algorithm(X_train, y_train, X_test, y_test)
    plot_confusion_matrix(X_test, y_test, model,"MinMax Scalar",axs,2)

    standardize_using_normalizer(X_train,X_test)
    print("KNN standardize using normalizer accuracy:")
    model = k_nearest_algorithm(X_train, y_train, X_test, y_test)
    plot_confusion_matrix(X_test, y_test, model,"Normalizer", axs,3)

    fig, axs = plt.subplots(figsize=(30,5),ncols=4)
    
    standardize_using_scalar(X_train,X_test)
    print("Logistic Regression using scalar accuracy:")
    model = Logistic_Regression(X,X_train, y_train, X_test, y_test)
    plot_confusion_matrix(X_test, y_test, model,"Standard Scalar",axs,0)

    standardize_using_robust(X_train,X_test)
    print("Logistic Regression using robust accuracy:")
    model = Logistic_Regression(X,X_train, y_train, X_test, y_test)
    plot_confusion_matrix(X_test, y_test, model,"Robust Scalar", axs,1)

    standardize_using_minmax(X_train,X_test)
    print("Logistic Regression standardize using minmax accuracy:")
    model = Logistic_Regression(X,X_train, y_train, X_test, y_test)
    plot_confusion_matrix(X_test, y_test, model,"MinMax Scalar",axs,2)

    standardize_using_normalizer(X_train,X_test)
    print("Logistic Regression using normalizer accuracy:")
    model = Logistic_Regression(X,X_train, y_train, X_test, y_test)
    plot_confusion_matrix(X_test, y_test, model,"Normalizer", axs,3)



def k_nearest_algorithm(X_train, y_train, X_test, y_test):
    """
    Converts all categorical data to numerical
    """   
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print(knn.score(X_train, y_train))
    return knn
    
def Logistic_Regression(X,X_train, y_train, X_test, y_test):
    """
    Converts all categorical data to numerical
    """   
    from sklearn.linear_model import LogisticRegression
    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
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
    
    print(classification_report(y_test , y_pred))
    

def standardize_using_scalar(X_train,X_test):
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_train = standard_scaler.transform(X_test)
    return X_train

def standardize_using_robust(X_train,X_test):
    robust_scaler = RobustScaler()
    X_train = robust_scaler.fit_transform(X_train)
    X_train = robust_scaler.transform(X_test)
    return X_train

def standardize_using_minmax(X_train,X_test):
    minmax_scaler = MinMaxScaler()
    X_train = minmax_scaler.fit_transform(X_train)
    X_test= minmax_scaler.transform(X_test)
    return X_train

def standardize_using_normalizer(X_train,X_test):
    normalizer = Normalizer()
    X_train = normalizer.fit_transform(X_train)
    X_train = normalizer.transform(X_test)
    return X_train