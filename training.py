import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import sklearn

import preprocessing


def train_model(df):
    
    """
    Converts all categorical data to numerical
    """
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = k_nearest_algorithm(X_train, y_train, X_test, y_test)
    plot_confusion_matrix(X_test, y_test, model)


def k_nearest_algorithm(X_train, y_train, X_test, y_test):
    """
    Converts all categorical data to numerical
    """
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train, y_train)
    print("accuracy:")
    print(knn.score(X_test, y_test))
    return knn


def plot_confusion_matrix(X_test, y_test, model):
    """
    Converts all categorical data to numerical
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix

    y_pred = model.predict(X_test)
    confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(model, X_test, y_test, cmap="Blues")
