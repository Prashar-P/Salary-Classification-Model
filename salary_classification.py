# dataset used: https://www.kaggle.com/datasets/ayessa/salary-prediction-classification

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

def check_data(df):
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.nunique())
    print(df.isnull().sum())
    print(df.duplicated().sum())

"""
    The following function aims to preprocess the data so that is in the correct format for training:
    This includes:
    - removing null data
    - removing  duplicate data
    - handling of outliers   

"""
def data_preprocessing(df):
    print("preprocessing")
    remove_null_values(df)

"""
    The following function trains the data in order to make predictions based on the data provided.
"""
def train_data():
    print("training")

def remove_null_values(df):
    #set  ? to be null values
    df.replace(" ?", pd.NA, inplace=True)
    null_values = df.isnull().sum().sum()
    if (null_values) > 0:
        print("Null Values are present... removing relevant data samples")
        df.dropna(inplace=True)
    return df

def remove_duplicates(df):
    duplicate_values = df.duplicated().sum()
    if (duplicate_values > 0):
        print("Duplicate data was found. Removing duplicate data...")
        df.drop_duplicates(keep='first',inplace=True)
    return df

def remove_spaces(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].map(str.strip)

def main():
    df = pd.read_csv("salary.csv")
    # check_data(df)
    remove_spaces(df)
    remove_null_values(df)
    remove_duplicates(df)


if __name__ == "__main__":
    main()