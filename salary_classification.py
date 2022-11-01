# dataset used: https://www.kaggle.com/datasets/ayessa/salary-prediction-classification

#%%
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
    # check_data(df)
    remove_spaces(df)
    remove_null_values(df)
    remove_duplicates(df)
    convert_categorical_to_numerical(df)
    handle_outliers(df)
    # print(df.head(30))

"""
    The following function trains the data in order to make predictions based on the data provided.
"""
def train_data():
    print("training")

"""
    The following function aims to preprocess the data so that is in the correct format for training:
    This includes:
    - removing null data
    - removing  duplicate data
    - handling of outliers   
"""
def remove_null_values(df):
    #set  ? to be null values
    df.replace("?", pd.NA, inplace=True)
    null_values = df.isnull().sum().sum()
    if (null_values) > 0:
        print("Null Values are present... removing relevant data samples")
        df.dropna(inplace=True)
    return df

"""
    The following function aims to preprocess the data so that is in the correct format for training:
    This includes:
    - removing null data
    - removing  duplicate data
    - handling of outliers   
"""
def remove_duplicates(df):
    duplicate_values = df.duplicated().sum()
    if (duplicate_values > 0):
        print("Duplicate data was found. Removing duplicate data...")
        df.drop_duplicates(keep='first',inplace=True)
    return df

"""
    The following function aims to preprocess the data so that is in the correct format for training:
    This includes:
    - removing null data
    - removing  duplicate data
    - handling of outliers   
"""
def remove_spaces(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].map(str.strip)
    return df

"""
    The following function aims to preprocess the data so that is in the correct format for training:
    This includes:
    - removing null data
    - removing  duplicate data
    - handling of outliers   
"""
def convert_categorical_to_numerical(df):
    from sklearn import preprocessing 
    le = preprocessing.LabelEncoder()
    categorical_columns = df.select_dtypes(include = "object").columns
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

"""
    The following function aims to preprocess the data so that is in the correct format for training:
    This includes:
    - removing null data
    - removing  duplicate data
    - handling of outliers   
"""
def handle_outliers(df):
    print("handling outliers .. ")
    #display outliers on a graph  - we will use a box plot for this
    #predictions made on the persons occupation/education
    sns.boxplot(data=df, y="age", x="salary")
    sns.catplot(data=df, y="occupation", hue="salary", kind="count", palette="pastel")
    plt.pyplot.figure(figsize=(15, 10))
    from scipy import stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    from scipy.stats import zscore
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    sns.heatmap(df.corr(), annot=True)
    return df

def main():
    df = pd.read_csv("salary.csv")
    data_preprocessing(df)

if __name__ == "__main__":
    main()
# %%