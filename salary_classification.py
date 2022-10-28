# dataset used: https://www.kaggle.com/datasets/ayessa/salary-prediction-classification

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

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