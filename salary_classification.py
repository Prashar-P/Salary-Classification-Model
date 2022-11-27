# dataset used: https://www.kaggle.com/datasets/ayessa/salary-prediction-classification

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import preprocessing
import training


def main():
    """
    Main Code Function to run classification
    Accompanying Files:
        preprocessing.py
        model_tuning.py
        training.py
        salary.csv
    Modify these files to add scalars/sampling/fitting/gridsearch
    """
    df = pd.read_csv("salary.csv")
    # preprocessing.data_analysis(df)
    preprocessing.data_preprocessing(df)
    training.train_model(df)

if __name__ == "__main__":
    main()
# %%
