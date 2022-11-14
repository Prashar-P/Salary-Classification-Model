# dataset used: https://www.kaggle.com/datasets/ayessa/salary-prediction-classification

#%%
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

import preprocessing
import training


def main():
    df = pd.read_csv("salary.csv")
    # preprocessing.check_data(df)
    preprocessing.data_preprocessing(df)
    training.train_model(df)


if __name__ == "__main__":
    main()
# %%
