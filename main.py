#############################################
# ADVANCED FUNCTIONAL EXPLORATORY DATA ANALYSIS (EDA) ON TITANIC DATASET
#############################################
# 1. Overview
# 2. Categorical Variables Analysis
# 3. Numerical Variables Analysis
# 4. Target Variable Analysis
# 5. Correlation Analysis

#############################################

# Importing helper libraries and adjusting the display settings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 50)

# Calling the dataset
df = pd.read_csv("datasets/breast_cancer.csv")
df.head()
df.tail()
df.info()

#selecting the required columns
df = df.iloc[:, 1:-1]
df.head()


# Converting boolean values to integer for statistical analysis, visualization and modeling requirement
for col in df.columns:
    if df[col].dtypes == 'bool':
        df[col] = df[col].astype(int)

# Quick overview of the dataset
def quick_summary(dataframe, head=8):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head).T)
    print("##################### Tail #####################")
    print(dataframe.tail(head).T)
    print("##################### Info #####################")
    print(dataframe.info(head))
    print("##################### Missing Values #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

quick_summary(df)

# Classifying the columns

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Gives the names of categorical, numerical and categorical but cardinal variables in the data set.

    Parameters
    ----------
    dataframe: dataframe
        It is the dataframe from which variable names are to be retrieved.
    cat_th: int, float
        Threshold value for numeric but categorical variables.
    car_th: int, float
        Threshold value for categorical but not cardinal variables.

    Returns
    -------
    cat_cols: list
        categorical variable list
    num_cols: list
        numeric variable list
    cat_but_car:
        Cardinal variable list with categorical view

    Notes
    -------
    cat_cols + num_cols + cat_but_car = Total number of variables.
    num_but_cat is inside cat_cols.

    """
    cat_cols = [col for col in dataframe.columns if
                str(dataframe[col].dtypes) in ["category", "object", "bool"]]


    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]


    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]

    num_cols = [col for col in num_cols if col not in cat_cols]


    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {(len(num_but_cat))}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Summarizing and plotting categorical variables

def cat_summary(dataframe, categorical_col, plot=False):
    print(pd.DataFrame({categorical_col: dataframe[categorical_col].value_counts(),
    "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}))
    print("##########################################")


    if plot:
        sns.countplot(x=dataframe[categorical_col], data=dataframe)
        plt.show(block=True)

# Summarizing and plotting numerical variables
def num_summary(dataframe, numerical_col, plot=False):
            quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
            print(dataframe[numerical_col].describe(quantiles).T)

            if plot:
                dataframe[numerical_col].hist()
                plt.xlabel(numerical_col)
                plt.title(numerical_col)
                plt.show(block=True)

# Applying the functions cat_summary and num_summary to each column
for col in cat_cols:
    cat_summary(df, col, plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# Target variable analysis

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"Target_Count": dataframe.groupby(categorical_col, observed=False)[target].value_counts()}), end="\n\n\n")


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "diagnosis", col)

for col in num_cols:
    target_summary_with_num(df, "diagnosis", col)


# Correlation Analysis

num_cols = [col for col in df.columns if df[col].dtypes in [int, float]]

def high_correlated_cols(dataframe, num_cols, plot=False, corr_th=0.90):
    corr = dataframe[num_cols].corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (8, 8)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

drop_list = high_correlated_cols(df, num_cols, plot=True)

num_cols = [col for col in num_cols if col not in drop_list]
high_correlated_cols(df.drop(drop_list, axis=1), num_cols, plot=True)



