import pandas as pd
import numpy as np

def dataset_overview(df: pd.DataFrame):
    print(f"Dataset Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns)

    print("\nDatatypes:")
    print(df.dtypes)
    
def check_missing_values(df: pd.DataFrame):
    print(df.isna().sum())
    print((df[df.columns] == '?').sum())
    
def get_feature_type(df: pd.DataFrame):
    num_cols =  df.select_dtypes(
        include=['int64', 'float64']
        ).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=['object']
        ).columns.tolist()
    
    return num_cols, categorical_cols


def target_dist(df: pd.DataFrame):
    count = df['income'].value_counts()
    count_percentage = df['income'].value_counts(normalize=True)*100
    return count, count_percentage


def data_quality_report(df: pd.DataFrame):
    missing_count = df.isna().sum()
    missing_percentage = (
        df.isna().sum() / len(df)
    )*100
    
    dtype = df.dtypes
    report = pd.DataFrame({
        'dtype': dtype,
        'missing_count': missing_count,
        'missing_percentage': missing_percentage,
    })

    return report.sort_values(by="missing_percentage", ascending=False)








