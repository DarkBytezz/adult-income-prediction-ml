import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer



def load_data(path):
    return pd.read_csv(path)

def replace_question_marks(df: pd.DataFrame):
    df = df.replace("?", np.nan)
    return df

def get_feature_type(df: pd.DataFrame):
    num_cols =  df.select_dtypes(
        include=['int64', 'float64']
        ).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=['object']
        ).columns.tolist()
    
    return num_cols, categorical_cols


def fill_missing_values(df: pd.DataFrame):
    num_cols, categorical_cols = get_feature_type(df)
    # count = df[categorical_cols].isna().sum()  
    # print(count)
    
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    return df

def split_data(df: pd.DataFrame):
    X = df.drop(columns=['income'])
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def build_preprocessor(num_cols, categorical_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols
            ),
            (
                "num", StandardScaler(), num_cols
            )
        ],
        remainder='passthrough'
    )
    
    return preprocessor




