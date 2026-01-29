"""
Data Preprocessing Script for Crime Safety Classification Project
"""
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('../data/RCD06.20251204131643.csv')

# Inspect columns and basic info
def inspect_data(df):
    print(df.info())
    print(df.head())
    print(df.describe(include='all'))

# Handle missing values
def handle_missing(df):
    # Example: fill missing values with mode or median
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

# Encode categorical variables
def encode_categorical(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

# Main preprocessing pipeline
def preprocess():
    inspect_data(df)
    df_clean = handle_missing(df)
    df_encoded = encode_categorical(df_clean)
    df_encoded.to_csv('../data/processed_crime_data.csv', index=False)
    print('Preprocessing complete. Processed data saved.')

if __name__ == '__main__':
    preprocess()
