import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# [1] load data

def preprocessed_data():
    BASE_DIR = Path.cwd()

    DATA_PATH = BASE_DIR.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv" 

    data = pd.read_csv(DATA_PATH)

    print('[1] Dataset loaded')
    print(f'Dataset shape: {data.shape}')

    # [2] Basic preprocessing 

    #drop ID column
    data = data.drop('customerID', axis=1)

    #Change total charges to numeric
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors ='coerce')
    data = data.dropna()

    #seniorcitizen should be 0/1 ints if present
    if 'SeniorCitizen' in data.columns:
        data['SeniorCitizen'] = data['SeniorCitizen'].fillna(0).astype(int)

    print('--------------------------------------------------------')
    print('[2] Basic preprocessing completed.')

       # [3] Encode categorical features

    # Separate categorical columns
    categorical_cols = data.select_dtypes(include='object').columns

    # Binary categorical columns (2 unique values)
    binary_cols = [
        col for col in categorical_cols
        if data[col].nunique() == 2
    ]

    # Multi-class categorical columns (>2 unique values)
    multi_class_cols = [
        col for col in categorical_cols
        if data[col].nunique() > 2
    ]

    #encode binary columns
    # Convert Yes/No type columns to 0/1
    for col in binary_cols:
        data[col] = pd.factorize(data[col])[0]
    
    #encode multi-class columns
    # Apply one-hot encoding only to columns
    # with more than 2 categories

    encoded_data = pd.get_dummies(data,columns=multi_class_cols,drop_first=True,dtype=int)

    print('--------------------------------------------------------')
    print('[3] Categorical data encoded')
    print(f'encoded_data.shape: {encoded_data.shape}')
    print(encoded_data.columns)

    X = encoded_data.drop('Churn', axis=1)
    y = encoded_data['Churn']

    return X, y

    

