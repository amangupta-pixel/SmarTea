# backend/core/preprocess.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

DATA_PATH = os.path.join("data", "Lead_data.csv")

def load_data():
    """
    Loads the lead data and returns X (features) and y (target)
    """
    df = pd.read_csv(DATA_PATH)

    # Remove 'New' status rows
    df = df[df['status'] != 'New']

    # Map status to binary conversion
    df['is_converted'] = df['status'].apply(lambda x: 1 if x in ['Converted', 'Qualified'] else 0)

    features = ['leadSource', 'industry', 'state', 'numberOfEmployees', 'annualRevenue']
    target = 'is_converted'

    X = df[features]
    y = df[target]

    return X, y

def build_pipeline(model):
    """
    Builds a full pipeline combining preprocessing and model
    """
    categorical_cols = ['leadSource', 'industry', 'state']
    numerical_cols = ['numberOfEmployees', 'annualRevenue']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline