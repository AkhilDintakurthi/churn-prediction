# src/pipeline.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

# 1. Custom Transformer to drop ID and convert TotalCharges
class PreCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=['customerID'], errors='ignore')
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
        return X

# 2. Custom Transformer for Label Encoding
class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
    
    def fit(self, X, y=None):
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col, le in self.encoders.items():
            X[col] = le.transform(X[col].astype(str))
        return X

# 3. Build pipeline
def build_pipeline(top_10_features: list, model_params: dict):
    model = XGBClassifier(**model_params, use_label_encoder=False, eval_metric='logloss', random_state=42)

    pipeline = Pipeline(steps=[
        ('cleaning', PreCleaner()),
        ('label_encode', LabelEncoderWrapper()),
        ('feature_select', ColumnTransformer([
            ('top10', 'passthrough', top_10_features)
        ], remainder='drop')),
        ('model', model)
    ])

    return pipeline

