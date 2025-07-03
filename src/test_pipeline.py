import pandas as pd
from sklearn.model_selection import train_test_split

# Load pre-cleaned raw data
df = pd.read_csv(r'C:\Users\ask12\Desktop\ML Projects\Churn Prediction\data\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Split into features/target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({'No': 0, 'Yes': 1})  # map target to binary

# Train/test split
from sklearn.model_selection import train_test_split
X_raw_train, X_raw_test, y_train_sap, y_test_sap = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save for pipeline
X_raw_train.to_csv('data/X_raw_train.csv', index=False)
y_train_sap.to_csv('data/y_train_sap.csv', index=False)

from src.pipeline import build_pipeline
import joblib

# Top 10 selected features used in model
top_10 = [
    'MonthlyCharges', 'tenure', 'Contract', 'TotalCharges',
    'TechSupport', 'OnlineSecurity', 'PaperlessBilling',
    'InternetService', 'DeviceProtection', 'PaymentMethod'
]

# Best model parameters from your tuning
params = {
    'subsample': 0.8, 'reg_lambda': 1, 'reg_alpha': 0, 'n_estimators': 400,
    'min_child_weight': 3, 'max_depth': 10, 'learning_rate': 0.1, 'gamma': 0.3,
    'colsample_bytree': 0.6
}

# Build pipeline
pipeline = build_pipeline(top_10_features=top_10, model_params=params)


pipeline.fit(X_raw_train, y_train_sap)  # use your raw input DataFrame


joblib.dump(pipeline, 'model/churn_pipeline.pkl')

import pandas as pd

new_customer = pd.DataFrame([{
    'customerID': '1234-ZZZZ',
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 3,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 89.9,
    'TotalCharges': '269.7'
}])

pred = pipeline.predict(new_customer)[0]
prob = pipeline.predict_proba(new_customer)[0][1]

print(f"Prediction: {'Churn' if pred == 1 else 'No Churn'}")
print(f"Probability of Churn: {prob:.4f}")
