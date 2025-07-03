# 🧠 Customer Churn Prediction & Retention Strategy

## 📌 Project Overview

Customer churn is a major concern for subscription-based businesses, especially in the telecom sector. This project leverages the Telco Customer Churn dataset to develop machine learning models that can identify customers at risk of churning. The goal is to support proactive retention strategies that reduce churn and increase customer lifetime value.

---

## 📊 Dataset

* **Source**: [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
* **Rows**: \~7,000
* **Target Variable**: `Churn` (Yes/No)
* **Features include**:

  * **Demographics**: gender, SeniorCitizen, Partner, Dependents
  * **Service Usage**: PhoneService, InternetService, StreamingTV, etc.
  * **Account Info**: tenure, MonthlyCharges, TotalCharges
  * **Contracts & Billing**: Contract, PaymentMethod, PaperlessBilling
  * **Support**: TechSupport, OnlineBackup, DeviceProtection, etc.

---

## 🎯 Business Objective

* **Goal**: Predict customers likely to churn
* **Action**: Enable targeted retention strategies
* **Value**: Reduce churn rate, increase profitability, improve customer satisfaction

---

## 🔠 Evaluation Metrics

Given potential class imbalance, the following metrics were used:

* **F1 Score**
* **AUC-ROC**
* **Precision / Recall**
* **Confusion Matrix**

---

## 🚀 Final Pipeline Components

1. Data Cleaning (drop ID, convert TotalCharges)
2. Label Encoding for categorical features
3. Feature Selection (Top 10 features using SelectKBest)
4. Model: XGBoost Classifier (tuned with RandomizedSearchCV)
5. Model Interpretability with SHAP
6. Flask API for inference (via `src/app.py`)
7. Testing via `test_api.py`

> **Note**: Docker support is optional and currently disabled due to versioning incompatibility during model serialization.

---

## 🧱 Project Structure

```
Churn Prediction/
├── data/                       # Raw dataset (.csv)
├── notebooks/                 # Jupyter notebooks (EDA, modeling)
├── src/
│   ├── app.py                 # Flask API
│   ├── pipeline.py            # Pipeline definition
│   ├── data.py                # Processed input features
├── churn_pipeline.pkl         # Trained pipeline (scikit-learn==1.6.1)
├── test_api.py                # API test script
├── requirements.txt           # Dependencies
├── Dockerfile (optional)
└── README.md
```

---

## ⚙️ Setup Instructions

1. **(Optional) Create a virtual environment:**

   ```bash
   python -m venv venv
   .\venv\Scripts\Activate
   ```
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Run the API:**

   ```bash
   python src/app.py
   ```
4. **Test the API:**

   ```bash
   python test_api.py
   ```

---

## 🌿 EDA Highlights

> → See [`01_EDA.ipynb`](notebooks/01_EDA.ipynb) for full visual insights.

* **Churn rate** is around 26%
* **Short-tenure users** and **high monthly charges** increase churn
* **Month-to-month contracts** are more churn-prone
* **Electronic check** is the riskiest payment method
* **Senior citizens**, especially without tech support, churn more

---

## 📈 Model Performance

* **Model**: XGBoost (tuned)
* **Accuracy**: 96.13%
* **ROC-AUC**: 0.9953

---

## 🧠 SHAP Interpretability

### Top Global Features

* `MonthlyCharges`
* `tenure`
* `Contract`

### Insights

* **High MonthlyCharges + Low Tenure** = Highest churn risk
* **Bundled services and long-term contracts** reduce churn

---

## 🌟 Retention Strategy Recommendations

* 🏱 **Offer loyalty incentives** to high-paying new customers
* 🛌 **Encourage contract upgrades** from monthly to yearly
* 📧 **Educate users** with onboarding content for Fiber plans
* 🚑 **Promote support services** like Tech Support + Security

---

## 🚀 Deployment Notes

* Flask API is available via `src/app.py`
* Run `python test_api.py` to verify predictions
* `churn_pipeline.pkl` is compatible with `scikit-learn==1.6.1`
* Docker build was attempted but disabled due to versioning issues with pipeline serialization

---

## 👩‍💻 Author

* **Name**: Akhil Sai Kalyan Dintakurthi
* **Email**: [kalyan.dintakurthi@gmail.com](mailto:kalyan.dintakurthi@gmail.com)
