import pandas as pd

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap

def load_data(fraud_path='Fraud_Data_merged.csv', creditcard_path='creditcard_cleaned.csv'):
    print("Loading pre-processed datasets for Data Transformation...")
    try:
        fraud_data = pd.read_csv(fraud_path)
        creditcard_data = pd.read_csv(creditcard_path)
        if 'signup_time' in fraud_data.columns and not pd.api.types.is_datetime64_any_dtype(fraud_data['signup_time']):
            fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'], errors='coerce')
        if 'purchase_time' in fraud_data.columns and not pd.api.types.is_datetime64_any_dtype(fraud_data['purchase_time']):
            fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'], errors='coerce')
        initial_rows_fraud = fraud_data.shape[0]
        fraud_data.dropna(subset=['signup_time', 'purchase_time'], inplace=True)
        if fraud_data.shape[0] < initial_rows_fraud:
            print(f"Dropped {initial_rows_fraud - fraud_data.shape[0]} rows from Fraud_Data due to invalid signup_time or purchase_time after loading merged CSV.")
        print("Pre-processed datasets loaded successfully.")
        return fraud_data, creditcard_data
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please ensure the specified CSV files are in the same directory as this script.")
        exit()

def preprocess_data(fraud_data, creditcard_data):
    X_fraud = fraud_data.drop(columns=[
        'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'class'
    ])
    y_fraud = fraud_data['class']
    X_creditcard = creditcard_data.drop(columns=['Time', 'Class'])
    y_creditcard = creditcard_data['Class']
    print("\nPerforming Train-Test Split...")
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(
        X_fraud, y_fraud, test_size=0.3, random_state=42, stratify=y_fraud
    )
    X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test = train_test_split(
        X_creditcard, y_creditcard, test_size=0.3, random_state=42, stratify=y_creditcard
    )
    print("\nHandling Class Imbalance & Encoding Categorical Features...")
    numerical_cols_fraud_train = X_fraud_train.select_dtypes(include=np.number).columns
    categorical_cols_fraud_train = X_fraud_train.select_dtypes(include='object').columns
    encoder_fraud_smote = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_fraud_train_cat_encoded = encoder_fraud_smote.fit_transform(X_fraud_train[categorical_cols_fraud_train])
    X_fraud_train_cat_encoded_df = pd.DataFrame(X_fraud_train_cat_encoded, columns=encoder_fraud_smote.get_feature_names_out(categorical_cols_fraud_train), index=X_fraud_train.index)
    X_fraud_train_combined = pd.concat([X_fraud_train[numerical_cols_fraud_train], X_fraud_train_cat_encoded_df], axis=1)
    smote = SMOTE(random_state=42)
    X_fraud_train_res, y_fraud_train_res = smote.fit_resample(X_fraud_train_combined, y_fraud_train)
    print(f"Fraud_Data training set resampled shape (SMOTE): {Counter(y_fraud_train_res)}")

    rus = RandomUnderSampler(random_state=42)
    X_creditcard_train_res, y_creditcard_train_res = rus.fit_resample(X_creditcard_train, y_creditcard_train)
    print(f"Creditcard_Data training set resampled shape (RandomUnderSampler): {Counter(y_creditcard_train_res)}")
    print("\nPerforming Normalization and Scaling...")

    scaler_fraud = StandardScaler()
    numerical_cols_fraud_resampled = X_fraud_train_res.columns
    X_fraud_train_res[numerical_cols_fraud_resampled] = scaler_fraud.fit_transform(X_fraud_train_res[numerical_cols_fraud_resampled])
    X_fraud_test_cat_encoded = encoder_fraud_smote.transform(X_fraud_test[categorical_cols_fraud_train])
    X_fraud_test_cat_encoded_df = pd.DataFrame(X_fraud_test_cat_encoded, columns=encoder_fraud_smote.get_feature_names_out(categorical_cols_fraud_train), index=X_fraud_test.index)
    X_fraud_test_combined = pd.concat([X_fraud_test[numerical_cols_fraud_train], X_fraud_test_cat_encoded_df], axis=1)
    X_fraud_test_scaled = scaler_fraud.transform(X_fraud_test_combined)
    X_fraud_test_scaled = pd.DataFrame(X_fraud_test_scaled, columns=X_fraud_train_res.columns, index=X_fraud_test_combined.index)
    scaler_creditcard = StandardScaler()
    numerical_cols_creditcard = X_creditcard_train_res.select_dtypes(include=np.number).columns
    X_creditcard_train_res[numerical_cols_creditcard] = scaler_creditcard.fit_transform(X_creditcard_train_res[numerical_cols_creditcard])
    X_creditcard_test[numerical_cols_creditcard] = scaler_creditcard.transform(X_creditcard_test[numerical_cols_creditcard])
    return (X_fraud_train_res, y_fraud_train_res, X_fraud_test_scaled, y_fraud_test,
            X_creditcard_train_res, y_creditcard_train_res, X_creditcard_test, y_creditcard_test)

def train_models(X_fraud_train, y_fraud_train, X_creditcard_train, y_creditcard_train):
    print("\n--- Training Random Forest Models ---")

    rf_fraud = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
    rf_creditcard = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
    print("\nTraining Random Forest for Fraud_Data...")

    rf_fraud.fit(X_fraud_train, y_fraud_train)
    print("Random Forest for Fraud_Data trained.")
    print("\nTraining Random Forest for Creditcard_Data...")

    rf_creditcard.fit(X_creditcard_train, y_creditcard_train)
    print("Random Forest for Creditcard_Data trained.")
    return rf_fraud, rf_creditcard

def explain_with_shap(rf_fraud, X_fraud_test, rf_creditcard, X_creditcard_test):
    print("\n--- Model Explainability with SHAP ---")
    print("\nInterpreting Random Forest Model for Fraud_Data.csv...")

    explainer_rf_fraud = shap.TreeExplainer(rf_fraud)
    shap_sample_fraud = X_fraud_test.sample(n=min(1000, X_fraud_test.shape[0]), random_state=42)
    shap_values_rf_fraud = explainer_rf_fraud.shap_values(shap_sample_fraud)

    print("\nGenerating SHAP Summary Plot for Fraud_Data.csv...")

    shap.summary_plot(shap_values_rf_fraud[1], shap_sample_fraud, plot_type="bar", show=False)
    plt.title('SHAP Global Feature Importance (Fraud_Data.csv)')
    plt.tight_layout()
    plt.show()
    shap.summary_plot(shap_values_rf_fraud[1], shap_sample_fraud, show=False)
    plt.title('SHAP Summary Plot (Feature Impact and Direction) for Fraud_Data.csv')

    plt.tight_layout()
    plt.show()

    print("\nGenerating SHAP Force Plot for a sample prediction (Fraud_Data.csv)...")

    sample_idx_fraud = 0
    shap.initjs()
    shap.force_plot(explainer_rf_fraud.expected_value[1], shap_values_rf_fraud[1][sample_idx_fraud,:], shap_sample_fraud.iloc[sample_idx_fraud,:])
    plt.show()

    print("\nInterpreting Random Forest Model for Creditcard.csv...")

    explainer_rf_creditcard = shap.TreeExplainer(rf_creditcard)
    shap_sample_creditcard = X_creditcard_test.sample(n=min(1000, X_creditcard_test.shape[0]), random_state=42)
    shap_values_rf_creditcard = explainer_rf_creditcard.shap_values(shap_sample_creditcard)

    print("\nGenerating SHAP Summary Plot for Creditcard.csv...")

    shap.summary_plot(shap_values_rf_creditcard[1], shap_sample_creditcard, plot_type="bar", show=False)
    plt.title('SHAP Global Feature Importance (Creditcard.csv)')
    plt.tight_layout()
    plt.show()
    shap.summary_plot(shap_values_rf_creditcard[1], shap_sample_creditcard, show=False)
    plt.title('SHAP Summary Plot (Feature Impact and Direction) for Creditcard.csv')
    plt.tight_layout()
    plt.show()

    print("\nGenerating SHAP Force Plot for a sample prediction (Creditcard.csv)...")
    sample_idx_creditcard = 0
    shap.initjs()
    shap.force_plot(explainer_rf_creditcard.expected_value[1], shap_values_rf_creditcard[1][sample_idx_creditcard,:], shap_sample_creditcard.iloc[sample_idx_creditcard,:])
    plt.show()

def main():
    fraud_data, creditcard_data = load_data()
    
    (X_fraud_train_res, y_fraud_train_res, X_fraud_test, y_fraud_test,
     X_creditcard_train_res, y_creditcard_train_res, X_creditcard_test, y_creditcard_test) = preprocess_data(fraud_data, creditcard_data)
    rf_fraud, rf_creditcard = train_models(X_fraud_train_res, y_fraud_train_res, X_creditcard_train_res, y_creditcard_train_res)
   
    explain_with_shap(rf_fraud, X_fraud_test, rf_creditcard, X_creditcard_test)

if __name__ == "__main__":
    main()