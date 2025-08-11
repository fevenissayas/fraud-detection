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
    classification_report, roc_auc_score
)
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning) 

try:
    fraud_data = pd.read_csv('Fraud_Data_merged.csv')
    creditcard_data = pd.read_csv('creditcard_cleaned.csv')

    if 'signup_time' in fraud_data.columns and not pd.api.types.is_datetime64_any_dtype(fraud_data['signup_time']):
        fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'], errors='coerce')
    if 'purchase_time' in fraud_data.columns and not pd.api.types.is_datetime64_any_dtype(fraud_data['purchase_time']):
        fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'], errors='coerce')

    fraud_data.dropna(subset=['signup_time', 'purchase_time'], inplace=True)
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure the specified CSV files are in the same directory as this script.")
    exit()

X_fraud = fraud_data.drop(columns=[
    'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'class'
])
y_fraud = fraud_data['class']

X_creditcard = creditcard_data.drop(columns=['Time', 'Class'])
y_creditcard = creditcard_data['Class']

X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(
    X_fraud, y_fraud, test_size=0.3, random_state=42, stratify=y_fraud
)
X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test = train_test_split(
    X_creditcard, y_creditcard, test_size=0.3, random_state=42, stratify=y_creditcard
)

numerical_cols_fraud_train = X_fraud_train.select_dtypes(include=np.number).columns
categorical_cols_fraud_train = X_fraud_train.select_dtypes(include='object').columns

encoder_fraud_smote = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_fraud_train_cat_encoded = encoder_fraud_smote.fit_transform(X_fraud_train[categorical_cols_fraud_train])
X_fraud_train_cat_encoded_df = pd.DataFrame(X_fraud_train_cat_encoded, columns=encoder_fraud_smote.get_feature_names_out(categorical_cols_fraud_train), index=X_fraud_train.index)
X_fraud_train_combined = pd.concat([X_fraud_train[numerical_cols_fraud_train], X_fraud_train_cat_encoded_df], axis=1)

smote = SMOTE(random_state=42)
X_fraud_train_res, y_fraud_train_res = smote.fit_resample(X_fraud_train_combined, y_fraud_train)

rus = RandomUnderSampler(random_state=42)
X_creditcard_train_res, y_creditcard_train_res = rus.fit_resample(X_creditcard_train, y_creditcard_train)

scaler_fraud = StandardScaler()
numerical_cols_fraud_resampled = X_fraud_train_res.columns
X_fraud_train_res[numerical_cols_fraud_resampled] = scaler_fraud.fit_transform(X_fraud_train_res[numerical_cols_fraud_resampled])

X_fraud_test_cat_encoded = encoder_fraud_smote.transform(X_fraud_test[categorical_cols_fraud_train])
X_fraud_test_cat_encoded_df = pd.DataFrame(X_fraud_test_cat_encoded, columns=encoder_fraud_smote.get_feature_names_out(categorical_cols_fraud_train), index=X_fraud_test.index)
X_fraud_test_combined = pd.concat([X_fraud_test[numerical_cols_fraud_train], X_fraud_test_cat_encoded_df], axis=1)
X_fraud_test = scaler_fraud.transform(X_fraud_test_combined)
X_fraud_test = pd.DataFrame(X_fraud_test, columns=X_fraud_train_res.columns, index=X_fraud_test_combined.index)

scaler_creditcard = StandardScaler()
numerical_cols_creditcard = X_creditcard_train_res.select_dtypes(include=np.number).columns
X_creditcard_train_res[numerical_cols_creditcard] = scaler_creditcard.fit_transform(X_creditcard_train_res[numerical_cols_creditcard])
X_creditcard_test[numerical_cols_creditcard] = scaler_creditcard.transform(X_creditcard_test[numerical_cols_creditcard])

lr_fraud = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
lr_creditcard = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
rf_fraud = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
rf_creditcard = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')

lr_fraud.fit(X_fraud_train_res, y_fraud_train_res)
y_pred_lr_fraud = lr_fraud.predict(X_fraud_test)
y_prob_lr_fraud = lr_fraud.predict_proba(X_fraud_test)[:, 1]

print("Fraud_Data Logistic Regression:")
print("Confusion Matrix:\n", confusion_matrix(y_fraud_test, y_pred_lr_fraud))
print("Classification Report:\n", classification_report(y_fraud_test, y_pred_lr_fraud))
precision_lr_fraud, recall_lr_fraud, _ = precision_recall_curve(y_fraud_test, y_prob_lr_fraud)
auc_pr_lr_fraud = auc(recall_lr_fraud, precision_lr_fraud)
print(f"AUC-PR: {auc_pr_lr_fraud:.4f}")
print(f"F1-Score: {f1_score(y_fraud_test, y_pred_lr_fraud):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_fraud_test, y_prob_lr_fraud):.4f}")

rf_fraud.fit(X_fraud_train_res, y_fraud_train_res)
y_pred_rf_fraud = rf_fraud.predict(X_fraud_test)
y_prob_rf_fraud = rf_fraud.predict_proba(X_fraud_test)[:, 1]

print("Fraud_Data Random Forest:")
print("Confusion Matrix:\n", confusion_matrix(y_fraud_test, y_pred_rf_fraud))
print("Classification Report:\n", classification_report(y_fraud_test, y_pred_rf_fraud))
precision_rf_fraud, recall_rf_fraud, _ = precision_recall_curve(y_fraud_test, y_prob_rf_fraud)
auc_pr_rf_fraud = auc(recall_rf_fraud, precision_rf_fraud)
print(f"AUC-PR: {auc_pr_rf_fraud:.4f}")
print(f"F1-Score: {f1_score(y_fraud_test, y_pred_rf_fraud):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_fraud_test, y_prob_rf_fraud):.4f}")

lr_creditcard.fit(X_creditcard_train_res, y_creditcard_train_res)
y_pred_lr_creditcard = lr_creditcard.predict(X_creditcard_test)
y_prob_lr_creditcard = lr_creditcard.predict_proba(X_creditcard_test)[:, 1]

print("Creditcard_Data Logistic Regression:")
print("Confusion Matrix:\n", confusion_matrix(y_creditcard_test, y_pred_lr_creditcard))
print("Classification Report:\n", classification_report(y_creditcard_test, y_pred_lr_creditcard))
precision_lr_creditcard, recall_lr_creditcard, _ = precision_recall_curve(y_creditcard_test, y_prob_lr_creditcard)
auc_pr_lr_creditcard = auc(recall_lr_creditcard, precision_lr_creditcard)
print(f"AUC-PR: {auc_pr_lr_creditcard:.4f}")
print(f"F1-Score: {f1_score(y_creditcard_test, y_pred_lr_creditcard):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_creditcard_test, y_prob_lr_creditcard):.4f}")

rf_creditcard.fit(X_creditcard_train_res, y_creditcard_train_res)
y_pred_rf_creditcard = rf_creditcard.predict(X_creditcard_test)
y_prob_rf_creditcard = rf_creditcard.predict_proba(X_creditcard_test)[:, 1]

print("Creditcard_Data Random Forest:")
print("Confusion Matrix:\n", confusion_matrix(y_creditcard_test, y_pred_rf_creditcard))
print("Classification Report:\n", classification_report(y_creditcard_test, y_pred_rf_creditcard))
precision_rf_creditcard, recall_rf_creditcard, _ = precision_recall_curve(y_creditcard_test, y_prob_rf_creditcard)
auc_pr_rf_creditcard = auc(recall_rf_creditcard, precision_rf_creditcard)
print(f"AUC-PR: {auc_pr_rf_creditcard:.4f}")
print(f"F1-Score: {f1_score(y_creditcard_test, y_pred_rf_creditcard):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_creditcard_test, y_prob_rf_creditcard):.4f}")
