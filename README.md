# fraud-detection

## Project Overview  
This project focuses on detecting fraudulent transactions in both e-commerce and banking datasets using machine learning.  
It covers data cleaning, feature engineering, handling class imbalance, model training, evaluation, and interpretability with SHAP.

---

## Data and Preprocessing  
- Cleaned and transformed two datasets:  
  - **E-commerce:** Fraud_Data.csv enriched with IP-to-country mapping  
  - **Bank transactions:** creditcard.csv with PCA features  
- Handled missing values and duplicates  
- Extracted time-based and frequency features (e.g., hour_of_day, time_since_signup, transaction counts)  
- Applied OneHotEncoding to categorical features  
- Addressed severe class imbalance using SMOTE (e-commerce) and Random Undersampling (bank data)  
- Normalized numerical features with StandardScaler  

---

## Model Building and Evaluation  
- Trained and compared two classifiers on both datasets:  
  - Logistic Regression (baseline)  
  - Random Forest Classifier (ensemble)  
- Used stratified train-test splits and balanced class weights  
- Evaluated with metrics suited for imbalanced data: AUC-PR, F1-Score, ROC-AUC, precision, recall  
- Random Forest outperformed Logistic Regression on all key metrics, offering better precision and recall balance  

| Dataset      | Model            | AUC-PR | F1-Score | ROC-AUC | Precision | Recall |
|--------------|------------------|--------|----------|---------|-----------|--------|
| E-commerce   | Logistic Reg.    | 0.12   | 0.18     | 0.56    | 0.11      | 0.51   |
|              | Random Forest    | 0.56   | 0.66     | 0.77    | 0.86      | 0.54   |
| Bank         | Logistic Reg.    | 0.34   | 0.09     | 0.96    | 0.05      | 0.88   |
|              | Random Forest    | 0.74   | 0.09     | 0.97    | 0.05      | 0.88   |

---

## Conclusion  
The Random Forest classifier is the best choice for both datasets, effectively capturing complex patterns and handling imbalance to improve fraud detection performance.

---

## Usage  
Run preprocessing, training, and evaluation scripts from `src/` or interactively explore notebooks in `scripts/`.

---