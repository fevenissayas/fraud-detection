import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    fraud_data = pd.read_csv('Fraud_Data.csv')
    ip_to_country = pd.read_csv('IpAddress_to_Country.csv')
    creditcard_data = pd.read_csv('creditcard.csv')

    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    
    fraud_data['ip_address_int'] = fraud_data['ip_address'].round().astype(int)

    ip_to_country['lower_bound_ip_address'] = ip_to_country['lower_bound_ip_address'].astype(int)
    ip_to_country['upper_bound_ip_address'] = ip_to_country['upper_bound_ip_address'].astype(int)

    creditcard_data.drop_duplicates(inplace=True)
    
    print("Datasets loaded and preprocessed for EDA.")
except FileNotFoundError as e:
    print(f"Error loading file for EDA: {e}. Please ensure the CSV files are in the same directory.")
    exit()


print("\n--- Starting Exploratory Data Analysis (EDA) ---")

# --- EDA for Fraud_Data.csv ---
print("\n--- EDA for E-commerce Fraud Data (Fraud_Data.csv) ---")

# Univariate Analysis
print("\n1. Univariate Analysis for Fraud_Data.csv:")
print("\nDescriptive Statistics for Numerical Features:")
print(fraud_data[['purchase_value', 'age']].describe())

# Distribution of 'purchase_value'
plt.figure(figsize=(10, 6))
sns.histplot(fraud_data['purchase_value'], bins=50, kde=True)
plt.title('Distribution of Purchase Value (Fraud_Data.csv)')
plt.xlabel('Purchase Value')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Distribution of 'age'
plt.figure(figsize=(10, 6))
sns.histplot(fraud_data['age'], bins=30, kde=True)
plt.title('Distribution of Age (Fraud_Data.csv)')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Count plots for Categorical Features
categorical_cols_fraud = ['source', 'browser', 'sex']
for col in categorical_cols_fraud:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=fraud_data, x=col, palette='viridis')
    plt.title(f'Distribution of {col} (Fraud_Data.csv)')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

# Class Imbalance Check for Fraud_Data.csv
plt.figure(figsize=(7, 5))
sns.countplot(data=fraud_data, x='class', palette='coolwarm')
plt.title('Class Distribution (Fraud_Data.csv)')
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Fraud (0)', 'Fraud (1)'])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
print("\nClass distribution for Fraud_Data.csv:\n", fraud_data['class'].value_counts(normalize=True))


# Bivariate Analysis for Fraud_Data.csv
print("\n2. Bivariate Analysis for Fraud_Data.csv:")

# Purchase Value vs. Class
plt.figure(figsize=(10, 6))
sns.boxplot(data=fraud_data, x='class', y='purchase_value', palette='pastel')
plt.title('Purchase Value vs. Class (Fraud_Data.csv)')
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
plt.ylabel('Purchase Value')
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Age vs. Class
plt.figure(figsize=(10, 6))
sns.boxplot(data=fraud_data, x='class', y='age', palette='pastel')
plt.title('Age vs. Class (Fraud_Data.csv)')
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
plt.ylabel('Age')
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Categorical features vs. Class (using crosstabs and stacked bar plots)
for col in categorical_cols_fraud:
    plt.figure(figsize=(10, 6))
    cross_tab = pd.crosstab(fraud_data[col], fraud_data['class'], normalize='index')
    cross_tab.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title(f'Fraud Rate by {col} (Fraud_Data.csv)')
    plt.xlabel(col)
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.legend(title='Class', labels=['Non-Fraud', 'Fraud'])
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()
    print(f"\nCrosstab for {col} vs. Class:\n", pd.crosstab(fraud_data[col], fraud_data['class']))


# --- EDA for creditcard.csv ---
print("\n--- EDA for Bank Transaction Fraud Data (creditcard.csv) ---")

# Univariate Analysis
print("\n1. Univariate Analysis for creditcard.csv:")
print("\nDescriptive Statistics for Numerical Features (Time, Amount):")
print(creditcard_data[['Time', 'Amount']].describe())

# Distribution of 'Time'
plt.figure(figsize=(12, 6))
sns.histplot(creditcard_data['Time'], bins=100, kde=False)
plt.title('Distribution of Time (creditcard.csv)')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Distribution of 'Amount'
plt.figure(figsize=(12, 6))
sns.histplot(creditcard_data['Amount'], bins=100, kde=True)
plt.title('Distribution of Amount (creditcard.csv)')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Class Imbalance Check for creditcard.csv
plt.figure(figsize=(7, 5))
sns.countplot(data=creditcard_data, x='Class', palette='coolwarm')
plt.title('Class Distribution (creditcard.csv)')
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Fraud (0)', 'Fraud (1)'])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
print("\nClass distribution for creditcard.csv:\n", creditcard_data['Class'].value_counts(normalize=True))


# Bivariate Analysis for creditcard.csv
print("\n2. Bivariate Analysis for creditcard.csv:")

# Amount vs. Class
plt.figure(figsize=(10, 6))
sns.boxplot(data=creditcard_data, x='Class', y='Amount', palette='pastel')
plt.title('Transaction Amount vs. Class (creditcard.csv)')
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
plt.ylabel('Amount')
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Time vs. Class (using KDE plot for density comparison)
plt.figure(figsize=(12, 6))
sns.kdeplot(data=creditcard_data, x='Time', hue='Class', fill=True, common_norm=False, palette='viridis')
plt.title('Time Distribution by Class (creditcard.csv)')
plt.xlabel('Time (seconds)')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

correlation_matrix = creditcard_data.drop(columns=['Time', 'Class']).corr()
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of V Features and Amount (creditcard.csv)')
plt.show()
