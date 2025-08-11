import pandas as pd
import numpy as np

print("Loading datasets...")
try:
    fraud_data = pd.read_csv('Fraud_Data.csv')
    ip_to_country = pd.read_csv('IpAddress_to_Country.csv')
    creditcard_data = pd.read_csv('creditcard.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure the CSV files are in the same directory as this script.")
    exit()

print("\n--- Initial Data Info ---")
print("\nFraud_Data.csv Info:")
fraud_data.info()
print("\nIpAddress_to_Country.csv Info:")
ip_to_country.info()
print("\ncreditcard.csv Info:")
creditcard_data.info()

print("\n--- Handling Missing Values ---")

print("\nProcessing Fraud_Data.csv for missing values...")
print("Missing values before handling:\n", fraud_data.isnull().sum())


initial_rows_fraud = fraud_data.shape[0]
fraud_data.dropna(subset=['ip_address', 'device_id'], inplace=True)
print(f"Dropped {initial_rows_fraud - fraud_data.shape[0]} rows from Fraud_Data due to missing ip_address or device_id.")

# Impute 'sex' (categorical) with mode
if fraud_data['sex'].isnull().any():
    mode_sex = fraud_data['sex'].mode()[0]
    fraud_data['sex'].fillna(mode_sex, inplace=True)
    print(f"Imputed missing 'sex' values with mode: {mode_sex}")

# Impute 'age' (numerical) with median
if fraud_data['age'].isnull().any():
    median_age = fraud_data['age'].median()
    fraud_data['age'].fillna(median_age, inplace=True)
    print(f"Imputed missing 'age' values with median: {median_age}")

print("Missing values after handling for Fraud_Data.csv:\n", fraud_data.isnull().sum())


print("\nProcessing IpAddress_to_Country.csv for missing values...")
print("Missing values before handling:\n", ip_to_country.isnull().sum())
print("Missing values after handling for IpAddress_to_Country.csv:\n", ip_to_country.isnull().sum())


print("\nProcessing creditcard.csv for missing values...")
print("Missing values before handling:\n", creditcard_data.isnull().sum())
print("Missing values after handling for creditcard.csv:\n", creditcard_data.isnull().sum())


print("\n--- Removing Duplicate Rows ---")
initial_rows_fraud = fraud_data.shape[0]
fraud_data.drop_duplicates(inplace=True)
print(f"Removed {initial_rows_fraud - fraud_data.shape[0]} duplicate rows from Fraud_Data.csv.")

initial_rows_ip = ip_to_country.shape[0]
ip_to_country.drop_duplicates(inplace=True)
print(f"Removed {initial_rows_ip - ip_to_country.shape[0]} duplicate rows from IpAddress_to_Country.csv.")

# creditcard.csv
initial_rows_credit = creditcard_data.shape[0]
creditcard_data.drop_duplicates(inplace=True)
print(f"Removed {initial_rows_credit - creditcard_data.shape[0]} duplicate rows from creditcard.csv.")


# --- 4. Data Cleaning - Correct Data Types ---
print("\n--- Correcting Data Types ---")

# Fraud_Data.csv
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
print("Converted 'signup_time' and 'purchase_time' in Fraud_Data.csv to datetime.")

fraud_data['ip_address_int'] = fraud_data['ip_address'].round().astype(int)
print("Converted 'ip_address' in Fraud_Data.csv to integer format ('ip_address_int').")

# IpAddress_to_Country.csv
ip_to_country['lower_bound_ip_address'] = ip_to_country['lower_bound_ip_address'].astype(int)
ip_to_country['upper_bound_ip_address'] = ip_to_country['upper_bound_ip_address'].astype(int)

print("\n--- Final Data Info After Preprocessing Steps ---")
fraud_data.info()
ip_to_country.info()
creditcard_data.info()

print(fraud_data.head())
print(ip_to_country.head())
print(creditcard_data.head())
