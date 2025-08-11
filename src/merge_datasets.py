import pandas as pd
import numpy as np


try:
    fraud_data = pd.read_csv('../data/Fraud_Data_cleaned.csv')
    ip_to_country = pd.read_csv('../data/IpAddress_to_Country_cleaned.csv')
    creditcard_data = pd.read_csv('../data/creditcard_cleaned.csv')
    #to be sure :AGAIN
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    
    fraud_data['ip_address_int'] = fraud_data['ip_address'].round().astype(int)

    ip_to_country['lower_bound_ip_address'] = ip_to_country['lower_bound_ip_address'].astype(int)
    ip_to_country['upper_bound_ip_address'] = ip_to_country['upper_bound_ip_address'].astype(int)

    creditcard_data.drop_duplicates(inplace=True)
    
    print("Datasets loaded for EDA.")
except FileNotFoundError as e:
    print(f"Error loading file for EDA: {e}. Please ensure the CSV files are in the data directory.")
    exit()


print("\n--- Starting Merging Datasets for Geolocation Analysis ---")

ip_to_country_sorted = ip_to_country.sort_values(by='lower_bound_ip_address').reset_index(drop=True)


def get_country(ip_int, ip_ranges_df):

    idx = ip_ranges_df['lower_bound_ip_address'].searchsorted(ip_int, side='right') - 1

    if idx >= 0 and ip_int <= ip_ranges_df.loc[idx, 'upper_bound_ip_address']:
        return ip_ranges_df.loc[idx, 'country']
    else:
        return np.nan

print("Mapping IP addresses to countries. This may take a moment...")
fraud_data['country'] = fraud_data['ip_address_int'].apply(lambda x: get_country(x, ip_to_country_sorted))

unmapped_ips_count = fraud_data['country'].isnull().sum()
if unmapped_ips_count > 0:
    print(f"\nWarning: {unmapped_ips_count} IP addresses could not be mapped to a country.")
else:
    print("\nAll IP addresses successfully mapped to a country.")


print("\n--- Merged Fraud Data Info ---")
fraud_data.info()
print(fraud_data.head())

