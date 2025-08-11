X_fraud = fraud_data.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'class'])
y_fraud = fraud_data['class']

X_creditcard = creditcard_data.drop(columns=['Time', 'Class'])
y_creditcard = creditcard_data['Class']

X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(
    X_fraud, y_fraud, test_size=0.3, random_state=42, stratify=y_fraud
)
print(f"Fraud_Data train shape: {X_fraud_train.shape}, test shape: {X_fraud_test.shape}")

X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test = train_test_split(
    X_creditcard, y_creditcard, test_size=0.3, random_state=42, stratify=y_creditcard
)
print(f"Creditcard_Data train shape: {X_creditcard_train.shape}, test shape: {X_creditcard_test.shape}")


print(f"Original Fraud_Data training set shape: {Counter(y_fraud_train)}")
smote = SMOTE(random_state=42)
X_fraud_train_res, y_fraud_train_res = smote.fit_resample(X_fraud_train.select_dtypes(include=np.number), y_fraud_train)
print(f"Resampled Fraud_Data training set shape (SMOTE): {Counter(y_fraud_train_res)}")

print(f"Original Creditcard_Data training set shape: {Counter(y_creditcard_train)}")
rus = RandomUnderSampler(random_state=42)
X_creditcard_train_res, y_creditcard_train_res = rus.fit_resample(X_creditcard_train, y_creditcard_train)
print(f"Resampled Creditcard_Data training set shape (RandomUnderSampler): {Counter(y_creditcard_train_res)}")


#Performing Normalization and Scaling

numerical_cols_fraud = X_fraud_train_res.select_dtypes(include=np.number).columns
scaler_fraud = StandardScaler()
X_fraud_train_res[numerical_cols_fraud] = scaler_fraud.fit_transform(X_fraud_train_res[numerical_cols_fraud])
X_fraud_test[numerical_cols_fraud] = scaler_fraud.transform(X_fraud_test[numerical_cols_fraud])

numerical_cols_creditcard = X_creditcard_train_res.select_dtypes(include=np.number).columns
scaler_creditcard = StandardScaler()
X_creditcard_train_res[numerical_cols_creditcard] = scaler_creditcard.fit_transform(X_creditcard_train_res[numerical_cols_creditcard])
X_creditcard_test[numerical_cols_creditcard] = scaler_creditcard.transform(X_creditcard_test[numerical_cols_creditcard])


categorical_cols_fraud = X_fraud_train.select_dtypes(include='object').columns
encoder_fraud = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

encoded_fraud_train = encoder_fraud.fit_transform(X_fraud_train_res[categorical_cols_fraud])
encoded_fraud_train_df = pd.DataFrame(encoded_fraud_train, columns=encoder_fraud.get_feature_names_out(categorical_cols_fraud), index=X_fraud_train_res.index)

# Transform test data
encoded_fraud_test = encoder_fraud.transform(X_fraud_test[categorical_cols_fraud])
encoded_fraud_test_df = pd.DataFrame(encoded_fraud_test, columns=encoder_fraud.get_feature_names_out(categorical_cols_fraud), index=X_fraud_test.index)

# Drop original categorical columns and concatenate encoded ones
X_fraud_train_res = X_fraud_train_res.drop(columns=categorical_cols_fraud)
X_fraud_train_res = pd.concat([X_fraud_train_res, encoded_fraud_train_df], axis=1)

X_fraud_test = X_fraud_test.drop(columns=categorical_cols_fraud)
X_fraud_test = pd.concat([X_fraud_test, encoded_fraud_test_df], axis=1)

print(f"Fraud_Data X_train_res: {X_fraud_train_res.shape}, y_train_res: {y_fraud_train_res.shape}")
print(f"Fraud_Data X_test: {X_fraud_test.shape}, y_test: {y_fraud_test.shape}")
print(f"Creditcard_Data X_train_res: {X_creditcard_train_res.shape}, y_train_res: {y_creditcard_train_res.shape}")
print(f"Creditcard_Data X_test: {X_creditcard_test.shape}, y_test: {y_creditcard_test.shape}")

print(X_fraud_train_res.head())
print(X_creditcard_train_res.head())