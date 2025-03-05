import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the training data
train_df = pd.read_csv('train.csv')

# Load the testing data
test_df = pd.read_csv('test.csv')
# Take a look at the first few rows of the data
print(train_df.head())

# Check the data types
print(train_df.info())

# Get a sense of the features and target variable
print(train_df.describe())
# Split the data into features (X) and target (y)
X_train = train_df.drop('target', axis=1, errors='ignore')
y_train = train_df.get('target')

if y_train is None:
    print("Target column not found.")
else:
    # Proceed with the rest of the code
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(train_df.columns)
X_train = train_df.drop('Target', axis=1)
y_train = train_df['Target']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Handle missing values
train_numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
test_numeric_cols = test_df.select_dtypes(include=['int64', 'float64']).columns

# Fill missing values in train_df
train_df[train_numeric_cols] = train_df[train_numeric_cols].fillna(train_df[train_numeric_cols].mean())

# Fill missing values in test_df
test_df[test_numeric_cols] = test_df[test_numeric_cols].fillna(test_df[test_numeric_cols].mean())

# Split the data into features (X) and target (y)
X_train = train_df.drop(['ID', 'Target'], axis=1)
y_train = train_df['Target']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Combine the training and test data to fit the OneHotEncoder
all_data = pd.concat([X_train, X_val, test_df.drop('ID', axis=1)])

# One-hot encode the categorical data
categorical_cols = all_data.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(all_data[categorical_cols])

# Transform the categorical data
X_train_categorical_encoded = encoder.transform(X_train[categorical_cols])
X_val_categorical_encoded = encoder.transform(X_val[categorical_cols])
test_df_categorical_encoded = encoder.transform(test_df.drop('ID', axis=1)[categorical_cols])

# Scale the numeric data
scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
X_train_numeric_scaled = scaler.fit_transform(X_train[numeric_cols])
X_val_numeric_scaled = scaler.transform(X_val[numeric_cols])
test_df_numeric_scaled = scaler.transform(test_df.drop('ID', axis=1)[numeric_cols])

# Combine the scaled numeric and encoded categorical data
X_train_scaled = pd.concat([pd.DataFrame(X_train_numeric_scaled), pd.DataFrame(X_train_categorical_encoded.toarray())], axis=1)
X_val_scaled = pd.concat([pd.DataFrame(X_val_numeric_scaled), pd.DataFrame(X_val_categorical_encoded.toarray())], axis=1)
test_df_scaled = pd.concat([pd.DataFrame(test_df_numeric_scaled), pd.DataFrame(test_df_categorical_encoded.toarray())], axis=1)

# Train a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_val = model.predict(X_val_scaled)
y_pred_test = model.predict(test_df_scaled)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred_val)
print(f'Mean Squared Error: {mse:.2f}')

# Save the predictions
submission_df = pd.DataFrame({'ID': test_df['ID'], 'Target': y_pred_test})
submission_df.to_csv('submission.csv', index=False)
