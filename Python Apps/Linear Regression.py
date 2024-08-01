import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'D:/projects/Big-Data/Datasets/egypt_House_prices.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Initial Data:")
print(data.head())

# Display the column names
print("\nColumn Names:")
print(data.columns)

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Display basic statistics
data_description = data.describe(include='all')
print("\nData Description:")
print(data_description)

# Drop rows with missing values
data_cleaned = data.dropna()

# Identify numerical columns
numerical_columns = data_cleaned.select_dtypes(include=[np.number]).columns
print("\nNumerical Columns:")
print(numerical_columns)

# Define a function to identify outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    filter = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
    return df.loc[filter]

# Apply the function to relevant columns
for col in numerical_columns:
    data_cleaned = remove_outliers(data_cleaned, col)

# Display the cleaned data
print("\nCleaned Data Description:")
print(data_cleaned.describe())

# Define the target column
target_column = 'Price'

# Extract the target column before encoding
if target_column in data_cleaned.columns:
    y = data_cleaned[target_column]
    X = data_cleaned.drop(target_column, axis=1)
else:
    raise ValueError(f"Target column '{target_column}' not found in the data.")

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
print("\nCategorical Columns:")
print(categorical_columns)

# Apply one-hot encoding
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Add the target column back
data_encoded = X_encoded.copy()
data_encoded[target_column] = y

# Ensure the target column is numeric
data_encoded[target_column] = pd.to_numeric(data_encoded[target_column], errors='coerce')

# Drop rows where the target column could not be converted to numeric
data_encoded = data_encoded.dropna(subset=[target_column])

# Display the encoded data
print("\nEncoded Data:")
print(data_encoded.head())

# Print the columns to identify the correct target column
print("\nEncoded Data Columns:")
print(data_encoded.columns)

# Ensure the target column exists in the encoded data
if target_column not in data_encoded.columns:
    raise ValueError(f"Target column '{target_column}' not found in the encoded data.")
else:
    # Split the data into features and target
    X = data_encoded.drop(target_column, axis=1)
    y = data_encoded[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'\nMean Squared Error: {mse}')
    print(f'R-squared: {r2}')
