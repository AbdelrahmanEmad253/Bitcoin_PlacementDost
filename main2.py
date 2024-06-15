import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
#############################################################################################################################################################################
# Define the ticker symbol
tickerSymbol = 'BTC-USD'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
bitcoin_data = tickerData.history(period='1d', start='2015-01-01', end='2023-06-01')

# Save the data to a CSV file
bitcoin_data.to_csv('bitcoin_data.csv')

# Display the first few rows of the dataset
print(bitcoin_data.head())

# Plot the closing prices
plt.figure(figsize=(10, 6))
plt.plot(bitcoin_data['Close'], label='Bitcoin Price')
plt.title('Bitcoin Price (2015-2023)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
#############################################################################################################################################################################
# Load the data from the CSV file
bitcoin_data = pd.read_csv('bitcoin_data.csv', index_col='Date', parse_dates=True)

# Display the first few rows of the dataset
print(bitcoin_data.head())

# Check for missing values
missing_values = bitcoin_data.isnull().sum()
print('Missing values in each column:\n', missing_values)

# Drop rows with missing values
bitcoin_data = bitcoin_data.dropna()

# Verify that there are no missing values left
missing_values_after = bitcoin_data.isnull().sum()
print('Missing values after dropping:\n', missing_values_after)

# Create a 7-day moving average of the closing price
bitcoin_data['MA7'] = bitcoin_data['Close'].rolling(window=7).mean()

# Create a 30-day moving average of the closing price
bitcoin_data['MA30'] = bitcoin_data['Close'].rolling(window=30).mean()

# Display the first few rows to see the new features
print(bitcoin_data.head(10))

# Select the features for normalization
features_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA30']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply the scaler to the selected features
bitcoin_data[features_to_scale] = scaler.fit_transform(bitcoin_data[features_to_scale])

# Display the first few rows to see the scaled features
print(bitcoin_data.head(35))
#############################################################################################################################################################################
# Plot the closing price
plt.figure(figsize=(12, 6))
plt.plot(bitcoin_data.index, bitcoin_data['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Closing Price Over Time')
plt.legend()
plt.show()

# Plot the closing price and moving averages
plt.figure(figsize=(12, 6))
plt.plot(bitcoin_data.index, bitcoin_data['Close'], label='Close Price')
plt.plot(bitcoin_data.index, bitcoin_data['MA7'], label='7-Day Moving Average')
plt.plot(bitcoin_data.index, bitcoin_data['MA30'], label='30-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Closing Price and Moving Averages')
plt.legend()
plt.show()

# Calculate the correlation matrix
corr_matrix = bitcoin_data.corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()
#############################################################################################################################################################################
# Define features and target
features = ['Open', 'High', 'Low', 'Volume', 'MA7', 'MA30']
target = 'Close'

X = bitcoin_data[features]
y = bitcoin_data[target]

# Initialize the SimpleImputer with different strategies for different columns
imputer = SimpleImputer(strategy='mean')

# Define which columns to impute with mean and which with zeros
mean_impute_columns = ['Open', 'High', 'Low', 'Volume']
zero_impute_columns = ['MA7', 'MA30']

# Impute missing values with mean for mean_impute_columns
X_mean_imputed = imputer.fit_transform(X[mean_impute_columns])

# Impute missing values with zeros for zero_impute_columns
X_zero_imputed = np.nan_to_num(X[zero_impute_columns], nan=0)

# Concatenate the imputed columns
X_imputed = np.concatenate((X_mean_imputed, X_zero_imputed), axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the HistGradientBoostingRegressor model
model = HistGradientBoostingRegressor()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate MAE, MSE, and RMSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
#############################################################################################################################################################################
# Define the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_iter': [100, 200, 300],
    'max_depth': [3, 5, 7]
}

# Initialize the model
model = HistGradientBoostingRegressor()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Fit GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test data using the best model
y_pred_best = best_model.predict(X_test)

# Calculate MAE, MSE, and RMSE for the best model
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)

print(f'Best Model Mean Absolute Error (MAE): {mae_best}')
print(f'Best Model Mean Squared Error (MSE): {mse_best}')
print(f'Best Model Root Mean Squared Error (RMSE): {rmse_best}')
#############################################################################################################################################################################
# Load the dataset
data = pd.read_csv('bitcoin_data.csv')

# Print column names to verify
print(data.columns)

# Convert date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Extract useful date features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek

# Drop the original date column
data = data.drop(columns=['Date'])

# Use 'Close' as the target variable
X = data.drop(columns=['Close'])
y = data['Close']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the best model
best_model = HistGradientBoostingRegressor(learning_rate=0.1, max_depth=7, max_iter=300)

# Train the model
best_model.fit(X_train, y_train)

# Make predictions
predictions = best_model.predict(X_test)

# Save predictions to a CSV file
predictions_df = pd.DataFrame(predictions, columns=['Predicted_Close'])
predictions_df.to_csv('predictions.csv', index=False)