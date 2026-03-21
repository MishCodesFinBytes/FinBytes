# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.linear_model import LinearRegression  # For linear regression model
from sklearn.model_selection import train_test_split  # To split dataset into training and test sets
from sklearn.metrics import mean_squared_error, r2_score  # For model evaluation metrics

# Load the dataset (replace "AAPL.csv" with your actual file path)
df = pd.read_csv("C:\codebase\FinBytes\AAPL.csv")

# Step 1: Handle missing values by removing rows with null values
df = df.dropna()

# Step 2: Convert the "Date" column into a numerical format for analysis
# Convert Date from string to datetime and then to an integer (YYYYMMDD format)
df["Date"] = pd.to_datetime(df["Date"])
df["NumericDate"] = (df["Date"] - df["Date"].min()).dt.days  # Convert dates into days since start for numeric analysis

# Step 3: Prepare features (X) and target variable (y)
X = df[["NumericDate"]]  # Use the numeric date as the feature
y = df["Close"]  # Target variable (stock closing price)

# Step 4: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Step 6: Predict stock prices on the test set
y_pred = lr.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # Coefficient of Determination (R²)

# Print evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Coefficient of Determination (R²):", r2)

# Step 8: Add predictions and visualize
# Combine test data and predictions into a DataFrame for better visualization
predictions_df = pd.DataFrame({
    "Date": df["Date"].iloc[X_test.index],  # Convert numeric dates back to original dates
    "Actual": y_test,
    "Predicted": y_pred
}).sort_values(by="Date")  # Sort by Date for better plotting

# Print the predicted values
print("\nPredicted Stock Prices:")
print(predictions_df)

# Step 9: Plot Actual vs Predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(predictions_df["Date"], predictions_df["Actual"], label="Actual Prices", marker='o')
plt.plot(predictions_df["Date"], predictions_df["Predicted"], label="Predicted Prices", marker='x', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Actual vs Predicted Stock Prices")
plt.legend()
plt.grid(True)
plt.show()
