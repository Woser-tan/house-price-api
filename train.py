import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load the Boston housing dataset
california = fetch_california_housing()
x = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'model.joblib')

print("Model trained and saved as 'model.joblib'")