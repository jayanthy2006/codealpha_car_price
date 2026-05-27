import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("car data.csv")

print("First 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

label_encoder = LabelEncoder()

categorical_columns = ['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission']

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])


X = df.drop('Selling_Price', axis=1)

y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("Mean Absolute Error :", mae)
print("Mean Squared Error  :", mse)
print("R2 Score            :", r2)


plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()

sample_data = [[5, 6.5, 35000, 0, 10, 1, 0, 1]]

predicted_price = model.predict(sample_data)

print("\nPredicted Car Price:", predicted_price[0])