import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

# 1. Load dataset

data = pd.read_csv("dataset.csv")
print("Dataset preview:")
print(data.head())

# 2. Split features and target

X = data[['Size', 'Bedrooms', 'Age']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model 
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate model
predictions = model.predict(X_test)
print("\nPredicted price:",predictions)
print("Actual prices:", list(y_test))

'''# 5. Visualize results
plt.scatter(y_test, predictions, color='blue')
plt.xlabel("Actual Price")
plt.ylabel("predicted Prices")
plt.title("House Price Prediction")
plt.show()'''

'''# 6. Try a Custom prediction
new_house = [[2500, 4, 10]] #size=2500, Bedrooms= 4, Age=10
predicted_price = model.predict(new_house)
print("\nPredicted price for new house:", predicted_price[0])'''


# 5. Loop for multiple user inputs
print("\n--- Custom Prediction Loop ---")
while True:
    try:
        size = float(input("Enter house size (sq ft): "))
        bedrooms = int(input("Enter number of bedrooms: "))
        age = int(input("Enter age of house (years): "))

        # Pass input as DataFrame to avoid warning
        new_house = pd.DataFrame([[size, bedrooms, age]], columns=['Size', 'Bedrooms', 'Age'])
        predicted_price = model.predict(new_house)
        print(f"Predicted price for your house: {predicted_price[0]:.0f}")

        # Show graph AFTER prediction
        plt.scatter(y_test, predictions, color='blue', label="Test Data")
        plt.scatter(predicted_price, predicted_price, color='red', label="Your Prediction")
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("House Price Prediction")
        plt.legend()
        plt.show()

        # Ask if user wants to continue
        choice = input("Do you want to predict another house? (y/n): ").lower()
        if choice != 'y':
            print("Exiting prediction loop. Goodbye!")
            break
    except Exception as e:
        print("Error:", e)
        break