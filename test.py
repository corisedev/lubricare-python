import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the test dataset
test_data_path = 'oiltypes.csv'

# Load the trained models
brand_model_path = 'brand_model.pkl'
brand_model = joblib.load(brand_model_path)

oil_model_path = 'oil_model.pkl'
oil_model = joblib.load(oil_model_path)

# Function to predict brand based on user input
def predict_brand(engine_cc, min_milage, max_milage):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'Engine CC': [engine_cc],
        'minMilage': [min_milage],
        'maxMilage': [max_milage]
    })
    
    # Make prediction with the trained model
    prediction = brand_model.predict(input_data)
    return prediction[0]

# Function to predict oil based on user input
def predict_oil(engine_cc, min_milage, max_milage):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'Engine CC': [engine_cc],
        'minMilage': [min_milage],
        'maxMilage': [max_milage]
    })
    
    # Make prediction with the trained model
    prediction = oil_model.predict(input_data)
    return prediction[0]

# Function to find the closest match in the dataset
def find_closest_match(engine_cc, min_milage, max_milage):
    # Load the dataset
    data = pd.read_csv(test_data_path)

    # Find the closest match based on input values
    closest_match = data.loc[
        (data['Engine CC'] == engine_cc) &
        (data['minMilage'] == min_milage) &
        (data['maxMilage'] == max_milage),
        'Brand Names'
    ]
    
    # If there's a match, return the brand; otherwise, return a default message
    if not closest_match.empty:
        return closest_match.iloc[0]  # Return the first match if multiple found
    else:
        return "No exact match found in the dataset."

# Input values
engine_cc = float(input("Enter Engine CC: "))
min_milage = float(input("Enter Minimum Mileage: "))
max_milage = float(input("Enter Maximum Mileage: "))

# Predict the brand
predicted_brand = predict_brand(engine_cc, min_milage, max_milage)

# Predict the oil
predicted_oil = predict_oil(engine_cc, min_milage, max_milage)

# Display the results
print(f"The predicted brand is: {predicted_brand}")
print(f"The predicted oil is: {predicted_oil}")

# Preprocess the test data
data = pd.read_csv(test_data_path)
X_test = data[['Engine CC', 'minMilage', 'maxMilage']]
y_brand_test = data['Brand Names']

# Make predictions with the trained model
y_brand_pred = brand_model.predict(X_test)


