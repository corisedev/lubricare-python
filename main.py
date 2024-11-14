from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Define paths to the models
brand_model_path = 'brand_model.pkl'
oil_model_path = 'oil_model.pkl'

# Load the trained models
brand_model = joblib.load(brand_model_path)
oil_model = joblib.load(oil_model_path)

# Define a Pydantic model for the input data
class InputData(BaseModel):
    engine_cc: float
    min_milage: float
    max_milage: float

# Define a Pydantic model for the response data
class PredictionResponse(BaseModel):
    predicted_brand: str
    predicted_oil: str

# Load the dataset for finding closest match
test_data_path = 'oiltypes.csv'
data = pd.read_csv(test_data_path)

def find_closest_match(engine_cc, min_milage, max_milage):
    closest_match = data.loc[
        (data['Engine CC'] == engine_cc) &
        (data['minMilage'] == min_milage) &
        (data['maxMilage'] == max_milage),
        'Brand Names'
    ]
    return closest_match.iloc[0] if not closest_match.empty else "No exact match found in the dataset."

@app.post("/", response_model=PredictionResponse)
def predict(input_data: InputData):
    # Predict brand
    input_df = pd.DataFrame({
        'Engine CC': [input_data.engine_cc],
        'minMilage': [input_data.min_milage],
        'maxMilage': [input_data.max_milage]
    })

    predicted_brand = brand_model.predict(input_df)[0]
    predicted_oil = oil_model.predict(input_df)[0]

    # Find the closest match
    closest_match = find_closest_match(input_data.engine_cc, input_data.min_milage, input_data.max_milage)

    return PredictionResponse(predicted_brand=predicted_brand, predicted_oil=predicted_oil)

# Run the FastAPI app with Uvicorn
# Use the following command to run the server:
# 
