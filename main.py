from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model
model = joblib.load("random_forest_model.pkl")  # Change path if needed

# Define FastAPI app
app = FastAPI(title="Weekly Sales Predictor API")

# Define input schema
class SalesFeatures(BaseModel):
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float
    week: int
    Month: int
    Year: int
    Weekly_Sales_Lag_1: float
    Weekly_Sales_Lag_2: float
    Weekly_Sales_Lag_3: float
    Weekly_Sales_Rolling_Mean: float
    Weekly_Sales_Rolling_Std: float
    Weekly_Sales_Cumulative_Sum: float

@app.post("/predict")
def predict_sales(data: SalesFeatures):
    # Convert to array for model
    features = np.array([[ 
        data.Temperature, data.Fuel_Price, data.CPI, data.Unemployment,
        data.week, data.Month, data.Year,
        data.Weekly_Sales_Lag_1, data.Weekly_Sales_Lag_2, data.Weekly_Sales_Lag_3,
        data.Weekly_Sales_Rolling_Mean, data.Weekly_Sales_Rolling_Std, data.Weekly_Sales_Cumulative_Sum
    ]])

    prediction = model.predict(features)[0]
    return {"Predicted_Weekly_Sales": round(prediction, 2)}
