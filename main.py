from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# ========== STEP 1: LOAD MODEL ==========
model = joblib.load("battery_model.pkl")

# ========== STEP 2: DEFINE APP ==========
app = FastAPI(title="Battery Capacity Predictor")

# ========== STEP 3: INPUT SCHEMA ==========
class BatteryInput(BaseModel):
    Cycle: float
    Current_A: float
    Voltage_V: float
    Temperature_C: float

# ========== STEP 4: PREDICTION ENDPOINT ==========
@app.post("/predict")
def predict_battery(input: BatteryInput):
    # Create features
    Cycle_squared = input.Cycle ** 2
    Cycle_sqrt = np.sqrt(input.Cycle)
    Voltage_Current = input.Voltage_V * input.Current_A
    Temp_Current = input.Temperature_C * input.Current_A

    data = pd.DataFrame([{
        'Cycle': input.Cycle,
        'Cycle_squared': Cycle_squared,
        'Cycle_sqrt': Cycle_sqrt,
        'Current_A': input.Current_A,
        'Voltage_V': input.Voltage_V,
        'Temperature_C': input.Temperature_C,
        'Voltage_Current': Voltage_Current,
        'Temp_Current': Temp_Current
    }])

    # Predict
    pred = model.predict(data)[0]
    return {"Predicted_Capacity_mAh": round(pred, 2)}
