import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib
import numpy as np
import os

# ========== STEP 1: LOAD MODEL ==========
try:
    model = joblib.load("battery_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ========== STEP 2: DEFINE APP ==========
app = FastAPI(title="Battery Capacity Predictor")

# ========== STEP 3: INPUT SCHEMA ==========
class BatteryInput(BaseModel):
    Cycle: float = Field(..., gt=0, description="Battery cycle count, must be positive")
    Current_A: float = Field(..., gt=0, description="Current in Amperes, must be positive")
    Voltage_V: float = Field(..., gt=0, description="Voltage in Volts, must be positive")
    Temperature_C: float = Field(..., description="Temperature in Celsius")

    class Config:
        schema_extra = {
            "example": {
                "Cycle": 50,
                "Current_A": 1.2,
                "Voltage_V": 3.7,
                "Temperature_C": 25
            }
        }

# ========== STEP 4: PREDICTION ENDPOINT ==========
@app.post("/predict")
def predict_battery(input: BatteryInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        data = pd.DataFrame([{
            'Cycle': input.Cycle,
            'Cycle_squared': input.Cycle ** 2,
            'Cycle_sqrt': np.sqrt(input.Cycle),
            'Current_A': input.Current_A,
            'Voltage_V': input.Voltage_V,
            'Temperature_C': input.Temperature_C,
            'Voltage_Current': input.Voltage_V * input.Current_A,
            'Temp_Current': input.Temperature_C * input.Current_A
        }])

        pred = model.predict(data)[0]
        return {"Predicted_Capacity_mAh": round(float(pred), 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

# ========== STEP 5: HEALTH CHECK ==========
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ========== STEP 6: RUN APP ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's PORT env variable
    uvicorn.run(app, host="0.0.0.0", port=port)
