from fastapi import FastAPI
import pandas as pd
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Sample data: mileage vs price
data = {
    'mileage': [15, 30, 45, 60, 75],
    'price': [550000, 480000, 430000, 390000, 350000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[['mileage']]
y = df['price']

# Train Linear Regression model (once, when app starts)
model = LinearRegression()
model.fit(X, y)


@app.get("/")
async def root():
    return {"message": "Linear Regression Model is Ready!"}


@app.get("/predict/{mileage}")
async def predict_price(mileage: int):
    """Predict car price based on mileage"""
    predicted_price = model.predict([[mileage]])[0]
    return {
        "mileage": mileage,
        "predicted_price": round(predicted_price, 2)
    }


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
#ddsd