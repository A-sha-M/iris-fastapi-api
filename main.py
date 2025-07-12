from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.datasets import load_iris

app = FastAPI(title="🌸 Iris Flower Classifier API")

# Load model and label names
model = joblib.load("iris_model.pkl")
iris_data = load_iris()

# Input schema
class FlowerInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Welcome to the Iris Flower Classifier API!"}

@app.post("/predict")
def predict_species(data: FlowerInput):
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(input_data)
    species = iris_data.target_names[prediction[0]]
    return {"predicted_species": species}
