# 🌸 Iris Flower Classifier API with FastAPI

This project builds and deploys a machine learning model to classify Iris flowers using FastAPI.

## 🚀 How It Works

- Trains a Random Forest model on the Iris dataset (`train_model.py`)
- Deploys it using a FastAPI web server (`main.py`)
- Exposes an API to predict flower species from measurements

## 🧪 Sample Prediction Input

POST `/predict`:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
