# """
# Application
# """

# import os

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import pandas as pd
# from loguru import logger

# from src.inference import load_model, predict


# logger.info("Loading model")
# MODEL_PATH = os.path.join("models", "model.joblib")
# MODEL = load_model(MODEL_PATH)
# logger.info("Model loaded")


# app = FastAPI()

# class IrisFeatures(BaseModel):
#     """Iris features"""
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float

# @app.get("/")
# def health_check() -> dict:
#     """Health check"""
#     return {"status": "okay"}


# @app.post("/predict")
# def make_prediction(features: IrisFeatures) -> dict:
#     """Make a prediction by model"""
#     logger.info(f"Making prediction for: {features}")
#     try:
#         data = pd.DataFrame([features.model_dump()])
#         prediction = predict(MODEL, data)
#         classes = ["setosa", "versicolor", "virginica"]
#         pred_class = classes[prediction[0]]
#     except Exception as e:
#         logger.error(f"Prediction error: {e}")
#         raise HTTPException(
#             status_code=500, 
#             detail="An error occurred during prediction"
#         )
    
#     return {"prediction": pred_class}

# # {
# #     "sepal_length": 5.1,
# #     "sepal_width": 3.3,
# #     "petal_length": 1.7,
# #     "petal_width": 0.5
# # }


"""
Insurance Cost Prediction Service
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from loguru import logger

from src.inference import load_model, predict  # предполагаем, что функции load_model и predict уже есть

logger.info("Loading model")
MODEL_PATH = os.path.join("models", "insurance_cost_model.joblib")
MODEL = load_model(MODEL_PATH)
logger.info("Model loaded successfully")


app = FastAPI(title="Insurance Cost Prediction API")

class InsuranceFeatures(BaseModel):
    """Input features for insurance cost prediction"""
    age: float
    sex: str          # "male" or "female"
    bmi: float
    children: int
    smoker: str       # "yes" or "no"
    region: str       # e.g., "southwest"

@app.get("/")
def health_check() -> dict:
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/predict")
def make_prediction(features: InsuranceFeatures) -> dict:
    """Make a prediction of insurance cost"""
    logger.info(f"Received features for prediction: {features}")
    try:
        # Преобразуем Pydantic модель в DataFrame
        data = pd.DataFrame([features.model_dump()])
        
        # Получаем прогноз от модели
        prediction = predict(MODEL, data)
        predicted_cost = float(prediction[0])  # это число
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An error occurred during prediction"
        )
    
    return {"predicted_cost": predicted_cost}
