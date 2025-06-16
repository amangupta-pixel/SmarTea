# backend/core/predict_model.py

import joblib
import numpy as np
import os
import pandas as pd
from core.logger import log_prediction, log_error

MODEL_PATH = os.path.join("models", "lead_model.pkl")

def predict_conversion(input_data: dict):
    try:
        pipeline = joblib.load(MODEL_PATH)

        input_df = pd.DataFrame([input_data])

        prob = pipeline.predict_proba(input_df)[0][1]
        prediction = int(prob >= 0.5)

        result = {
            "converted": prediction,
            "probability": float(round(prob, 3))
        }

        log_prediction(input_data, result)
        return result

    except Exception as e:
        log_error(str(e))
        return {"error": "Prediction failed"}
