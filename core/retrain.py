# backend/core/retrain.py

from core.train_model_skopt import train_and_save_model
from core.logger import log_error

def trigger_retrain():
    try:
        train_and_save_model()
        return {"status": "Model retrained successfully"}
    except Exception as e:
        log_error(f"Retraining error: {str(e)}")
        return {"error": "Retraining failed"}
