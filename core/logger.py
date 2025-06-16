# backend/core/logger.py

import logging
import os

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_prediction(input_data, result):
    logging.info(f"Prediction Input: {input_data} | Output: {result}")

def log_error(error_msg):
    logging.error(f"Error: {error_msg}")
