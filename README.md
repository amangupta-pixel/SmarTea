# 🚀 SmarTea CRM Backend – AI-Powered Lead Scoring API

This is the backend module of the **SmarTea Business Automation Software**. It powers the **AI-driven lead scoring system**, built using FastAPI, XGBoost, and Bayesian hyperparameter optimization (`skopt`).

---

## 📦 Features

- ✅ FastAPI-based REST API
- ✅ Predicts lead conversion probability
- ✅ XGBoost model optimized via `BayesSearchCV`
- ✅ Fully containerized with Docker
- ✅ Logs predictions and errors
- ✅ Retraining endpoint (`/retrain`) for admin panel

---

## 🗂️ Complete Backend Project Structure

backend/
├── api/
│ ├── pycache/
│ ├── main.py # FastAPI app w/ prediction + retrain routes
│ └── schemas.py # Pydantic models for request/response validation
│
├── core/
│ ├── pycache/
│ ├── logger.py # Centralized logging setup
│ ├── predict_model.py # Loads model, runs prediction
│ ├── preprocess.py # Data loading, preprocessing pipeline
│ ├── retrain.py # Triggers model retraining (used in /retrain)
│ └── train_model_skopt.py # XGBoost model training with skopt
│
├── data/
│ └── Lead_data.csv # Cleaned and labeled dataset for training
│
├── logs/
│ └── app.log # Logs predictions and retrain events
│
├── models/
│ └── lead_model.pkl # Trained XGBoost pipeline (joblib)
│
├── .dockerignore # Prevents unnecessary files from being in Docker builds
├── .gitignore # Git exclusions (ignores model, venv, logs, etc.)
├── Dockerfile # For Dockerizing the backend API
├── README.md # This file
├── requirements.txt # Python dependency list
├── test_predict.py # Local model test using sample input
└── test_requests.py # Sends request to running FastAPI server