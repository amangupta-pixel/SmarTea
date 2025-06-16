# backend/api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from core.predict_model import predict_conversion
from api.schemas import LeadInput
from fastapi.middleware.cors import CORSMiddleware
from core.retrain import trigger_retrain

app = FastAPI(title="SmarTea Lead Scoring API")

# Enable CORS so frontend can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class LeadInput(BaseModel):
    leadSource: str
    industry: str
    state: str
    numberOfEmployees: int
    annualRevenue: float

@app.get("/")
def read_root():
    return {"message": "SmarTea Lead Scoring API is live."}

# Prediction route
@app.post("/predict")
def predict(lead: LeadInput):
    return predict_conversion(lead.dict())

# Retraining route
@app.post("/retrain")
def retrain():
    return trigger_retrain()