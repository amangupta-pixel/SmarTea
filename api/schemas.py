# backend/api/schemas.py

from pydantic import BaseModel

class LeadInput(BaseModel):
    leadSource: str
    industry: str
    state: str
    numberOfEmployees: int
    annualRevenue: float
