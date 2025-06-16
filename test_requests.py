import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "leadSource": "Email Campaign",
    "industry": "Retail",
    "state": "Montana",
    "numberOfEmployees": 279,
    "annualRevenue": 5406867
}

response = requests.post(url, json=payload)
print("Prediction:", response.json())
