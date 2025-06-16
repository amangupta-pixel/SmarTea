from core.predict_model import predict_conversion

test_input = {
    "leadSource": "Email Campaign",
    "industry": "Retail",
    "state": "Montana",
    "numberOfEmployees": 279,
    "annualRevenue": 5406867
}
result = predict_conversion(test_input)
print(result)
