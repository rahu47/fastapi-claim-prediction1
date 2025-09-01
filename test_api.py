import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "age": 35,
    "annual_income": 50000,
    "health_score": 7,
    "number_of_dependents": 2,
    "is_smoker": 0
}

response = requests.post(url, json=payload)
print(response.json())
