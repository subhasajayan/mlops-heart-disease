import requests

url = "http://127.0.0.1:54529/predict"
print("Sending request...")
sample = {
    "age": 45,
    "sex": 1,
    "cp": 3,
    "trestbps": 120,
    "chol": 230,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 2,
    "ca": 0,
    "thal": 2
}

try:
    response = requests.post(url, json=sample)
    print("Status:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
