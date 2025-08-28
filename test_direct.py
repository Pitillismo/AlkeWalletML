# test_direct.py
import requests

# Prueba directa al endpoint de ingresos
url = "http://127.0.0.1:5000/predict_income"

# Enviar solo las características que el modelo de regresión espera
data = {
    "features": {
        "A1": "b", "A2": 30.83, "A4": "u", "A5": "g",
        "A6": "w", "A7": "v", "A8": 1.25, "A9": "t",
        "A10": "t", "A11": 1, "A12": "f", "A13": "g",
        "A14": 202.0, "A15": 0
    }
}

response = requests.post(url, json=data)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")