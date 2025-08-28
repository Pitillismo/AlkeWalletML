import requests
import sys

# Configurar encoding UTF-8 para toda la salida
sys.stdout.reconfigure(encoding='utf-8')

base = "http://127.0.0.1:5000"

# Datos para clasificación (deben incluir todas las columnas)
test_data_classification = {
    "features": {
        "A1": "b", "A2": 30.83, "A3": 0.0, "A4": "u", "A5": "g",
        "A6": "w", "A7": "v", "A8": 1.25, "A9": "t", "A10": "t",
        "A11": 1, "A12": "f", "A13": "g", "A14": 202.0, "A15": 0
    }
}

# Datos para regresión (sin A3)
test_data_regression = {
    "features": {
        "A1": "b", "A2": 30.83, "A4": "u", "A5": "g",
        "A6": "w", "A7": "v", "A8": 1.25, "A9": "t", "A10": "t",
        "A11": 1, "A12": "f", "A13": "g", "A14": 202.0, "A15": 0
    }
}

def test_endpoint(endpoint, method="GET", data=None):
    """Función auxiliar para probar endpoints con manejo de errores"""
    try:
        if method == "GET":
            response = requests.get(f"{base}{endpoint}", timeout=10)
        else:
            response = requests.post(f"{base}{endpoint}", json=data, timeout=10)

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error en {endpoint}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Respuesta del servidor: {e.response.text}")
        return None

# Probar health check
print("=" * 60)
print("Testing health endpoint:")
health_result = test_endpoint("/health")
if health_result:
    print(f"✓ Status: {health_result.get('status', 'N/A')}")
    print(f"✓ sklearn_version: {health_result.get('sklearn_version', 'N/A')}")
    print(f"✓ Label encoder cargado: {health_result.get('label_encoder_loaded', 'N/A')}")
print()
# Probar predicción de crédito (Logistic Regression)
print("=" * 60)
print("Testing credit prediction (Logistic Regression):")
credit_result = test_endpoint("/predict_credit", "POST", test_data_classification)
if credit_result:
    print(f"✓ Status: {credit_result.get('status', 'N/A')}")
    print(f"✓ Predicción: {credit_result.get('prediction', 'N/A')} ({credit_result.get('prediction_label', 'N/A')})")
    print(f"✓ Probabilidad: {credit_result.get('probability', 'N/A'):.4f}")
    print(f"✓ Mensaje: {credit_result.get('message', 'N/A')}")
else:
    print("✗ Predicción de crédito (Logistic Regression) falló")
print()

# Probar predicción de crédito (KNN)
print("=" * 60)
print("Testing credit prediction (KNN):")
knn_result = test_endpoint("/predict_credit_knn", "POST", test_data_classification)
if knn_result:
    print(f"✓ Status: {knn_result.get('status', 'N/A')}")
    print(f"✓ Predicción: {knn_result.get('prediction', 'N/A')} ({knn_result.get('prediction_label', 'N/A')})")
    print(f"✓ Probabilidad: {knn_result.get('probability', 'N/A'):.4f}")
    print(f"✓ Mensaje: {knn_result.get('message', 'N/A')}")
else:
    print("✗ Predicción de crédito (KNN) falló")
print()

# Probar predicción de ingresos
print("=" * 60)
print("Testing income prediction:")
income_result = test_endpoint("/predict_income", "POST", test_data_regression)
if income_result:
    print(f"✓ Ingreso predicho: {income_result.get('predicted_income', 'N/A'):.2f}")
    print(f"✓ Mensaje: {income_result.get('message', 'N/A')}")
else:
    print("✗ Predicción de ingresos falló")