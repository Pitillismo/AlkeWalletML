# Proyecto Alke Wallet - Machine Learning de Evaluación Crediticia Predictiva
## Descripción
Sistema de machine learning para predecir la aprobación crediticia de usuarios utilizando datos históricos. Este proyecto fue desarrollado como parte del módulo de Machine Learning para Ingenieros de Datos.
## Estructura del Proyecto
```
AlkeWalletML/
├── .venv/                     # Entorno virtual
├── img/                       # Gráficos y visualizaciones
│   ├── class_distribution.png
│   └── roc_curve.png
├── modelos/                   # Modelos serializados
│   ├── label_encoder.pkl      # Encoder para variable objetivo
│   ├── modelo_clasificacion.pkl  # Modelo de Regresión Logística
│   ├── modelo_knn.pkl         # Modelo KNN (alternativo)
│   └── modelo_regresion.pkl   # Modelo de Regresión Lineal
├── api_pipeline.py            # API principal con endpoints
├── check_regression_model.py  # Valida el modelo de regresión
├── diagnostic.py              # Herramientas de diagnóstico
├── main.py                    # Pipeline completo de ML
├── README.md                  # Este archivo
├── requirements.txt           # Dependencias
├── test_api.py               # Pruebas automatizadas de la API
├── test_direct.py            # Prueba específica de predicción de ingresos
└── test_api.http             # Pruebas con REST Client

```
## Lecciones Aplicadas
### Lección 1: Fundamentos del Aprendizaje de Máquina
- Análisis exploratorio del dataset UCI Credit Approval
- Identificación del problema de clasificación binaria (aprobación crediticia) y regresión (predicción de ingresos)
### Lección 2: Validación Cruzada y Ajuste del Modelo
- Implementación de k-fold cross-validation (k=5)
- Análisis de overfitting/underfitting con métricas de entrenamiento y prueba
### Lección 3: Preprocesamiento y Escalamiento de Datos
- Imputación de valores faltantes con mediana (numéricas) y moda (categóricas)
- One-Hot Encoding para variables categóricas
- StandardScaler para normalización
### Lección 4: Modelado de Regresión
- Entrenamiento de modelo de Regresión Lineal para predecir ingresos
- Evaluación con MAE, MSE, RMSE y R²
### Lección 5: Modelado de Clasificación
- Entrenamiento de modelos de Regresión Logística y KNN para clasificación
- Evaluación con matriz de confusión, accuracy, precisión, recall y AUC-ROC
### Lección 6: Despliegue del Modelo como API
- Creación de API RESTful con Flask
- Endpoints para predicciones en tiempo real
- Guardado de modelos con joblib
### Lección 7: Evaluación, Monitoreo y Cierre
- Pruebas integrales de la API con REST Client/Postman y scripts automáticos (test_api.py y test_direct.py)
- Pruebas integrales con Archivos .http para REST Client (extensión de VS Code) en test_api.http
- Documentación del pipeline y decisiones técnicas
## Instalación
1. Clonar el repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd AlkeWalletML
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar el pipeline de ML:
   ```bash
   python main.py
   ```
4. Iniciar la API:
   ```bash
   python api_pipeline.py
   ```
## Uso de la API
La API estará disponible en `http://localhost:5000`.
### Endpoints
- `GET /health` - Verificar estado del servicio y modelos cargados
- `POST /predict_credit` - Evaluación crediticia (Regresión Logística)
- `POST /predict_credit_knn` - Evaluación crediticia (KNN)
- `POST /predict_income` - Predicción de ingresos
### Ejemplo de Request y Response
```json
{
  "features": {
    "A1": "b",
    "A2": 30.83,
    "A4": "u",
    "A5": "g",
    "A6": "w",
    "A7": "v",
    "A8": 1.25,
    "A9": "t",
    "A10": "t",
    "A11": 1,
    "A12": "f",
    "A13": "g",
    "A14": 202.0,
    "A15": 0
  }
}
```
```json
{
  "prediction": 0,
  "prediction_label": "-",
  "probability": 0.1428,
  "status": "denied",
  "message": "Evaluación completada usando Regresión Logística"
}
```
### Ejemplo Predicción de ingresos 
##### (Nota importante: El campo A3 se omite deliberadamente porque es la variable objetivo que el modelo de regresión intenta predecir.)
```json
{
  "features": {
    "A1": "b",
    "A2": 30.83,
    "A4": "u",
    "A5": "g", 
    "A6": "w",
    "A7": "v",
    "A8": 1.25,
    "A9": "t",
    "A10": "t",
    "A11": 1,
    "A12": "f",
    "A13": "g",
    "A14": 202.0,
    "A15": 0
  }
}
```
```json
{
  "predicted_income": 4.98,
  "message": "Predicción de ingresos completada"
}
```
### Ejemplo de uso con Python:
```python
import requests

# URL del endpoint de predicción de ingresos
api_url = "http://localhost:5000/predict_income"

# Datos de ejemplo (sin A3)
data = {
    "features": {
        "A1": "b", "A2": 30.83, "A4": "u",
        "A5": "g", "A6": "w", "A7": "v", "A8": 1.25,
        "A9": "t", "A10": "t", "A11": 1, "A12": "f",
        "A13": "g", "A14": 202.0, "A15": 0
    }
}

try:
    # Realizar la solicitud
    response = requests.post(api_url, json=data, timeout=10)
    response.raise_for_status()  # Verificar errores HTTP
    
    # Procesar la respuesta
    result = response.json()
    print(f"Ingreso predicho: ${result['predicted_income']:.2f}")
    print(f"Mensaje: {result['message']}")
    
except requests.exceptions.RequestException as e:
    print(f"Error en la solicitud: {e}")
except KeyError as e:
    print(f"Error en el formato de la respuesta: {e}")
```
## 🔌 Integración con Sistemas Existentes

El sistema está diseñado para integración sencilla con aplicaciones existentes:

```python
# Ejemplo de función de integración
import requests

def evaluar_credito(datos_solicitante, api_base_url="http://localhost:5000"):
    api_url = f"{api_base_url}/predict_credit"
    response = requests.post(api_url, json=datos_solicitante, timeout=30)
    response.raise_for_status()  # Lanza excepción para errores HTTP
    return response.json()

# Uso
resultado = evaluar_credito({
    "features": {
        "A1": "b", "A2": 30.83, "A3": 0.0, "A4": "u",
        "A5": "g", "A6": "w", "A7": "v", "A8": 1.25,
        "A9": "t", "A10": "t", "A11": 1, "A12": "f",
        "A13": "g", "A14": 202.0, "A15": 0
    }
})

# Para producción: Reemplazar localhost:5000 con la URL del servidor de producción y agregar mecanismos de autenticación.
#  Este código es un ejemplo de cómo cualquier aplicación externa (escrita en Python o cualquier otro lenguaje que pueda hacer HTTP requests) puede consumir la API. No es parte de los scripts del proyecto, sino una guía para desarrolladores que quieran integrar la API en sus sistemas.
```
## 📈 Resultados

### Clasificación (Evaluación Crediticia) - Regresión Logística
- **Accuracy en validación cruzada**: 86.6%
- **Accuracy en prueba**: 83.3%
- **Precisión**: 87.5% (aprobados), 78.8% (denegados)
- **Recall**: 81.8% (aprobados), 85.2% (denegados)
- **F1-score**: 0.845 (aprobados), 0.82 (denegados)
- **AUC-ROC**: 0.903 (excelente)
- **Matriz de Confusión**: 
[[52 9]
[14 63]]

### Regresión (Predicción de Ingresos)
- **R²**: 0.026
- **MAE**: 4.111
- **MSE**: 30.162
- **RMSE**: 5.492

**Nota importante**: El modelo de regresión muestra un poder predictivo muy bajo (R² cercano a 0), lo que indica que las variables disponibles no son buenas predictoras del ingreso. Se recomienda no usar este modelo para decisiones críticas o buscar mejores características.
## ⚠️ Notas Técnicas

El proyecto genera warnings de "categorías desconocidas" durante la validación cruzada, lo cual es normal y esperado. Estos warnings indican que algunas categorías en los conjuntos de prueba no estaban presentes en los conjuntos de entrenamiento durante la validación cruzada, pero el pipeline maneja estas situaciones correctamente mediante la parameterización `handle_unknown='ignore'` en el OneHotEncoder.

## Solución de Problemas

### Error: Categorías desconocidas
Si ves warnings sobre categorías desconocidas, esto es normal durante la validación cruzada y no afecta el funcionamiento.

### Error: Modelos no encontrados
Asegúrate de ejecutar `python main.py` antes de iniciar la API para generar los modelos.

### Error: Puerto en uso
Si el puerto 5000 está ocupado, puedes cambiarlo en `api_pipeline.py`:
```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)  # Cambia el puerto
```

## Referencias
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Flask](https://flask.palletsprojects.com/en/stable/)
- [UCI Credit Approval Dataset](https://archive.ics.uci.edu/ml/datasets/Credit+Approval)

## Autor
[Catalina Millán Coronado]
