from flask import Flask, request, jsonify, g
import joblib
import pandas as pd
import time
import uuid
import logging
import sys
import os  # Importar para verificar existencia de archivos
from sklearn import __version__ as skl_version
from functools import wraps

# -----------------------------
# Configuración de logging
# -----------------------------
logger = logging.getLogger("alke_api")
handler = logging.StreamHandler(sys.stdout)


# Formatter que maneja la ausencia de req_id
class SafeFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'req_id'):
            record.req_id = 'N/A'
        return super().format(record)


formatter = SafeFormatter(
    fmt="%(asctime)s | %(levelname)s | req_id=%(req_id)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z"
)

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Función segura para obtener req_id
def get_req_id():
    try:
        return g.get('req_id', 'N/A')
    except RuntimeError:
        return 'N/A'


def log_extra():
    return {"req_id": get_req_id()}


# Decorador para manejar excepciones de API
def handle_api_exceptions(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as val_err:
            logger.warning(f"bad_request: {val_err}", extra=log_extra())
            return jsonify({"error": str(val_err)}), 400
        except Exception as other_ex:
            logger.exception(f"internal_error: {other_ex}", extra=log_extra())
            error_msg = "Internal server error" if f.__name__ != 'predict_income' else str(other_ex)
            return jsonify({"error": error_msg}), 500

    return decorated_function


# -----------------------------
# App
# -----------------------------
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Para mostrar correctamente caracteres especiales

# Verificar que los archivos de modelos existen antes de cargarlos
model_files = {
    'clasificacion': 'modelos/modelo_clasificacion.pkl',
    'knn': 'modelos/modelo_knn.pkl',
    'regresion': 'modelos/modelo_regresion.pkl',
    'label_encoder': 'modelos/label_encoder.pkl'
}

for model_name, file_path in model_files.items():
    if not os.path.exists(file_path):
        logger.error(f"Archivo de modelo no encontrado: {file_path}")
        # Puedes decidir si quieres salir o continuar dependiendo de la importancia del modelo

# Cargar pipelines entrenados y label encoder
try:
    model_clf = joblib.load(model_files['clasificacion'])
    model_knn = joblib.load(model_files['knn'])
    model_reg = joblib.load(model_files['regresion'])
    label_encoder = joblib.load(model_files['label_encoder'])

    logger.info("Modelos cargados exitosamente")
    logger.info(f"Modelo clasificación: {model_clf.__class__.__name__}")
    logger.info(f"Modelo KNN: {model_knn.__class__.__name__}")
    logger.info(f"Modelo regresión: {model_reg.__class__.__name__}")
    logger.info(f"Label encoder: {label_encoder.__class__.__name__}")

except Exception as load_exception:
    logger.error(f"Error al cargar modelos: {str(load_exception)}")
    # Dependiendo de tu caso, podrías querer salir o continuar
    raise

RAW_COLS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
NUMERIC_COLS = {'A2', 'A3', 'A8', 'A11', 'A14', 'A15'}
CATEGORICAL_COLS = set(RAW_COLS) - NUMERIC_COLS


# -----------------------------
# Hooks para tracing
# -----------------------------
@app.before_request
def before_request():
    g.req_id = str(uuid.uuid4())[:8]
    g.t0 = time.time()
    logger.info(f"{request.method} {request.path}", extra=log_extra())


@app.after_request
def after_request(response):
    ms = int((time.time() - g.t0) * 1000) if hasattr(g, "t0") else -1
    logger.info(f"status={response.status_code} latency_ms={ms}", extra=log_extra())
    return response


# -----------------------------
# Utilidades de validación
# -----------------------------
def frame_payload(payload: dict, require_all_columns=True) -> pd.DataFrame:
    if 'features' not in payload or not isinstance(payload['features'], dict):
        raise ValueError("Debes enviar un objeto 'features' con claves A1..A15.")

    feats = payload['features']
    logger.info(f"Datos recibidos: {feats}", extra=log_extra())

    # SOLO validar columnas faltantes si require_all_columns es True
    if require_all_columns:
        missing = [col for col in RAW_COLS if col not in feats]
        if missing:
            raise ValueError(f"Faltan columnas: {missing}")

    row = {}
    # Solo procesar las columnas presentes en los features
    columns_to_process = list(feats.keys()) if not require_all_columns else RAW_COLS

    for col in columns_to_process:
        val = feats[col]

        if col in NUMERIC_COLS:
            if val in (None, "", " ", "NA", "NaN"):
                row[col] = None
            else:
                try:
                    row[col] = float(val)
                except Exception as conversion_error:
                    raise ValueError(f"'{col}' debe ser numérico; recibido: {val!r}") from conversion_error
        else:
            row[col] = str(val) if val is not None else None

    return pd.DataFrame([row], columns=list(row.keys()))


# Función auxiliar para predicciones de crédito (reduce duplicación)
def predict_credit_common(model, df, model_type_name):
    pred = model.predict(df)
    proba = getattr(model, "predict_proba", None)
    p1 = float(proba(df)[:, 1][0]) if proba else None

    original_label = label_encoder.inverse_transform(pred)[0]

    return jsonify({
        "prediction": int(pred[0]),
        "prediction_label": str(original_label),
        "probability": p1,
        "status": 'approved' if pred[0] == 1 else 'denied',
        "message": f"Evaluación completada usando {model_type_name}"
    })

# -----------------------------
# Endpoints
# -----------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "sklearn_version": skl_version,
        "models_loaded": {
            "classification_logreg": model_clf.__class__.__name__,
            "classification_knn": model_knn.__class__.__name__,
            "regression": model_reg.__class__.__name__
        },
        "label_encoder_loaded": True,
        "expected_columns": RAW_COLS
    })


@app.route('/', methods=['GET'])
def home():
    return "API de Evaluación Crediticia - Pipelines (RAW A1..A15). Endpoints: /health, /predict_credit, /predict_credit_knn, /predict_income"


@app.route('/predict_credit', methods=['POST'])
@handle_api_exceptions
def predict_credit():
    payload = request.get_json(force=True, silent=False)
    df = frame_payload(payload)
    return predict_credit_common(model_clf, df, "Regresión Logística")

@app.route('/predict_credit_knn', methods=['POST'])
@handle_api_exceptions
def predict_credit_knn():
    payload = request.get_json(force=True, silent=False)
    df = frame_payload(payload)
    return predict_credit_common(model_knn, df, "K-Nearest Neighbors")

@app.route('/predict_income', methods=['POST'])
@handle_api_exceptions
def predict_income():
    payload = request.get_json(force=True, silent=False)
    df = frame_payload(payload, require_all_columns=False)

    logger.info(f"DataFrame inicial: {df.columns.tolist()}", extra=log_extra())

    if hasattr(model_reg, 'feature_names_in_'):
        expected_features = set(model_reg.feature_names_in_)
        logger.info(f"El modelo espera: {expected_features}", extra=log_extra())

        current_features = set(df.columns)
        missing_features = expected_features - current_features
        extra_features = current_features - expected_features

        logger.info(f"Características actuales: {current_features}", extra=log_extra())
        logger.info(f"Faltantes: {missing_features}", extra=log_extra())
        logger.info(f"Sobrantes: {extra_features}", extra=log_extra())

        for feature in missing_features:
            logger.info(f"Añadiendo característica faltante: {feature} = 0", extra=log_extra())
            df[feature] = 0

        for feature in extra_features:
            if feature in df.columns:
                logger.info(f"Eliminando característica sobrante: {feature}", extra=log_extra())
                df = df.drop(columns=[feature])

        df = df[list(model_reg.feature_names_in_)]
        logger.info(f"DataFrame final: {df.columns.tolist()}", extra=log_extra())

    yhat = model_reg.predict(df)
    logger.info(f"Predicción: {yhat[0]}", extra=log_extra())

    return jsonify({
        "predicted_income": float(yhat[0]),
        "message": "Predicción de ingresos completada"
    })


# -----------------------------
# Arranque
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)