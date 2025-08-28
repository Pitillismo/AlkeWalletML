# diagnostic.py
import joblib
import pandas as pd

def debug_model(model_path, model_name):
    print(f"\n=== DEBUG: {model_name} ===")

    try:
        model = joblib.load(model_path)
        print(f"✅ {model_name} cargado correctamente")
        print(f"Tipo: {type(model)}")

        if hasattr(model, 'steps'):
            print("Es un Pipeline con los siguientes pasos:")
            for i, (name, step) in enumerate(model.steps):
                print(f"  {i + 1}. {name}: {type(step)}")

                if name == 'preprocessor' and hasattr(step, 'transformers'):
                    print("     Transformers:")
                    for transformer_name, transformer, columns in step.transformers:
                        print(f"       - {transformer_name}: {columns}")

        if hasattr(model, 'feature_names_in_'):
            print(f"Características esperadas: {model.feature_names_in_}")
            print(f"Número de características: {len(model.feature_names_in_)}")

            if 'Class' in model.feature_names_in_:
                print("❌ EL MODELO ESPERA LA COLUMNA 'Class'")
            else:
                print("✅ El modelo NO espera la columna 'Class'")

        elif hasattr(model, 'named_steps'):
            if 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                if hasattr(preprocessor, 'feature_names_in_'):
                    print(f"Características del preprocesador: {preprocessor.feature_names_in_}")
                    if 'Class' in preprocessor.feature_names_in_:
                        print("❌ EL PREPROCESADOR ESPERA LA COLUMNA 'Class'")

        return True

    except Exception as ex:  # Cambié 'e' por 'ex' para evitar el shadowing
        print(f"❌ Error al cargar {model_name}: {str(ex)}")
        return False

# Debug de todos los modelos
debug_model('modelos/modelo_clasificacion.pkl', 'Modelo Clasificación')
debug_model('modelos/modelo_knn.pkl', 'Modelo KNN')
debug_model('modelos/modelo_regresion.pkl', 'Modelo Regresión')

# Probar con datos de ejemplo
print("\n=== PRUEBA CON DATOS DE EJEMPLO ===")
test_data = {
    'A1': 'b', 'A2': 30.83, 'A3': 0.0, 'A4': 'u', 'A5': 'g',
    'A6': 'w', 'A7': 'v', 'A8': 1.25, 'A9': 't', 'A10': 't',
    'A11': 1, 'A12': 'f', 'A13': 'g', 'A14': 202.0, 'A15': 0
}

df_test = pd.DataFrame([test_data])
print(f"DataFrame de prueba: {df_test.shape[1]} columnas")
print(f"Columnas: {list(df_test.columns)}")

# Probar predicción con el modelo de regresión
try:
    model_reg = joblib.load('modelos/modelo_regresion.pkl')
    prediction = model_reg.predict(df_test)
    print(f"✅ Predicción exitosa: {prediction[0]}")
except Exception as ex:  # Cambié 'e' por 'ex' para evitar el shadowing
    print(f"❌ Error en predicción: {str(ex)}")