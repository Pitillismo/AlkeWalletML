# check_regression_model.py
import joblib
import pandas as pd

try:
    model_reg = joblib.load('modelos/modelo_regresion.pkl')
    print("✅ Modelo de regresión cargado correctamente")

    if hasattr(model_reg, 'steps'):
        print("🔧 Es un pipeline con los siguientes pasos:")
        for i, (name, step) in enumerate(model_reg.steps):
            print(f"   {i + 1}. {name}: {type(step)}")

    if hasattr(model_reg, 'feature_names_in_'):
        print(f"📋 Características esperadas: {list(model_reg.feature_names_in_)}")
        print(f"🔢 Número de características: {len(model_reg.feature_names_in_)}")

        if 'Class' in model_reg.feature_names_in_:
            print("❌ PROBLEMA: El modelo todavía espera la columna 'Class'")
        else:
            print("✅ El modelo NO espera la columna 'Class'")

    test_data = {
        'A1': 'b', 'A2': 30.83, 'A3': 0.0, 'A4': 'u', 'A5': 'g',
        'A6': 'w', 'A7': 'v', 'A8': 1.25, 'A9': 't', 'A10': 't',
        'A11': 1, 'A12': 'f', 'A13': 'g', 'A14': 202.0, 'A15': 0
    }

    df_test = pd.DataFrame([test_data])
    print(f"\n🧪 DataFrame de prueba: {df_test.shape[1]} columnas")
    print(f"   Columnas: {list(df_test.columns)}")

    try:
        prediction = model_reg.predict(df_test)
        print(f"✅ Predicción exitosa: {prediction[0]}")
    except Exception as ex:  # Cambié 'e' por 'ex' para evitar el shadowing
        print(f"❌ Error en predicción: {str(ex)}")

except Exception as ex:  # Cambié 'e' por 'ex' para evitar el shadowing
    print(f"❌ Error al cargar el modelo: {str(ex)}")