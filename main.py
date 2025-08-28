# main.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Importar para verificar directorios
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.metrics import roc_curve, auc

# Actualizar estilo de seaborn (solución para la advertencia)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Crear directorio para modelos si no existe
os.makedirs('modelos', exist_ok=True)

# -------------------------------
# Carga de datos
# -------------------------------
print("Cargando dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
column_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'Class']
data = pd.read_csv(url, names=column_names, na_values='?')

print("Dimensiones:", data.shape)
print("Valores faltantes por columna:\n", data.isnull().sum())

# Distribución de la clase
data['Class'].value_counts().plot(kind='bar')
plt.title('Distribución de Class')
plt.xlabel('Class')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()

# -------------------------------
# Codificación variable objetivo (clasificación)
# -------------------------------
le = LabelEncoder()
data['Class'] = le.fit_transform(data['Class'])  # '+'->1, '-'->0

# -------------------------------
# CLASIFICACIÓN (Logistic Regression)
# -------------------------------
X = data.drop('Class', axis=1)
y = data['Class']

# Columnas detectadas desde X (mejor que fijarlas a mano)
num_cols_clf = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_clf = X.select_dtypes(include=['object']).columns.tolist()


# Función para crear pipeline de preprocesamiento (evita duplicación)
def create_preprocessor(num_cols, cat_cols):
    num_tf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_tf = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', num_tf, num_cols),
        ('cat', cat_tf, cat_cols)
    ])


preprocessor_clf = create_preprocessor(num_cols_clf, cat_cols_clf)

clf_pipeline = Pipeline([
    ('preprocessor', preprocessor_clf),
    ('model', LogisticRegression(max_iter=1000))  # Eliminado n_jobs=None (es el valor por defecto)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores_clf = cross_val_score(clf_pipeline, X_train, y_train, cv=kfold, scoring='accuracy')
print("Accuracy CV (clasif - LogReg):", scores_clf.mean())

clf_pipeline.fit(X_train, y_train)
y_pred = clf_pipeline.predict(X_test)
print("Accuracy test (clasif - LogReg):", accuracy_score(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte clasificación:\n", classification_report(y_test, y_pred))

joblib.dump(clf_pipeline, 'modelos/modelo_clasificacion.pkl')

# -------------------------------
# CLASIFICACIÓN (KNN opcional)
# -------------------------------
knn_pipeline = Pipeline([
    ('preprocessor', preprocessor_clf),
    ('model', KNeighborsClassifier(n_neighbors=5))
])
knn_pipeline.fit(X_train, y_train)
joblib.dump(knn_pipeline, 'modelos/modelo_knn.pkl')

# -------------------------------
# REGRESIÓN (predecir A3 - ingreso)
# -------------------------------
X_reg = data.drop(['A3', 'Class'], axis=1)
y_reg = data['A3']

# Importante: construir otro preprocessor con columnas de X_reg
num_cols_reg = X_reg.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_reg = X_reg.select_dtypes(include=['object']).columns.tolist()

preprocessor_reg = create_preprocessor(num_cols_reg, cat_cols_reg)

reg_pipeline = Pipeline([
    ('preprocessor', preprocessor_reg),
    ('model', LinearRegression())
])

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

scores_reg = cross_val_score(reg_pipeline, X_train_reg, y_train_reg, cv=kfold, scoring='r2')
print("R2 CV (regresión):", scores_reg.mean())

reg_pipeline.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_pipeline.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg))
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("R2:", r2_score(y_test_reg, y_pred_reg))

# Para Logistic Regression
y_pred_proba = clf_pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"AUC-ROC: {roc_auc}")

# Gráfico ROC
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

joblib.dump(reg_pipeline, 'modelos/modelo_regresion.pkl')

# Guarda el encoder para mapear 0/1 a '+/-' si hace falta
joblib.dump(le, 'modelos/label_encoder.pkl')

print("✅ Artefactos guardados: modelo_clasificacion.pkl, modelo_knn.pkl, modelo_regresion.pkl, label_encoder.pkl")