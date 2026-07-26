# ============================================================
# RANDOM FOREST PARA DATASET YFINANCE
# Columnas esperadas: date, symbol, close
# Outputs:
# 1. Clasificación de tendencia
# 2. Regresión de retorno
# 3. Predicción de volatilidad
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

# Cambia este nombre por el archivo real que tengas
input_path = "market_data.csv"

# Carpeta donde se guardarán los modelos y resultados
# En Mac quedará dentro del mismo lugar donde está corriendo tu notebook
output_dir = Path.cwd() / "models_yfinance"
output_dir.mkdir(parents=True, exist_ok=True)

print("Directorio actual del notebook:")
print(Path.cwd())

print("\nLos modelos y resultados se guardarán en:")
print(output_dir)


# Parámetros del ejercicio
RETURN_HORIZON = 1        # 1 = retorno del siguiente día
VOLATILITY_WINDOW = 5     # volatilidad futura usando próximos 5 días


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida y limpia el dataset base.
    Espera columnas: date, symbol, close
    """

    df = df.copy()

    # Normalizar nombres de columnas por si vienen con mayúsculas
    df.columns = [col.strip().lower() for col in df.columns]

    required_columns = ["date", "symbol", "close"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["date", "symbol", "close"])

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("El dataset quedó vacío después de la limpieza.")

    return df


def add_future_volatility(group: pd.DataFrame, volatility_window: int) -> pd.DataFrame:
    """
    Calcula volatilidad futura simple por símbolo.

    Para cada fecha t, estima la desviación estándar de los retornos
    de los próximos N días.
    """

    group = group.copy()

    # Retorno del siguiente día alineado en la fecha actual
    future_1d_return = group["return_1d"].shift(-1)

    # Rolling hacia adelante usando reversa
    group["target_volatility"] = (
        future_1d_return
        .iloc[::-1]
        .rolling(window=volatility_window, min_periods=volatility_window)
        .std()
        .iloc[::-1]
    )

    return group


def create_features_and_targets(
    df: pd.DataFrame,
    return_horizon: int = 1,
    volatility_window: int = 5
) -> pd.DataFrame:
    """
    Crea variables predictoras y targets:

    target_trend:
        1 si el retorno futuro es positivo, 0 si es negativo o cero.

    target_return_pct:
        Retorno porcentual futuro.

    target_volatility:
        Volatilidad futura simple.
    """

    df = df.copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # --------------------------
    # Features históricas
    # --------------------------

    df["return_1d"] = df.groupby("symbol")["close"].pct_change(1)
    df["return_2d"] = df.groupby("symbol")["close"].pct_change(2)
    df["return_5d"] = df.groupby("symbol")["close"].pct_change(5)
    df["return_10d"] = df.groupby("symbol")["close"].pct_change(10)

    df["ma_5"] = (
        df.groupby("symbol")["close"]
        .transform(lambda x: x.rolling(5).mean())
    )

    df["ma_10"] = (
        df.groupby("symbol")["close"]
        .transform(lambda x: x.rolling(10).mean())
    )

    df["ma_20"] = (
        df.groupby("symbol")["close"]
        .transform(lambda x: x.rolling(20).mean())
    )

    df["volatility_5"] = (
        df.groupby("symbol")["return_1d"]
        .transform(lambda x: x.rolling(5).std())
    )

    df["volatility_10"] = (
        df.groupby("symbol")["return_1d"]
        .transform(lambda x: x.rolling(10).std())
    )

    df["close_to_ma_5"] = df["close"] / df["ma_5"] - 1
    df["close_to_ma_10"] = df["close"] / df["ma_10"] - 1
    df["close_to_ma_20"] = df["close"] / df["ma_20"] - 1

    # Variables de calendario
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # --------------------------
    # Targets futuros
    # --------------------------

    df["future_close"] = (
        df.groupby("symbol")["close"]
        .shift(-return_horizon)
    )

    # Target 1 y 2: retorno futuro porcentual
    df["target_return_pct"] = df["future_close"] / df["close"] - 1

    # Target de clasificación
    df["target_trend"] = np.where(
        df["target_return_pct"] > 0,
        1,
        0
    )

    # Target 3: volatilidad futura
    df = (
        df.groupby("symbol", group_keys=False)
        .apply(lambda x: add_future_volatility(x, volatility_window))
    )

    # Eliminar filas sin suficiente historia o futuro
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Después de crear features y targets no quedaron registros. "
            "Revisa que tengas suficiente historia por símbolo."
        )

    return df


def regression_metrics(y_true, y_pred) -> dict:
    """
    Calcula métricas de regresión.
    """

    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2_score": float(r2_score(y_true, y_pred))
    }


def get_positive_probability(model, X):
    """
    Obtiene probabilidad de clase positiva cuando existe.
    Maneja casos donde el modelo sólo vio una clase.
    """

    classifier = model.named_steps["model"]

    if not hasattr(classifier, "predict_proba"):
        return np.nan

    probabilities = model.predict_proba(X)

    if probabilities.shape[1] == 1:
        only_class = classifier.classes_[0]

        if only_class == 1:
            return np.ones(len(X))
        else:
            return np.zeros(len(X))

    positive_class_index = list(classifier.classes_).index(1)

    return probabilities[:, positive_class_index]


# ============================================================
# 3. LECTURA Y PREPARACIÓN DEL DATASET
# ============================================================

print("\nLeyendo dataset...")

df = pd.read_csv(input_path, sep = ',')

print("Primeras filas del dataset original:")
display(df.head())

print("\nValidando columnas y tipos de datos...")
df = validate_input_data(df)

print("\nDataset limpio:")
display(df.head())

print("\nCreando features y targets...")
df_model = create_features_and_targets(
    df=df,
    return_horizon=RETURN_HORIZON,
    volatility_window=VOLATILITY_WINDOW
)

print("\nDataset modelado:")
display(df_model.head())

print("\nShape dataset modelado:")
print(df_model.shape)


# ============================================================
# 4. DEFINICIÓN DE FEATURES Y TARGETS
# ============================================================

categorical_features = [
    "symbol"
]

numerical_features = [
    "close",
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "ma_5",
    "ma_10",
    "ma_20",
    "volatility_5",
    "volatility_10",
    "close_to_ma_5",
    "close_to_ma_10",
    "close_to_ma_20",
    "day_of_week",
    "month"
]

features = categorical_features + numerical_features

target_classification = "target_trend"
target_return = "target_return_pct"
target_volatility = "target_volatility"


# ============================================================
# 5. SPLIT TEMPORAL TRAIN / TEST
# ============================================================

# Importante para series financieras:
# no hacemos split aleatorio, sino que entrenamos con datos antiguos
# y probamos con datos más recientes.

df_model = df_model.sort_values(["date", "symbol"]).reset_index(drop=True)

split_index = int(len(df_model) * 0.8)

train_df = df_model.iloc[:split_index].copy()
test_df = df_model.iloc[split_index:].copy()

X_train = train_df[features]
X_test = test_df[features]

y_trend_train = train_df[target_classification]
y_trend_test = test_df[target_classification]

y_return_train = train_df[target_return]
y_return_test = test_df[target_return]

y_vol_train = train_df[target_volatility]
y_vol_test = test_df[target_volatility]

print("\nTamaño train:")
print(X_train.shape)

print("\nTamaño test:")
print(X_test.shape)

print("\nDistribución target tendencia en train:")
print(y_trend_train.value_counts(normalize=True))

print("\nDistribución target tendencia en test:")
print(y_trend_test.value_counts(normalize=True))


# ============================================================
# 6. PREPROCESADOR
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipeline, categorical_features),
        ("numerical", numerical_pipeline, numerical_features)
    ]
)


# ============================================================
# 7. MODELOS RANDOM FOREST
# ============================================================

trend_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

return_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42
            )
        )
    ]
)

volatility_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42
            )
        )
    ]
)


# ============================================================
# 8. ENTRENAMIENTO
# ============================================================

print("\nEntrenando modelo 1: Clasificación de tendencia...")
trend_model.fit(X_train, y_trend_train)

print("Entrenando modelo 2: Regresión de retorno...")
return_model.fit(X_train, y_return_train)

print("Entrenando modelo 3: Predicción de volatilidad...")
volatility_model.fit(X_train, y_vol_train)

print("\nEntrenamiento terminado correctamente.")


# ============================================================
# 9. PREDICCIONES
# ============================================================

print("\nGenerando predicciones...")

trend_pred = trend_model.predict(X_test)
return_pred = return_model.predict(X_test)
volatility_pred = volatility_model.predict(X_test)

positive_probability = get_positive_probability(trend_model, X_test)


# ============================================================
# 10. MÉTRICAS
# ============================================================

classification_metrics = {
    "accuracy": float(accuracy_score(y_trend_test, trend_pred)),
    "precision": float(precision_score(y_trend_test, trend_pred, zero_division=0)),
    "recall": float(recall_score(y_trend_test, trend_pred, zero_division=0)),
    "f1_score": float(f1_score(y_trend_test, trend_pred, zero_division=0))
}

return_metrics = regression_metrics(y_return_test, return_pred)
volatility_metrics = regression_metrics(y_vol_test, volatility_pred)

metrics = {
    "trend_classification": classification_metrics,
    "return_regression": return_metrics,
    "volatility_regression": volatility_metrics
}

print("\nMétricas finales:")
print(json.dumps(metrics, indent=4))


# ============================================================
# 11. DATAFRAME FINAL DE PREDICCIONES
# ============================================================

predictions_df = test_df.copy()

predictions_df["predicted_trend"] = trend_pred

predictions_df["predicted_trend_label"] = np.where(
    predictions_df["predicted_trend"] == 1,
    "positive",
    "negative"
)

predictions_df["predicted_return_pct"] = return_pred
predictions_df["predicted_volatility"] = volatility_pred
predictions_df["predicted_positive_probability"] = positive_probability

print("\nPrimeras predicciones:")
display(
    predictions_df[
        [
            "date",
            "symbol",
            "close",
            "target_trend",
            "target_return_pct",
            "target_volatility",
            "predicted_trend_label",
            "predicted_return_pct",
            "predicted_volatility",
            "predicted_positive_probability"
        ]
    ].head()
)


# ============================================================
# 12. GUARDAR MODELOS Y RESULTADOS
# ============================================================

print("\nGuardando modelos y resultados en:")
print(output_dir)

trend_model_path = output_dir / "yf_trend_classification_random_forest.joblib"
return_model_path = output_dir / "yf_return_regression_random_forest.joblib"
volatility_model_path = output_dir / "yf_volatility_random_forest.joblib"

metrics_path = output_dir / "yf_metrics.json"
predictions_path = output_dir / "yf_predictions.csv"
metadata_path = output_dir / "yf_feature_metadata.json"

joblib.dump(trend_model, trend_model_path)
joblib.dump(return_model, return_model_path)
joblib.dump(volatility_model, volatility_model_path)

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)

predictions_df.to_csv(predictions_path, index=False)

metadata = {
    "input_columns": ["date", "symbol", "close"],
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "targets": {
        "target_trend": "1 if future return is positive else 0",
        "target_return_pct": "future_close / close - 1",
        "target_volatility": f"standard deviation of next {VOLATILITY_WINDOW} daily returns"
    },
    "parameters": {
        "return_horizon": RETURN_HORIZON,
        "volatility_window": VOLATILITY_WINDOW,
        "model_type": "RandomForest"
    }
}

with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)

print("\nArchivos generados:")

for file in output_dir.iterdir():
    print(file.name)

print("\nProceso finalizado correctamente.")
