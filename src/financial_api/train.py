from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from financial_api.features import PROCESSED_DATA_PATH, build_features

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

FEATURE_COLS = ["return_daily", "sma_10", "sma_50", "volatility_10d"]

def train_model():
    if not PROCESSED_DATA_PATH.exists():
        build_features()

    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df[FEATURE_COLS]
    y = df["target_up"]

    # División por serie de tiempo simple (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = round(float(accuracy_score(y_test, y_pred)), 4)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_version": "random_forest_v1",
        "training_date": datetime.now().isoformat(),
        "symbols_used": ["AAPL", "MSFT", "NVDA"],
        "accuracy_metric": acc,
        "prediction_horizon": "next_day",
        "features": FEATURE_COLS
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Modelo entrenado con Accuracy: {acc}. Artefactos guardados.")

if __name__ == "__main__":
    train_model()