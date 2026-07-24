import joblib
import pandas as pd
from financial_api.train import MODEL_PATH, FEATURE_COLS
from financial_api.features import PROCESSED_DATA_PATH

def make_prediction(symbol: str) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("El artefacto del modelo no se encuentra.")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Obtener el último registro disponible para el símbolo solicitado
    symbol_df = df[df["symbol"] == symbol.upper()].sort_values("date")
    
    if symbol_df.empty:
        raise ValueError(f"Símbolo '{symbol}' no disponible en los datos procesados.")

    latest_features = symbol_df[FEATURE_COLS].iloc[[-1]]
    
    prob_up = float(model.predict_proba(latest_features)[0][1])
    pred_label = "up" if prob_up >= 0.5 else "down"

    return {
        "symbol": symbol.upper(),
        "prediction": pred_label,
        "probability_up": round(prob_up, 2),
        "model_version": "random_forest_v1",
        "prediction_horizon": "next_day"
    }