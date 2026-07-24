from pathlib import Path
import pandas as pd
from financial_api.data import RAW_DATA_PATH

PROCESSED_DATA_PATH = Path("data/processed/processed_data.csv")

def build_features() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError("Primero ejecute data.py para generar el dataset raw.")

    df = pd.read_csv(RAW_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    processed_dfs = []
    for sym, group in df.groupby("symbol"):
        g = group.copy()
        g["return_daily"] = g["close"].pct_change()
        g["sma_10"] = g["close"].rolling(10).mean()
        g["sma_50"] = g["close"].rolling(50).mean()
        g["volatility_10d"] = g["return_daily"].rolling(10).std()
        
        # Target: Clasificación de tendencia (1 si el precio de mañana sube, 0 si baja/igualmente)
        g["target_up"] = (g["close"].shift(-1) > g["close"]).astype(int)
        
        processed_dfs.append(g)

    final_df = pd.concat(processed_dfs).dropna().reset_index(drop=True)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Features procesadas guardadas en {PROCESSED_DATA_PATH}.")
    return final_df

if __name__ == "__main__":
    build_features()