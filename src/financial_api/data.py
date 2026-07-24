from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

RAW_DATA_PATH = Path("data/raw/market_data.csv")
SYMBOLS = ["AAPL", "MSFT", "NVDA"]

def fetch_data(start_date: str = "2020-01-01", end_date: str = "2024-01-01") -> pd.DataFrame:
    """Descarga datos de yfinance o genera datos de respaldo si falla la red."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Descargando datos para {SYMBOLS} desde yfinance...")
        df = yf.download(SYMBOLS, start=start_date, end=end_date)["Close"]
        df = df.reset_index().melt(id_vars=["Date"], var_name="symbol", value_name="close")
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna()
    except Exception as e:
        print(f"⚠️ Error conectando a yfinance ({e}). Generando datos de respaldo...")
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        records = []
        for sym in SYMBOLS:
            base_price = 150.0
            prices = base_price + np.cumsum(np.random.normal(0.1, 2.0, size=len(dates)))
            for d, p in zip(dates, prices):
                records.append({"date": d, "symbol": sym, "close": max(p, 10.0)})
        df = pd.DataFrame(records)

    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"✅ Datos guardados en {RAW_DATA_PATH} ({len(df)} registros).")
    return df

if __name__ == "__main__":
    fetch_data()