import json
from fastapi import FastAPI, HTTPException
import pandas as pd
from financial_api.schemas import (
    HealthResponse, MarketDataResponse, PredictRequest,
    PredictResponse, ModelMetadataResponse
)
from financial_api.train import MODEL_PATH, METADATA_PATH
from financial_api.features import PROCESSED_DATA_PATH
from financial_api.predict import make_prediction
from statistics import mean

app = FastAPI(
    title="API de Señales Financieras Educativa",
    description="Herramienta académica de inferencia. No constituye asesoría ni recomendación de inversión.",
    version="1.0.0"
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists()
    }

@app.get("/market-data/{symbol}", response_model=MarketDataResponse)
def get_market_data(symbol: str):
    if not PROCESSED_DATA_PATH.exists():
        raise HTTPException(status_code=500, detail="Datos no disponibles.")
    
    df = pd.read_csv(PROCESSED_DATA_PATH)
    sym_df = df[df["symbol"] == symbol.upper()].sort_values("date")
    
    if sym_df.empty:
        raise HTTPException(status_code=404, detail=f"Símbolo {symbol} no encontrado.")
        
    latest = sym_df.iloc[-1]
    return {
        "symbol": symbol.upper(),
        "latest_date": str(latest["date"]),
        "close": round(float(latest["close"]), 2),
        "return_daily": round(float(latest["return_daily"]), 4)
    }


@app.get("/market-data-average/{symbol}")
def get_stats(symbol: str):
    df = pd.read_csv(PROCESSED_DATA_PATH)

    sym_df = df[df["symbol"] == symbol.upper()]

    if sym_df.empty:
        raise HTTPException(status_code=404, detail="Símbolo no encontrado")

    return {
        "symbol": symbol.upper(),
        "average_close": round(sym_df["close"].mean(), 2),
        "min_close": round(sym_df["close"].min(), 2),
        "max_close": round(sym_df["close"].max(), 2),
        "latest_close": round(sym_df.iloc[-1]["close"], 2)
    }

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        res = make_prediction(payload.symbol)
        return res
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/model/metadata", response_model=ModelMetadataResponse)
def get_metadata():
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Metadatos no encontrados.")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data