from typing import List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    model_loaded: bool = Field(..., example=True)


class MarketDataResponse(BaseModel):
    symbol: str = Field(..., example="AAPL")
    latest_date: str = Field(..., example="2023-12-29")
    close: float = Field(..., example=192.53)
    return_daily: float = Field(..., example=0.002)


class PredictRequest(BaseModel):
    symbol: str = Field(..., example="AAPL")
    prediction_horizon: int = Field(default=1, example=1)
    use_cached_data: bool = Field(default=True, example=True)


class PredictResponse(BaseModel):
    symbol: str = Field(..., example="AAPL")
    prediction: str = Field(..., example="up")
    probability_up: float = Field(..., example=0.63)
    model_version: str = Field(..., example="random_forest_v1")
    prediction_horizon: str = Field(..., example="next_day")


class ModelMetadataResponse(BaseModel):
    model_version: str
    training_date: str
    symbols_used: List[str]
    accuracy_metric: float
    prediction_horizon: str
    features: List[str]
