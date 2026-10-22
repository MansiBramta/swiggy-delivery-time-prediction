import os
from typing import Literal, Optional

from fastapi import FastAPI
import joblib
from loguru import logger
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from swiggy_delivery_time_prediction.config import CAT_COLS, MODEL_FILE, NUM_COLS
from swiggy_delivery_time_prediction.dataset import haversine

app = FastAPI(title="Swiggy delivery time API")

MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")      # "local" or "registry"
REGISTRY_URI = "models:/swiggy-delivery-time-model@champion"


def load_model():
    if MODEL_SOURCE == "registry":
        try:
            # imported here so the Docker image doesn't need these libraries
            import dagshub
            import mlflow.sklearn

            dagshub.init(
                repo_owner=os.getenv("DAGSHUB_REPO_OWNER"),
                repo_name=os.getenv("DAGSHUB_REPO_NAME", "swiggy-delivery-time-prediction"),
                mlflow=True,
            )
            m = mlflow.sklearn.load_model(REGISTRY_URI)
            logger.success(f"Loaded model from the registry: {REGISTRY_URI}")
            return m, "registry"
        except Exception as e:
            logger.warning(f"Could not load from the registry ({type(e).__name__}: {e}). Using the local file.")
    m = joblib.load(MODEL_FILE)
    logger.info(f"Loaded model from {MODEL_FILE}")
    return m, "local"


_state: dict = {}


def get_model():
    """Load the model on first use (so tests can run without the model file)."""
    if "model" not in _state:
        _state["model"], _state["source"] = load_model()
    return _state["model"], _state["source"]


class Order(BaseModel):
    # every field is optional: a missing value is treated as a blank, as in training
    age: Optional[float] = Field(None, ge=18, le=70)
    rating: Optional[float] = Field(None, ge=1, le=5)
    weather: Optional[Literal["fog", "stormy", "cloudy", "sandstorms", "windy", "sunny"]] = None
    traffic: Optional[Literal["low", "medium", "high", "jam"]] = None
    vehicle_condition: Optional[Literal[0, 1, 2]] = None
    vehicle_type: Optional[Literal["motorcycle", "scooter", "electric_scooter"]] = None
    no_of_deliveries: Optional[int] = Field(None, ge=0, le=3)
    festival: Optional[Literal["yes", "no"]] = None
    city: Optional[Literal["metropolitian", "urban", "semi-urban"]] = None
    order_hour: Optional[int] = Field(None, ge=0, le=23)
    distance_km: Optional[float] = Field(None, ge=0)

    # optional shortcuts: used only if distance_km / order_hour are not given
    restaurant_latitude: Optional[float] = None
    restaurant_longitude: Optional[float] = None
    delivery_latitude: Optional[float] = None
    delivery_longitude: Optional[float] = None
    order_time: Optional[str] = Field(None, pattern=r"^([01]?\d|2[0-3]):[0-5]\d$")

    def to_row(self) -> dict:
        d = self.model_dump()
        coords = [d["restaurant_latitude"], d["restaurant_longitude"],
                  d["delivery_latitude"], d["delivery_longitude"]]
        if d["distance_km"] is None and all(c is not None for c in coords):
            d["distance_km"] = float(haversine(*coords))
        if d["order_hour"] is None and d["order_time"]:
            d["order_hour"] = int(d["order_time"].split(":")[0])
        row = {c: d[c] for c in NUM_COLS + CAT_COLS}
        return {k: (np.nan if v is None else v) for k, v in row.items()}


@app.get("/health")
def health():
    _, source = get_model()
    return {"status": "ok", "model_source": source}


@app.post("/predict")
def predict(order: Order):
    X = pd.DataFrame([order.to_row()])
    float_cols = NUM_COLS + ["vehicle_condition"]
    X[float_cols] = X[float_cols].astype(float)
    model, _ = get_model()
    pred = float(model.predict(X)[0])
    return {"predicted_delivery_time_min": round(pred, 1)}