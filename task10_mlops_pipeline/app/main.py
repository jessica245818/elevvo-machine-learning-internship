"""FastAPI service for the Task 9 predictive-maintenance model."""

from __future__ import annotations

import math
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        Path(__file__).resolve().parents[1] / "model" / "selected_alarm_model.pkl",
    )
)
StrictNumber = Annotated[float, Field(strict=True)]


class SensorReading(BaseModel):
    """Validated raw sensor values accepted by the model service."""

    model_config = ConfigDict(extra="forbid")

    product_type: Literal["L", "M", "H"] = Field(
        description="AI4I product quality type."
    )
    air_temperature_k: StrictNumber = Field(ge=250, le=350)
    process_temperature_k: StrictNumber = Field(ge=250, le=400)
    rotational_speed_rpm: StrictNumber = Field(gt=0, le=10_000)
    torque_nm: StrictNumber = Field(ge=0, le=500)
    tool_wear_min: StrictNumber = Field(ge=0, le=1_000)


class PredictionResponse(BaseModel):
    model_name: str
    failure_probability: float
    alarm_threshold: float
    machine_failure_alarm: bool
    decision: Literal["maintenance_required", "normal_operation"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_name: str


def engineer_features(readings: list[SensorReading]) -> pd.DataFrame:
    """Reproduce the Task 9 feature engineering without accepting derived inputs."""
    rows = []
    for reading in readings:
        angular_speed = reading.rotational_speed_rpm * 2 * math.pi / 60
        rows.append(
            {
                "Type": reading.product_type,
                "Air temperature [K]": reading.air_temperature_k,
                "Process temperature [K]": reading.process_temperature_k,
                "Rotational speed [rpm]": reading.rotational_speed_rpm,
                "Torque [Nm]": reading.torque_nm,
                "Tool wear [min]": reading.tool_wear_min,
                "Temperature difference [K]": (
                    reading.process_temperature_k - reading.air_temperature_k
                ),
                "Mechanical power [W]": angular_speed * reading.torque_nm,
                "Tool stress": reading.torque_nm * reading.tool_wear_min,
                "Torque-speed ratio": (
                    reading.torque_nm / reading.rotational_speed_rpm
                ),
            }
        )
    return pd.DataFrame(rows)


def load_model_bundle(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Model artifact not found: {path}")
    with path.open("rb") as source:
        bundle = pickle.load(source)
    required = {"model", "threshold", "model_name", "features"}
    missing = required.difference(bundle)
    if missing:
        raise RuntimeError(f"Invalid model bundle; missing keys: {sorted(missing)}")
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = load_model_bundle(MODEL_PATH)
    yield


app = FastAPI(
    title="Industrial Predictive Maintenance API",
    version="1.0.0",
    description=(
        "Serves the Task 9 low-false-discovery Random Forest alarm model."
    ),
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return the assignment-required 400 instead of FastAPI's default 422."""
    errors = []
    for error in exc.errors():
        cleaned = {
            key: value
            for key, value in error.items()
            if key not in {"input", "ctx"}
        }
        errors.append(cleaned)
    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_request",
            "message": "Request data failed schema validation.",
            "details": errors,
        },
    )


def predict_readings(
    readings: list[SensorReading], model_bundle: dict
) -> list[PredictionResponse]:
    features = engineer_features(readings)
    expected_features = model_bundle["features"]
    if list(features.columns) != expected_features:
        raise RuntimeError("Serving features do not match the trained model contract.")
    probabilities = model_bundle["model"].predict_proba(features)[:, 1]
    threshold = float(model_bundle["threshold"])
    responses = []
    for probability in probabilities:
        probability = float(np.clip(probability, 0, 1))
        alarm = probability >= threshold
        responses.append(
            PredictionResponse(
                model_name=model_bundle["model_name"],
                failure_probability=round(probability, 6),
                alarm_threshold=threshold,
                machine_failure_alarm=alarm,
                decision=(
                    "maintenance_required" if alarm else "normal_operation"
                ),
            )
        )
    return responses


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health(request: Request) -> HealthResponse:
    bundle = request.app.state.model_bundle
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=bundle["model_name"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(reading: SensorReading, request: Request) -> PredictionResponse:
    return predict_readings([reading], request.app.state.model_bundle)[0]


@app.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
    tags=["inference"],
)
def predict_batch(
    readings: Annotated[list[SensorReading], Field(min_length=1, max_length=1_000)],
    request: Request,
) -> list[PredictionResponse]:
    return predict_readings(readings, request.app.state.model_bundle)
