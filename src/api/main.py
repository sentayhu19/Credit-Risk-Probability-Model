"""FastAPI service exposing model predictions."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pathlib import Path
import pandas as pd
import mlflow
from src.api.pydantic_models import PredictionResponse, Transaction
from src.data_processing import engineer_features

MODEL_NAME = "credit-risk-best"  # MLflow registered model name
MODEL_STAGE = "Production"
PICKLE_PATH = Path("artifacts/best_model.pkl")
app = FastAPI(title="Credit Risk Scoring API", version="0.1.0")


@app.on_event("startup")
def load_model() -> None:
    # Try Production stage first, then latest version, then local pickle
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        app.state.model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        try:
            model_uri = f"models:/{MODEL_NAME}/latest"
            app.state.model = mlflow.sklearn.load_model(model_uri)
        except Exception:
            if PICKLE_PATH.exists():
                import joblib
                app.state.model = joblib.load(PICKLE_PATH)
            else:
                raise RuntimeError("No suitable model found in MLflow or artifacts.")


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(tx: Transaction):
    df = pd.DataFrame([tx.dict()])

    # ensure target column exists for feature pipeline
    if "FraudResult" not in df.columns:
        df["FraudResult"] = 0  

    X_raw, _ = engineer_features(df)

    num_cols = X_raw.select_dtypes(include=["number"]).columns.tolist()
    low_card_cols = [c for c in X_raw.columns if c not in num_cols and X_raw[c].nunique() <= 50]
    X_enc = pd.get_dummies(X_raw[low_card_cols], drop_first=True)
    X_proc = pd.concat([X_raw[num_cols], X_enc], axis=1)

    model = app.state.model
    ref_cols = getattr(model, "feature_names_in_", None)
    if ref_cols is not None:
        missing = [c for c in ref_cols if c not in X_proc.columns]
        for c in missing:
            X_proc[c] = 0
        X_proc = X_proc[ref_cols]

    prob = float(model.predict_proba(X_proc)[:, 1][0])
    return PredictionResponse(risk_probability=prob)
