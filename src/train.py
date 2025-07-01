"""Model training script.

Usage:
    python -m src.train --raw-path data/raw/transactions.csv --model-out artifacts/model.pkl
"""

from __future__ import annotations

import argparse
import os
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import mlflow
from sklearn.model_selection import train_test_split

from src.data_processing import engineer_features, load_raw

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit-risk model")
    parser.add_argument("--raw-path", type=Path, required=True, help="Path to raw CSV data")
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/model.pkl"))
    return parser.parse_args()



MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "credit-risk")


def main() -> None:
    args = parse_args()
    df_raw = load_raw(args.raw_path)
    X, y = engineer_features(df_raw)

    # basic preprocessing: numeric + low-cardinality categoricals (<=50 unique)
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    low_card_cols = [c for c in X.columns if c not in num_cols and X[c].nunique() <= 50]
    logger.info("Encoding %d low-cardinality categoricals out of %d total columns", len(low_card_cols), X.shape[1])
    X_enc = pd.get_dummies(X[low_card_cols], drop_first=True)
    X = pd.concat([X[num_cols], X_enc], axis=1)
    logger.info("After preprocessing: %s features", X.shape[1])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # define model grid
    models = {
        "logreg": (
            LogisticRegression(max_iter=1000, solver="lbfgs"),
            {"C": [0.1, 1, 10]},
        ),
        "rf": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 300], "max_depth": [None, 10]},
        ),
        "gbm": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1]},
        ),
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    best_auc = -1.0
    best_estimator = None
    best_name = ""

    for name, (estimator, param_grid) in models.items():
        with mlflow.start_run(run_name=name):
            logger.info("Running GridSearch for %s", name)
            search = GridSearchCV(
                estimator,
                param_grid,
                scoring="roc_auc",
                cv=3,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_

            y_pred = best.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            logger.info("%s validation AUC = %.4f", name, auc)

            # log params & metrics
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("val_auc", auc)

            if auc > best_auc:
                best_auc = auc
                best_estimator = best
                best_name = name
            mlflow.sklearn.log_model(best, artifact_path="model")

    logger.info("Best model: %s (AUC=%.4f)", best_name, best_auc)

    # register best model
    with mlflow.start_run(run_name=f"register_{best_name}") as run:
        mlflow.sklearn.log_model(best_estimator, "model", registered_model_name="credit-risk-best")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_estimator, args.model_out)
    logger.info("Saved best model to %s", args.model_out)


if __name__ == "__main__":
    main()
