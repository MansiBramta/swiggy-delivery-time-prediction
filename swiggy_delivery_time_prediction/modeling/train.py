import os
from pathlib import Path

import dagshub
import joblib
from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import typer

from swiggy_delivery_time_prediction.config import (
    CAT_COLS,
    MODEL_FILE,
    MODEL_TABLE_FILE,
    NUM_COLS,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
)

app = typer.Typer()

# best settings found with Optuna (5-fold CV RMSE about 3.92)
BEST_PARAMS = dict(
    learning_rate=0.027400385249378316,
    max_iter=500,
    max_leaf_nodes=110,
    min_samples_leaf=10,
    l2_regularization=0.19861366859847862,
)


def build_model(params: dict) -> Pipeline:
    prep = ColumnTransformer([
        ("num", "passthrough", NUM_COLS),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CAT_COLS),
    ])
    hgb = HistGradientBoostingRegressor(
        categorical_features=[False] * len(NUM_COLS) + [True] * len(CAT_COLS),
        random_state=RANDOM_SEED,
        **params,
    )
    return Pipeline([("prep", prep), ("hgb", hgb)])


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


@app.command()
def main(
    input_path: Path = MODEL_TABLE_FILE,
    model_path: Path = MODEL_FILE,
    log_to_dagshub: bool = True,
):
    df = pd.read_csv(input_path)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    logger.info(f"Train {X_train.shape}, test {X_test.shape}")

    model = build_model(BEST_PARAMS)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_rmse = float(-cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
    ).mean())
    logger.info(f"5-fold CV RMSE: {cv_rmse:.3f}")

    model.fit(X_train, y_train)
    metrics = evaluate(y_test, model.predict(X_test))
    logger.info(f"Test RMSE {metrics['rmse']:.3f} | MAE {metrics['mae']:.3f} | R2 {metrics['r2']:.3f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.success(f"Saved model to {model_path}")

    if log_to_dagshub:
        owner = os.getenv("DAGSHUB_REPO_OWNER")
        if not owner:
            raise typer.BadParameter("Add DAGSHUB_REPO_OWNER=your_username to the .env file")
        dagshub.init(
            repo_owner=owner,
            repo_name=os.getenv("DAGSHUB_REPO_NAME", "swiggy-delivery-time-prediction"),
            mlflow=True,
        )
        mlflow.set_experiment("swiggy-delivery-time")
        with mlflow.start_run(run_name="hgb_final_script"):
            mlflow.log_params(BEST_PARAMS)
            mlflow.log_param("blank_handling", "kept as blanks")
            mlflow.log_metrics({"cv_rmse": cv_rmse, **{f"test_{k}": v for k, v in metrics.items()}})
            mlflow.log_artifact(str(model_path))
        logger.success("Logged the run to DagsHub")


if __name__ == "__main__":
    app()