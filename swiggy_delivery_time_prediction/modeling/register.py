import os
import tempfile
from pathlib import Path

import dagshub
import joblib
from loguru import logger
import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
import typer

from swiggy_delivery_time_prediction.config import MODEL_FILE

app = typer.Typer()

MODEL_NAME = "swiggy-delivery-time-model"

TRUSTED_TYPES = [
    "functools.partial",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
]


@app.command()
def main(model_path: Path = MODEL_FILE, alias: str = "champion"):
    owner = os.getenv("DAGSHUB_REPO_OWNER")
    if not owner:
        raise typer.BadParameter("Add DAGSHUB_REPO_OWNER=your_username to the .env file")
    dagshub.init(
        repo_owner=owner,
        repo_name=os.getenv("DAGSHUB_REPO_NAME", "swiggy-delivery-time-prediction"),
        mlflow=True,
    )
    mlflow.set_experiment("swiggy-delivery-time")

    model = joblib.load(model_path)

    with mlflow.start_run(run_name="register_final_model") as run:
        try:
            # route A: log the model and register it in one call
            mlflow.sklearn.log_model(
                model, name="model", registered_model_name=MODEL_NAME,
                skops_trusted_types=TRUSTED_TYPES,
            )
            logger.success("Route A worked: logged and registered with log_model")
        except Exception as e:
            logger.warning(f"Route A failed ({type(e).__name__}: {e}). Trying route B.")
            # route B: upload the saved model folder, then register it
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / "model"
                mlflow.sklearn.save_model(model, str(local), skops_trusted_types=TRUSTED_TYPES)
                mlflow.log_artifacts(str(local), artifact_path="model")
            mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
            logger.success("Route B worked: uploaded the folder and registered it")

    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest = max(int(v.version) for v in versions)
        logger.info(f"{MODEL_NAME} has {len(versions)} version(s); the latest is {latest}")
        client.set_registered_model_alias(MODEL_NAME, alias, str(latest))
        logger.success(f"Alias '{alias}' now points to version {latest}")
    except Exception as e:
        logger.warning(f"Could not set the alias ({e}). You can set it in the MLflow page on DagsHub.")


if __name__ == "__main__":
    app()