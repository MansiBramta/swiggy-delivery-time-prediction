from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from swiggy_delivery_time_prediction.config import CLEANED_FILE, DROP_COLS, MODEL_TABLE_FILE

app = typer.Typer()


def build_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the 12 columns used for modelling (blanks stay blank)."""
    return df.drop(columns=DROP_COLS)


@app.command()
def main(input_path: Path = CLEANED_FILE, output_path: Path = MODEL_TABLE_FILE):
    logger.info(f"Reading cleaned data from {input_path}")
    model_df = build_model_table(pd.read_csv(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(output_path, index=False)
    logger.success(f"Saved modelling table {model_df.shape} to {output_path}")


if __name__ == "__main__":
    app()