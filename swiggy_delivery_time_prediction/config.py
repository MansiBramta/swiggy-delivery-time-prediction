from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# folders
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# files
RAW_FILE = RAW_DATA_DIR / "swiggy.csv"
CLEANED_FILE = PROCESSED_DATA_DIR / "swiggy_cleaned.csv"
MODEL_TABLE_FILE = PROCESSED_DATA_DIR / "model_df.csv"
MODEL_FILE = MODELS_DIR / "final_model.joblib"

# settings
TARGET = "total_time_taken"
RANDOM_SEED = 42
TEST_SIZE = 0.2

NUM_COLS = ["age", "rating", "no_of_deliveries", "order_hour", "distance_km"]
CAT_COLS = ["weather", "traffic", "vehicle_type", "festival", "city", "vehicle_condition"]

DROP_COLS = [
    "order_type", "pickup_time(min)", "is_weekend", "day_name", "time_of_day",
    "month_name", "city_code", "restaurant_latitude", "restaurant_longitude",
    "location_latitude", "location_longitude", "date", "time_of_order", "pickup_time",
]