# Swiggy Delivery Time Prediction

[![CI](https://github.com/MansiBramta/swiggy-delivery-time-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/MansiBramta/swiggy-delivery-time-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)

Predicts how many minutes a Swiggy food delivery will take, from order, traffic, weather and delivery partner details.

This is an end-to-end project: data cleaning, EDA and hypothesis tests, model comparison and tuning, experiment tracking and a model registry on DagsHub (MLflow), a FastAPI service, Docker, tests, CI, and a deployment on AWS.

> ![Extension popup](reports/figures/extension_popup.png)`

- GitHub: https://github.com/MansiBramta/swiggy-delivery-time-prediction
- Experiments and model registry (DagsHub): https://dagshub.com/MansiBramta/swiggy-delivery-time-prediction

---

## Results

Scores on a held-out test set of 9,101 orders (80/20 split, random seed 42). RMSE and MAE are in minutes.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Always predict the average (floor) | 9.38 | 7.60 | 0.00 |
| Linear regression (best fill method) | 5.93 | 4.73 | 0.60 |
| **Tuned gradient boosting, blanks left as blanks (final)** | **3.90** | **3.10** | **0.83** |

A typical prediction is off by about 3 to 4 minutes, and the model explains about 83% of the variation in delivery time. The 5-fold cross-validation RMSE on the training rows is 3.92, close to the test score.

### How the models compared

5-fold cross-validation RMSE on the training rows (lower is better):

| Model | Blanks | RMSE |
|---|---|---|
| Linear regression | simple fill | 5.951 |
| Linear regression | fill + "was blank" flags | 5.894 |
| Random forest (untuned) | simple fill | 4.130 |
| XGBoost (Optuna) | left as blanks | 3.947 |
| LightGBM (Optuna) | left as blanks | 3.945 |
| CatBoost (Optuna, 7 trials) | text blanks as own category | 3.959 |
| **HistGradientBoosting, scikit-learn (Optuna)** | **left as blanks** | **3.920** |

- The model family mattered far more than anything else. Linear regression stalls near 5.9 because many effects are steps and curves, not straight lines. The boosted trees reach about 3.9.
- The three boosting libraries are within the fold-to-fold spread (about 0.04) of each other, so there is no clear winner among them.
- How blanks are handled changed the score by less than 0.1 minutes in every comparison (drop rows, median / most common fill, fill + flags, KNN fill, native handling).
- Stacking the boosting models gave no real gain on the test set (3.898 vs 3.901), so the final model is a single tuned scikit-learn boosting model.

---

## Dataset

[Food Delivery Dataset](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset) on Kaggle (by Gaurav Malik).

- `train.csv`: 45,593 orders, 20 columns, including the delivery time in minutes (the target). It is saved as `data/raw/swiggy.csv`.
- `test.csv` has no delivery time, so it is not used.
- After cleaning: 45,502 orders and 26 columns. The modelling table has 11 features plus the target.
- The data covers 44 dates (11 to 18 Feb and 1 Mar to 6 Apr 2022).

The raw data is not stored in this repository.

---

## What the analysis found

The full work is in `notebooks/cleaning_eda.ipynb`.

**The target.** Delivery time runs from 10 to 54 minutes (mean 26.3, median 26). It is slightly right skewed (0.49) with two or three bumps, and it is not normal (Jarque-Bera 1,972), because of hard limits at 10 and 54 minutes. The 52 to 54 minute outliers are real deliveries, so they were kept.

**What drives delivery time** (ANOVA effect size, eta²; Kruskal-Wallis agreed on every column):

| Column | eta² | Size |
|---|---|---|
| Number of deliveries | 0.188 | large |
| Traffic | 0.185 | large |
| Time of day | 0.093 | medium |
| Festival | 0.085 | medium |
| Vehicle condition | 0.081 | medium |
| Weather | 0.063 | medium |
| City type | 0.060 | medium |
| Vehicle type | 0.027 | small |
| Day of week | 0.008 | negligible |
| Order type, weekend, pickup minutes | about 0 | none |

**Other findings**
- Traffic and time of day overlap heavily (Cramér's V = 0.60): each time of day only has two of the four traffic levels. Within a time of day, traffic still changes delivery time by 4 to 9 minutes, so traffic is the real driver and time of day mostly acts through it.
- At the same distance, high and jam traffic are equal. The order is low < medium < high ≈ jam.
- Distance has its own effect. Delivery time is flat up to about 9 km, rises between 9 and 15 km, then flattens.
- Delivery time steps up at delivery partner age 30 (about 23 to 29.5 minutes) and steps down at rating 4.5 (about 34.5 to 24 minutes).
- Weather forms three groups: sunny (about 21.9 minutes), stormy / sandstorms / windy (about 26), and cloudy / fog (about 28.9). Weather is independent of traffic.
- Every number here is an association, not a cause.

**Tests used:** Jarque-Bera and Q-Q plots, one-way ANOVA with Levene and Kruskal-Wallis checks, Games-Howell pairwise comparisons, and chi-square tests with Cramér's V.

---

## Cleaning decisions

Cleaning lives in `swiggy_delivery_time_prediction/dataset.py`. The rule: **missing values stay missing**. Nothing is filled with a default.

- The fake text `"NaN "` in the raw file becomes a real blank.
- Text columns are stripped and lowercased.
- Delivery partners aged outside 20 to 39 are dropped (91 rows).
- Impossible values become blanks: ratings outside 1 to 5, `vehicle_condition` outside 0 to 2, and the `bicycle` vehicle type.
- Coordinates are made positive, and anything outside India becomes a blank.
- Derived columns: `distance_km` (straight-line, haversine), `order_hour`, `pickup_time(min)`, `time_of_day`, `day_name`, `month_name`, `is_weekend`, `city_code`.

Rows with a blank `festival`, `city` or `no_of_deliveries` were much faster than average, so they carry information and are kept.

Features used by the model (11): age, rating, weather, traffic, vehicle condition, vehicle type, number of deliveries, festival, city, order hour, distance (km).

---

## Project structure

Built from the [Cookiecutter Data Science](https://github.com/drivendataorg/cookiecutter-data-science) template.

```
├── .github/workflows/ci.yml        <- tests + Docker build on every push
├── Dockerfile                      <- container for the API
├── Makefile
├── README.md
├── data/                           <- not in git
│   ├── raw/                        <- swiggy.csv (original Kaggle train.csv)
│   └── processed/                  <- cleaned data and the modelling table
├── models/                         <- final_model.joblib (not in git)
├── notebooks/
│   ├── cleaning_eda.ipynb          <- cleaning, EDA and hypothesis tests
│   └── modelling.ipynb             <- model comparison, tuning, MLflow logging
├── reports/figures/
├── requirements.txt                <- full environment
├── requirements-api.txt            <- slim, pinned list used in the Docker image
├── tests/
│   ├── test_dataset.py             <- cleaning tests
│   └── test_api.py                 <- API tests
└── swiggy_delivery_time_prediction/
    ├── config.py                   <- paths and settings
    ├── dataset.py                  <- cleaning
    ├── features.py                 <- builds the modelling table
    ├── api.py                      <- FastAPI app
    └── modeling/
        ├── train.py                <- trains the final model, logs to DagsHub
        └── register.py             <- registers it in the DagsHub model registry
```

---

## Getting started

### 1. Set up the environment

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the data

Download `train.csv` from [Kaggle](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset) and save it as `data/raw/swiggy.csv`.

### 3. Create a `.env` file in the project root

```
DAGSHUB_REPO_OWNER=your_dagshub_username
DAGSHUB_REPO_NAME=swiggy-delivery-time-prediction
MODEL_SOURCE=local
```

Never commit this file. It is listed in `.gitignore`.

### 4. Run the pipeline

```bash
python -m swiggy_delivery_time_prediction.dataset          # cleaning        -> data/processed/swiggy_cleaned.csv
python -m swiggy_delivery_time_prediction.features         # modelling table -> data/processed/model_df.csv
python -m swiggy_delivery_time_prediction.modeling.train   # trains and saves models/final_model.joblib
python -m swiggy_delivery_time_prediction.modeling.register  # registers the model on DagsHub
```

`train` logs the run to DagsHub and asks you to log in the first time. To train without logging, add `--no-log-to-dagshub`.

### 5. Run the API

```bash
uvicorn swiggy_delivery_time_prediction.api:app --reload
```

Open http://127.0.0.1:8000/docs for an interactive page where you can try the endpoints.

The model comes from `models/final_model.joblib` by default. To load the `champion` version from the DagsHub registry instead, set `MODEL_SOURCE=registry` in `.env`. If the registry cannot be reached, the API falls back to the local file and says which one it used at `/health`.

### 6. Run it in Docker

```bash
docker build -t swiggy-api .
docker run -p 8000:8000 swiggy-api
```

The image copies `models/final_model.joblib`, so run the training step first.

### 7. Run the tests

```bash
pip install pytest httpx
python -m pytest -v
```

The same tests run in GitHub Actions on every push, followed by a Docker build and a smoke test of the running container (using a stub model, since the real model file is not in git).

---

## API

### `GET /health`

```json
{"status": "ok", "model_source": "local"}
```

### `POST /predict`

Every field is optional. A missing value is treated as a blank, the same way the model saw blanks in training.

| Field | Allowed values |
|---|---|
| `age` | 18 to 70 |
| `rating` | 1 to 5 |
| `weather` | `fog`, `stormy`, `cloudy`, `sandstorms`, `windy`, `sunny` |
| `traffic` | `low`, `medium`, `high`, `jam` |
| `vehicle_condition` | `0`, `1`, `2` |
| `vehicle_type` | `motorcycle`, `scooter`, `electric_scooter` |
| `no_of_deliveries` | 0 to 3 |
| `festival` | `yes`, `no` |
| `city` | `metropolitian`, `urban`, `semi-urban` (the dataset's spelling) |
| `order_hour` | 0 to 23 |
| `distance_km` | 0 or more |

Shortcuts, used only when `distance_km` or `order_hour` is not given:

- `restaurant_latitude`, `restaurant_longitude`, `delivery_latitude`, `delivery_longitude` give the distance.
- `order_time` (`"HH:MM"`) gives the order hour.

Example request:

```json
{
  "age": 30, "rating": 4.7, "weather": "sunny", "traffic": "low",
  "vehicle_condition": 1, "vehicle_type": "motorcycle", "no_of_deliveries": 1,
  "festival": "no", "city": "metropolitian", "order_hour": 22, "distance_km": 9
}
```

Example response for the request above:

```json
{"predicted_delivery_time_min": 16.2}
```

Values outside the allowed ranges return a `422` error.

### Sanity check

Changing one input at a time for the example order above (predictions in minutes):

| Input changed | Predictions |
|---|---|
| Number of deliveries: 0, 1, 2, 3 | 14.6, 16.2, 33.9, 41.2 |
| Distance: 3 km, 9 km, 15 km | 15.6, 16.2, 23.7 |
| Order hour: 0 to 10 vs 11 to 23 | 16.0 vs 16.2 |

More deliveries and longer distances increase the prediction, as in the analysis (the distance effect is flat up to about 9 km, then rises). The order hour barely matters once traffic and distance are fixed, because time of day mostly acts through traffic. These are for one example order, not averages.

---

## Deployment on AWS

The API runs on AWS in the Asia Pacific (Sydney) region, `ap-southeast-2`.

- **Amazon S3** stores the trained model file (`final_model.joblib`, about 2.7 MB), so the model is kept outside the code repository.
- **Amazon EC2** hosts the API and serves predictions over HTTP.
- **DagsHub** also holds the final model in its model registry (`swiggy-delivery-time-model`, alias `champion`), together with every experiment run.

The public address of an EC2 instance can change when it is stopped and started, so no fixed URL is listed here. To try the API yourself, follow "Run the API" or "Run it in Docker" above.


---

## Limitations

- **Association, not cause.** The analysis shows what goes with longer deliveries, not what makes them longer.
- **The data may be partly simulated.** Several patterns are unusually clean: sharp steps at age 30 and rating 4.5, near-perfectly even category counts, festival orders alternating between high and low on consecutive days, and distance changing in blocks by hour. A real delivery dataset would probably be harder to predict, so these scores may not carry over.
- **Short time span.** The data covers 44 dates, so seasonal effects cannot be studied, and month comparisons are not meaningful.
- **Small groups.** `semi-urban` (164 orders), festival days (894) and orders with 3 deliveries (361) look extreme but are small, so conclusions about them are tentative.
- **The test set was looked at several times** while comparing methods, so the cross-validation score (3.92) is the cleaner estimate.
- **Distance is straight-line**, not road distance.

---

## Tech stack

Python 3.12, pandas, NumPy, SciPy, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, MLflow and DagsHub, FastAPI and uvicorn, pytest, Docker, GitHub Actions, AWS (EC2 and S3).

---

## Author

Mansi Bramta · [GitHub: MansiBramta](https://github.com/MansiBramta)