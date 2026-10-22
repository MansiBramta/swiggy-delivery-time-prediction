import pandas as pd
import pytest

from swiggy_delivery_time_prediction.dataset import clean_data


def make_raw() -> pd.DataFrame:
    base = {
        "ID": "0x1", "Delivery_person_ID": "INDORES13DEL02",
        "Delivery_person_Age": "37", "Delivery_person_Ratings": "4.9",
        "Restaurant_latitude": 22.745049, "Restaurant_longitude": 75.892471,
        "Delivery_location_latitude": 22.765049, "Delivery_location_longitude": 75.912471,
        "Order_Date": "19-03-2022", "Time_Orderd": "11:30:00", "Time_Order_picked": "11:45:00",
        "Weatherconditions": "conditions Sunny", "Road_traffic_density": "High ",
        "Vehicle_condition": 2, "Type_of_order": "Snack ", "Type_of_vehicle": "motorcycle ",
        "multiple_deliveries": "0", "Festival": "No ", "City": "Urban ",
        "Time_taken(min)": "(min) 24",
    }
    rows = [dict(base) for _ in range(5)]
    # row 1: blanks written as the fake text "NaN "
    rows[1].update({"Delivery_person_Age": "NaN ", "Delivery_person_Ratings": "NaN ",
                    "Time_Orderd": "NaN ", "Weatherconditions": "conditions NaN",
                    "Road_traffic_density": "NaN ", "multiple_deliveries": "NaN ",
                    "Festival": "NaN ", "City": "NaN "})
    # row 2: impossible values
    rows[2].update({"Delivery_person_Ratings": "6.0", "Vehicle_condition": 3,
                    "Type_of_vehicle": "bicycle ", "Restaurant_latitude": 0.0,
                    "Restaurant_longitude": 0.0})
    # row 3: age 15, so the whole row is dropped
    rows[3].update({"Delivery_person_Age": "15"})
    # row 4: negative restaurant coordinates, made positive
    rows[4].update({"Restaurant_latitude": -22.745049, "Restaurant_longitude": -75.892471})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cleaned():
    return clean_data(make_raw())


def test_only_the_age_15_row_is_dropped(cleaned):
    assert len(cleaned) == 4


def test_expected_columns(cleaned):
    for col in ["age", "rating", "traffic", "weather", "distance_km", "order_hour",
                "time_of_day", "total_time_taken", "city_code", "is_weekend"]:
        assert col in cleaned.columns
    assert cleaned.shape[1] == 26


def test_normal_row_is_cleaned(cleaned):
    row = cleaned.iloc[0]
    assert row["traffic"] == "high"
    assert row["weather"] == "sunny"
    assert row["city"] == "urban"
    assert row["festival"] == "no"
    assert row["vehicle_type"] == "motorcycle"
    assert row["city_code"] == "indo"
    assert row["total_time_taken"] == 24
    assert row["order_hour"] == 11
    assert row["pickup_time(min)"] == 15
    assert row["time_of_day"] == "morning"
    assert row["day_name"] == "saturday"
    assert bool(row["is_weekend"]) is True
    assert row["distance_km"] == pytest.approx(3.02, abs=0.05)


def test_blanks_stay_blank(cleaned):
    row = cleaned.iloc[1]
    for col in ["age", "rating", "traffic", "weather", "festival", "city",
                "no_of_deliveries", "order_hour", "pickup_time(min)", "time_of_day"]:
        assert pd.isna(row[col]), f"{col} should be blank"
    assert row["vehicle_type"] == "motorcycle"


def test_nothing_is_filled(cleaned):
    assert cleaned["age"].isna().sum() == 1
    assert cleaned["rating"].isna().sum() == 2     # one blank, one impossible (6.0)


def test_impossible_values_become_blank(cleaned):
    row = cleaned.iloc[2]
    assert pd.isna(row["rating"])
    assert pd.isna(row["vehicle_condition"])
    assert pd.isna(row["vehicle_type"])
    assert pd.isna(row["restaurant_latitude"])
    assert pd.isna(row["distance_km"])


def test_negative_coordinates_are_made_positive(cleaned):
    assert cleaned.iloc[3]["restaurant_latitude"] > 0
    assert cleaned.iloc[3]["distance_km"] == pytest.approx(cleaned.iloc[0]["distance_km"])