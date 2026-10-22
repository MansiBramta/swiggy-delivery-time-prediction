from fastapi.testclient import TestClient
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from swiggy_delivery_time_prediction import api
from swiggy_delivery_time_prediction.config import CAT_COLS, NUM_COLS

GOOD_ORDER = {
    "age": 30, "rating": 4.7, "weather": "sunny", "traffic": "low",
    "vehicle_condition": 1, "vehicle_type": "motorcycle", "no_of_deliveries": 1,
    "festival": "no", "city": "metropolitian", "order_hour": 19, "distance_km": 9,
}


@pytest.fixture()
def client():
    # a stand-in model that always says 25 minutes (the real model file is not in git)
    stub = DummyRegressor(strategy="constant", constant=25.0).fit([[0]], [25.0])
    api._state.clear()
    api._state.update(model=stub, source="test")
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_source": "test"}


def test_predict_full_order(client):
    r = client.post("/predict", json=GOOD_ORDER)
    assert r.status_code == 200
    assert r.json()["predicted_delivery_time_min"] == 25.0


def test_predict_with_everything_blank(client):
    assert client.post("/predict", json={}).status_code == 200


@pytest.mark.parametrize("bad", [
    {"traffic": "foggy"}, {"rating": 7}, {"order_hour": 25},
    {"vehicle_condition": 3}, {"no_of_deliveries": 9}, {"order_time": "25:99"},
])
def test_bad_values_are_rejected(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_values_become_nan():
    row = api.Order().to_row()
    assert set(row) == set(NUM_COLS + CAT_COLS)
    assert all(np.isnan(v) for v in row.values())


def test_distance_from_coordinates():
    row = api.Order(restaurant_latitude=22.745049, restaurant_longitude=75.892471,
                    delivery_latitude=22.765049, delivery_longitude=75.912471).to_row()
    assert row["distance_km"] == pytest.approx(3.02, abs=0.05)


def test_given_distance_wins_over_coordinates():
    row = api.Order(distance_km=5, restaurant_latitude=22.7, restaurant_longitude=75.9,
                    delivery_latitude=22.8, delivery_longitude=76.0).to_row()
    assert row["distance_km"] == 5


def test_order_time_gives_order_hour():
    assert api.Order(order_time="19:45").to_row()["order_hour"] == 19