from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from swiggy_delivery_time_prediction.config import CLEANED_FILE, RAW_FILE

app = typer.Typer()


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # 1. fake "NaN " text -> real blanks
    df = df.replace(r'^\s*NaN\s*$', np.nan, regex=True)

    # 2. city code from delivery person ID, then drop ID columns
    df['city_code'] = df['Delivery_person_ID'].str.strip().str.lower().str[:4]
    df = df.drop(columns=['ID', 'Delivery_person_ID'])

    # 3. rename columns
    df = df.rename(columns={
        'Delivery_person_Age': 'age',
        'Delivery_person_Ratings': 'rating',
        'Restaurant_latitude': 'restaurant_latitude',
        'Restaurant_longitude': 'restaurant_longitude',
        'Delivery_location_latitude': 'location_latitude',
        'Delivery_location_longitude': 'location_longitude',
        'Order_Date': 'date',
        'Time_Orderd': 'time_of_order',
        'Time_Order_picked': 'pickup_time',
        'Weatherconditions': 'weather',
        'Road_traffic_density': 'traffic',
        'Vehicle_condition': 'vehicle_condition',
        'Type_of_order': 'order_type',
        'Type_of_vehicle': 'vehicle_type',
        'multiple_deliveries': 'no_of_deliveries',
        'Festival': 'festival',
        'City': 'city',
        'Time_taken(min)': 'total_time_taken',
    })

    # 4. text columns: strip spaces + lowercase (blanks stay blank)
    for col in ['traffic', 'weather', 'festival', 'city', 'order_type', 'vehicle_type']:
        df[col] = df[col].str.strip().str.lower()
    df['weather'] = df['weather'].str.replace('conditions ', '', regex=False)
    df['weather'] = df['weather'].mask(df['weather'] == 'nan')

    # 5. number columns
    for col in ['age', 'rating', 'no_of_deliveries']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['total_time_taken'] = pd.to_numeric(
        df['total_time_taken'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')

    # 6. date
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

    # 7. impossible values -> blank (blank ages are kept)
    df = df[df['age'].isna() | df['age'].between(20, 39)].copy()
    df.loc[~df['rating'].between(1, 5), 'rating'] = np.nan
    df.loc[~df['vehicle_condition'].between(0, 2), 'vehicle_condition'] = np.nan
    df.loc[df['vehicle_type'] == 'bicycle', 'vehicle_type'] = np.nan

    # 8. time features
    t_order = pd.to_datetime(df['time_of_order'].str.strip(), format='%H:%M:%S', errors='coerce')
    t_pick = pd.to_datetime(df['pickup_time'].str.strip(), format='%H:%M:%S', errors='coerce')
    df['order_hour'] = t_order.dt.hour
    df['pickup_time(min)'] = ((t_pick - t_order).dt.total_seconds() / 60) % 1440

    h = df['order_hour']
    df['time_of_day'] = np.select(
        [h.between(5, 11), h.between(12, 16), h.between(17, 20), (h >= 21) | (h < 5)],
        ['morning', 'afternoon', 'evening', 'night'],
        default='unknown')
    df.loc[h.isna(), 'time_of_day'] = np.nan

    # 9. date features
    df['day_name'] = df['date'].dt.day_name().str.lower()
    df['month_name'] = df['date'].dt.month_name().str.lower()
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype('boolean').mask(df['date'].isna())

    # 10. coordinates: positive, outside India -> blank
    ranges = {
        'restaurant_latitude': (8, 37), 'location_latitude': (8, 37),
        'restaurant_longitude': (68, 98), 'location_longitude': (68, 98),
    }
    for col, (low, high) in ranges.items():
        df[col] = df[col].abs()
        df.loc[~df[col].between(low, high), col] = np.nan

    # 11. distance (blank where a coordinate is blank)
    df['distance_km'] = haversine(
        df['restaurant_latitude'], df['restaurant_longitude'],
        df['location_latitude'], df['location_longitude']).round(2)

    return df.reset_index(drop=True)


@app.command()
def main(input_path: Path = RAW_FILE, output_path: Path = CLEANED_FILE):
    logger.info(f"Reading raw data from {input_path}")
    df = clean_data(pd.read_csv(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Saved cleaned data {df.shape} to {output_path}")


if __name__ == "__main__":
    app()