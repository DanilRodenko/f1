import pandas as pd

from src.data.data_manager import DataManager
from src.data.features_engineering import add_top3_column


# --- Merge and clean masterdata ---

def merge_datasets(base_df, merge_plan):
    """
    Sequentially merge multiple datasets into a base DataFrame.
    """
    df = base_df.copy()
    for i, (other_df, on_keys) in enumerate(merge_plan):
        df = pd.merge(df, other_df, how='left', on=on_keys)
        print(f" Merge step {i+1}: on {on_keys} → shape: {df.shape}")
    return df

def clean_masterdata(df):
    """
    Drop or rename duplicate or redundant columns after merging.
    """
    to_drop = ['constructorId_y', 'number', 'number_y']
    df = df.drop(columns=to_drop, errors='ignore')

    rename_map = {
        'name_x': 'gp_name',
        'name_y': 'constructor_name',
        'position_x': 'race_position',
        'position_y': 'qualifying_position',
        'time_x': 'race_time',
        'time_y': 'race_start',
        'url': 'url_constructor',
        'url_x': 'url_gp',
        'url_y': 'url_driver',
        'number_x' : 'number',
        'nationality_y' : 'constructor_nationality',
        'nationality_x' : 'driver_nationality',
        'constructorId_x' : 'constructorId'
    }

    df = df.rename(columns=rename_map)
    return df

# --- Entry point to build full masterdata ---

def get_masterdata():
    """
    Loads and returns the full master dataset (merged, cleaned, with top3 column added).
    """
    data = DataManager("C:/Users/danil/Desktop/PROJECTS/formula1/datasets/raw/")

    merge_plan = [
        (data.get("races"), "raceId"),
        (data.get("drivers"), "driverId"),
        (data.get("constructors"), "constructorId"),
        (data.get("qualifying"), ["raceId", "driverId"])
    ]

    df = merge_datasets(data.get("results"), merge_plan)
    df = clean_masterdata(df)
    df = add_top3_column(df)

    return df

def merge_weather(masterdata_path="datasets/processed/masterdata.csv"):
    master = pd.read_csv(masterdata_path)
    weather = pd.read_csv("datasets/processed/weather_conditions.csv")

    master['raceId'] = master['raceId'].astype(int)
    weather['raceId'] = weather['raceId'].astype(int)

    merged = master.merge(weather[['raceId', 'weather']], on='raceId', how='left')

    print(f"✅ Rows with weather info: {merged['weather'].notna().sum()} / {merged.shape[0]}")
    return merged

def load_masterdata_final(path="datasets/processed/masterdata.csv"):
    return pd.read_csv(path)

# --- Optional preview when running directly ---

if __name__ == "__main__":
    masterdata = get_masterdata()
    print(masterdata.head())
    print(f"Masterdata shape: {masterdata.shape}")
