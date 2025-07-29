# from src.data.data_loader import load_all_cvs

# class DataManager:
#     def __init__(self, raw_data_path: str):
#         # Load all CSV files from the specified directory
#         self.data_dict = load_all_cvs(raw_data_path)

#         # Select and store only main datasets
#         self.main_keys = ["constructors", "drivers", "qualifying", "races", "results"]
#         self.main_data = {key: self.data_dict.get(key) for key in self.main_keys}

#     def get(self, name):
#         # Get a specific dataset by name
#         return self.main_data.get(name)

#     def list_datasets(self):
#         # List all available main dataset names
#         return list(self.main_data.keys())

from src.data.data_loader import load_all_cvs
import pandas as pd

class DataManager:
    def __init__(self, raw_data_path: str):
        # Load all CSV files from the specified directory
        self.data_dict = load_all_cvs(raw_data_path)

        # Select and store only main datasets
        self.main_keys = [
    "constructors", "drivers", "qualifying", "races",
    "results"
]
        self.main_data = {key: self.data_dict.get(key) for key in self.main_keys}

    def get(self, name):
        return self.main_data.get(name)

    def list_datasets(self):
        return list(self.main_data.keys())

    def get_weather_data(self):
        return self.get("weather_conditions")

    def get_masterdata(self):
        return self.get("gp_weather")

    def get_merged_masterdata(self):
        master = self.get_masterdata()
        weather = self.get_weather_data()

        if master is None or weather is None:
            raise ValueError("Masterdata or weather_conditions not loaded.")

        if 'raceId' not in master.columns or 'raceId' not in weather.columns:
            raise ValueError("Missing 'raceId' column in one of the datasets.")

        merged = master.merge(weather, on='raceId', how='left')
        return merged
