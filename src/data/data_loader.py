import os
import pandas as pd


def load_all_cvs(directory_path):
    data_dict = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            try:
                data_name = filename.replace(".csv", "")
                data_dict[data_name] = pd.read_csv(file_path)
            except Exception as e:
                print(f"Uploading Error {filename}: {e}")
    return data_dict


