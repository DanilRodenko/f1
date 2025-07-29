import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

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


def get_weather_text_from_url(url: str) -> str | None:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        infobox = soup.find("table", class_="infobox")
        if not infobox:
            return None

        for row in infobox.find_all("tr"):
            header = row.find("th")
            if header and 'weather' in header.text.lower():
                cell = row.find("td")
                return cell.text.strip() if cell else None

        return None

    except Exception as e:
        print(f"Error for {url}: {e}")
        return None

