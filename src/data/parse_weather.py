import pandas as pd
from tqdm import tqdm
from src.data.data_loader import get_weather_text_from_url

def build_weather_dataset(gp_table: pd.DataFrame, save_path: str = None, verbose: bool = True):

    tqdm.pandas()  # Включает прогресс-бар для .progress_apply

    gp_table = gp_table.copy()
    gp_table['weather_raw'] = gp_table['url_gp'].progress_apply(get_weather_text_from_url)

    if save_path:
        gp_table.to_csv(save_path, index=False)
        if verbose:
            print(f"✅ Weather data saved to: {save_path}")
    return gp_table


if __name__ == "__main__":

    master = pd.read_csv("datasets/processed/masterdata.csv")
    gp_table = master[['raceId', 'year', 'url_gp']].drop_duplicates().sort_values(by='year')
    gp_table = gp_table[gp_table['year'] >= 2010]

    build_weather_dataset(
        gp_table,
        save_path="datasets/processed/gp_weather.csv"
    )
