import pandas as pd
from src.data.features_engineering import classify_weather_conditions

def run_weather_pipeline():
    df = pd.read_csv("datasets/processed/gp_weather.csv")
    weather_df = classify_weather_conditions(df)

    save_path = "datasets/processed/weather_conditions.csv"
    weather_df.to_csv(save_path, index=False)
    print(f"✅ Weather conditions saved to {save_path}")
    print(weather_df.head(5))


if __name__ == "__main__":
    run_weather_pipeline()
