import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util


# Get count of unique values in a column
def unique_values(column):
    return column.nunique()

# Identify values in column1 that are missing from column2
def missing_values(column1, column2):
    return set(column1) - set(column2)

# Count duplicate values in a column
def duplicates_values(column):
    return column.duplicated().sum()


# Add a binary "top3" column based on position
def add_top3_column(data, position_col='race_position', new_col = 'top3'):
    data[position_col] = pd.to_numeric(data[position_col], errors='coerce')
    data[new_col] = data[position_col].apply(lambda x:1 if x <=3 else 0)
    return data


def filtered_year(data, start_year, finish_year=None):
    if finish_year:
        return data[(data['year'] >= start_year) & (data['year'] <= finish_year)].copy()
    else:
        return data[data['year'] >= start_year].copy()

def filter_last_n_years(data, n=3):
    years = data['year'].dropna().unique()
    last_n = sorted(years)[-n:]
    return data[data['year'].isin(last_n)].copy()


def top3_percent_last3years(data, column):
    df = filter_last_n_years(data, 3).copy()

    # Create full name if needed, but use separate column
    if column == 'driverRef':
        df["full_name"] = df["forename"] + " " + df["surname"]
        group_col = "full_name"
    elif column == 'constructorRef':
        group_col = 'constructor_name'
    else:
        group_col = column

    # Group and calculate
    agg = (
        df.groupby([group_col, 'year'])
        .agg(top3_percent=('top3', lambda x: 100 * x.sum() / len(x)))
        .reset_index()
    )
    
    # Pivot table
    pivot = lambda val: agg.pivot(index=group_col, columns='year', values=val)
    top3 = pivot('top3_percent')

    return top3


def avg_position_last3years(data, column):
    df = filter_last_n_years(data, 3)
    
    # Create full name if needed, but use separate column
    if column == 'driverRef':
        df["full_name"] = df["forename"] + " " + df["surname"]
        group_col = "full_name"
    elif column == 'constructorRef':
        group_col = 'constructor_name'
    else:
        group_col = column
    
    agg = df.groupby([group_col, 'year']).agg(
        avg_pos = ('race_position', 'mean')
        ).reset_index()
    
    pivot = lambda val: agg.pivot(index=group_col, columns='year', values=val)
    avg = pivot('avg_pos')
    
    return avg
    
# Analyze top-3 finishes for driver-constructor pairs
def top3_by_driver_constructor(data, min_year=None):
    df = data.copy()
    if min_year:
        df = df[df['year'] >= min_year]

    if "full_name" not in df.columns:
        df["full_name"] = df["forename"] + " " + df["surname"]

    result = (
        df.groupby(['full_name', 'constructor_name'])
        .agg(
            top3_count=('top3', 'sum'),
            total_races=('raceId', 'count')
        )
        .assign(top3_percent=lambda d: 100 * d['top3_count'] / d['total_races'])
        .sort_values(by='top3_count', ascending=False)
        .reset_index()
    )
    return result





def classify_weather_conditions(df: pd.DataFrame, text_col: str = "weather_raw") -> pd.DataFrame:

    weather = {
        'Dry': ['sunny', 'mild', 'dry', 'overcast', 'hot', 'clear'],
        'Wet': ['rain', 'wet', 'light rain', 'heavy rain', 'rainy'],
        'Variable': ['half-distance', 'start', 'finish', 'part', 'then', 'rain', 'wet', 'dry', 'sunny', 'hot', 'end']
    }

    weather_names = list(weather.keys())
    category_texts = [" ".join(keywords) for keywords in weather.values()]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    category_embeddings = model.encode(category_texts, normalize_embeddings=True)

    def find_weather(user_text: str):
        if not isinstance(user_text, str) or user_text.strip() == '':
            return 'Unknown'
        user_embedding = model.encode(user_text, normalize_embeddings=True)
        scores = util.cos_sim(user_embedding, category_embeddings)[0]
        top_result_idx = scores.argmax().item()
        return weather_names[top_result_idx]

    df = df.copy()
    df = df.dropna(subset=[text_col])
    df['weather'] = df[text_col].apply(find_weather)
    return df[['raceId', 'weather']]


def avg_driver_position_current_year(df):
    df = df.sort_values(['year', 'raceId'])  # Упорядочим
    df['avg_position_ytd_driver'] = (
        df.groupby(['year', 'driverId'])['race_position']
        .expanding()
        .mean()
        .shift()
        .reset_index(level=[0,1], drop=True)
    )
    return df

def avg_constructor_position_current_year(df):
    df = df.sort_values(['year', 'raceId'])
    df['avg_position_ytd_constructor'] = (
        df.groupby(['year', 'constructorId'])['race_position']
        .expanding()
        .mean()
        .shift()
        .reset_index(level=[0,1], drop=True)
    )
    return df


def avg_driver_position_by_weather(df):
    df = df.sort_values(['year', 'raceId'])

    avg_weather_3yr = []

    for idx, row in df.iterrows():
        year = row['year']
        driver = row['driverId']
        weather = row['weather']
        race_id = row['raceId']

        past_data = df[
            (df['year'] < year) &
            (df['year'] >= year - 3) &
            (df['driverId'] == driver) &
            (df['weather'] == weather)
        ]

        avg_pos = past_data['race_position'].mean() if not past_data.empty else None
        avg_weather_3yr.append(avg_pos)

    df['avg_position_weather_last3yrs'] = avg_weather_3yr
    return df


def add_phase_2_6_7_8_features(df):
    df = avg_driver_position_current_year(df)
    df = avg_constructor_position_current_year(df)
    df = avg_driver_position_by_weather(df)
    return df

