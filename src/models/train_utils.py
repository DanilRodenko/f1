import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def preprocess_model_data(df: pd.DataFrame):
    df_clean = df.copy()

    required_cols = [
        'driverId', 'constructorId', 'year', 'circuitId',
        'avg_position_ytd_driver', 'avg_position_ytd_constructor',
        'avg_position_weather_last3yrs', 'weather'
    ]
    df_clean = df_clean.dropna(subset=required_cols + ['top3'])

    weather_map = {'Dry': 0, 'Wet': 1, 'Variable': 2}
    df_clean['weather_numeric'] = df_clean['weather'].map(weather_map)

    feature_cols = [
        'driverId', 'constructorId', 'year', 'circuitId',
        'avg_position_ytd_driver', 'avg_position_ytd_constructor',
        'avg_position_weather_last3yrs', 'weather_numeric'
    ]
    X = df_clean[feature_cols]
    y = df_clean['top3'].astype(int)

    return X, y

def split_train_test(X, y, test_size=0.2, shuffle=False, random_state=42):
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=random_state)


def evaluate_naive_bayes_test_confusion(model, X_test_scaled, y_test, threshold=0.5):
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"✅ Threshold: {threshold:.2f}")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Sensitivity (TPR): {sensitivity:.3f}")
    print(f"   Specificity (TNR): {specificity:.3f}")