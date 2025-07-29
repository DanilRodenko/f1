import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm

def preprocess_model_data(df):
    df = df.copy()

    weather_map = {'Dry': 0, 'Wet': 1, 'Variable': 2}
    df['weather_numeric'] = df['weather'].map(weather_map)

    feature_cols = [
        'driverId', 'constructorId', 'year', 'circuitId',
        'avg_position_ytd_driver', 'avg_position_ytd_constructor',
        'avg_position_weather_last3yrs', 'weather_numeric'
    ]

    df = df.dropna(subset=feature_cols + ['top3'])

    X = df[feature_cols]
    y = df['top3']

    return X, y


def run_logistic_regression(df, test_year=2024):
    train_df = df[df['year'] < test_year]
    test_df = df[df['year'] == test_year]

    print(f"📊 Train size: {train_df.shape[0]} | Test size: {test_df.shape[0]}")

    X_train, y_train = preprocess_model_data(train_df)
    X_test, y_test = preprocess_model_data(test_df)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n✅ Logistic Regression Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model


def check_logistic_significance(X, y):
    X_const = sm.add_constant(X)
    model = sm.Logit(y, X_const)
    result = model.fit()
    print(result.summary())
    return result
