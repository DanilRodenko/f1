# src/model/experiment_variants.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.models.train_utils import preprocess_model_data



def evaluate_variant(masterdata, feature_columns, variant_name):
    X, y = preprocess_model_data(masterdata)
    X = X[feature_columns]  # subset features
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)
    print(f"✅ Variant '{variant_name}' AUC: {auc:.4f}")

    return {
        "Variant": variant_name,
        "AUC": auc,
        "Model": model,
        "Features": feature_columns
    }


def run_all_variants(masterdata):
    results = []

    all_features = [
        'driverId', 'constructorId', 'year', 'circuitId',
        'avg_position_ytd_driver', 'avg_position_ytd_constructor',
        'avg_position_weather_last3yrs', 'weather_numeric'
    ]

    # Variant A: All features
    results.append(evaluate_variant(masterdata, all_features, "All features"))

    # Variant B: Drop IDs
    features_b = [f for f in all_features if f not in ['driverId', 'constructorId']]
    results.append(evaluate_variant(masterdata, features_b, "No IDs"))

    # Variant C: Drop weather
    features_c = [f for f in all_features if 'weather' not in f]
    results.append(evaluate_variant(masterdata, features_c, "No weather"))

    # Variant D: Only average position features
    features_d = [f for f in all_features if 'avg_position' in f]
    results.append(evaluate_variant(masterdata, features_d, "Only avg positions"))

    return pd.DataFrame(results)


if __name__ == "__main__":
    from src.data.data_preprocess import load_masterdata_final

    masterdata = load_masterdata_final()  # function should load final CSV
    result_df = run_all_variants(masterdata)
    print("\n=== Model Comparison by AUC ===")
    print(result_df.sort_values("AUC", ascending=False))
