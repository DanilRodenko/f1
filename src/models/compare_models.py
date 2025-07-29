import pandas as pd
from sklearn.metrics import roc_auc_score
from src.models.train_utils import preprocess_model_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def compare_models_auc(masterdata, features):
    X, y = preprocess_model_data(masterdata)
    X = X[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        )
    }

    results = []

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_probs = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_probs)
        results.append({"Model": name, "Test AUC": auc})

    return pd.DataFrame(results).sort_values("Test AUC", ascending=False)


if __name__ == "__main__":
    from src.data.data_preprocess import load_masterdata_final

    masterdata = load_masterdata_final()

    best_features = [
        'driverId', 'constructorId', 'year', 'circuitId',
        'avg_position_ytd_driver', 'avg_position_ytd_constructor',
        'avg_position_weather_last3yrs', 'weather_numeric'
    ]

    df_auc = compare_models_auc(masterdata, best_features)
    print("\n=== Phase 4.9: AUC Comparison on Test Data ===")
    print(df_auc.to_string(index=False))
