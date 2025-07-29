import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.models.train_utils import preprocess_model_data

def finalize_logistic_model(masterdata, features,
                            model_path="results/models/logreg_model.joblib",
                            scaler_path="results/models/logreg_scaler.joblib",
                            features_path="results/models/logreg_features.json"):

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    X, y = preprocess_model_data(masterdata)
    X = X[features]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(features_path, "w") as f:
        json.dump(features, f)

    print(f"✅ Logistic Regression model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    print(f"✅ Feature list saved to: {features_path}")
