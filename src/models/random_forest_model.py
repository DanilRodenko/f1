import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from src.models.train_utils import preprocess_model_data

def train_random_forest_model(masterdata, features,
                              model_path="models/rf_model.joblib",
                              scaler_path="models/rf_scaler.joblib",
                              features_path="models/rf_features.json"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    X, y = preprocess_model_data(masterdata)
    X = X[features]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(features_path, "w") as f:
        json.dump(features, f)

    print("✅ Random Forest model trained and saved.")
    return model, scaler
