import os
import json
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.train_utils import preprocess_model_data

def train_decision_tree_model(masterdata, features,
                              model_path="results/models/dt_model.joblib",
                              scaler_path="results/models/dt_scaler.joblib",
                              features_path="results/models/dt_features.json"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    X, y = preprocess_model_data(masterdata)
    X = X[features]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(features_path, "w") as f:
        json.dump(features, f)

    print("Decision Tree model trained and saved.")
