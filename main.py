# === Core and Utilities ===
import os
import json
import joblib
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# === Scikit-learn ===
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# === Project: Data ===
from src.data.data_manager import DataManager
from src.data.data_preprocess import merge_datasets, clean_masterdata, merge_weather
from src.data.features_engineering import (
    unique_values, missing_values, duplicates_values,
    add_top3_column, top3_percent_last3years, avg_position_last3years,
    avg_driver_position_current_year, avg_constructor_position_current_year,
    avg_driver_position_by_weather, add_phase_2_6_7_8_features
)
from src.data.data_visualize import plot_constructor_nationality, plot_bar
from src.data.analysis import Top3Analyser

# === Project: Models - Logistic Regression ===
from src.models.train_utils import preprocess_model_data, evaluate_naive_bayes_test_confusion
from src.models.train import run_logistic_regression_from_data, evaluate_roc_auc, evaluate_threshold_metrics, check_logistic_significance
from src.models.logistic_model import run_logistic_regression
from src.models.evaluation import plot_roc_curve, evaluate_model_classification, find_best_threshold
from src.models.experiment_variants import run_all_variants
from src.models.finalize_logreg import finalize_logistic_model

# === Project: Models - Naive Bayes ===
from src.models.naive_bayes_model import train_naive_bayes_model
from src.models.naive_bayes_utils import (
    evaluate_naive_bayes_train_roc,
    evaluate_naive_bayes_train_confusion,
    evaluate_naive_bayes_test_roc,
    evaluate_naive_bayes_train_confusion
)

# === Project: Models - Decision Tree ===
from src.models.decision_tree_model import train_decision_tree_model

# === Project: Models - Random Forest ===
from src.models.random_forest_model import train_random_forest_model

# === Project: Other ===
from src.models.vif_check import calculate_vif


def main():
    global masterdata, modelling_data, d_top3, c_top3, d_avg, c_avg

    print("=== Phase 1.1: Understand all data sets ===")
    dm = DataManager("datasets/raw")
    datasets = dm.list_datasets()
    print(f"✅ Loaded datasets: {datasets}")

    for name in datasets:
        df = dm.get(name)
        if df is None:
            print(f"⚠️ Dataset '{name}' not found or failed to load.")
            continue
        print(f"\n📦 Dataset: {name}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.head(2))

    print("\n=== Phase 1.2: Find unique count of Constructors, Drivers and Races from 'results' data")
    results = dm.get("results")
    unique_df = pd.DataFrame({
        'Category': ['Constructors', 'Drivers', 'Races'],
        'Unique Count': [
            unique_values(results['constructorId']),
            unique_values(results['driverId']),
            unique_values(results['raceId']),
        ]
    })
    print(unique_df)

    print("\n=== Phase 1.3: Check missing and duplicate IDs ===")
    categories = ['Constructors', 'Drivers', 'Races']
    columns = ['constructorId', 'driverId', 'raceId']
    tables = ['constructors', 'drivers', 'races']

    missing_list, duplicates_list = [], []
    for cat, col, table in zip(categories, columns, tables):
        base_ids = results[col]
        ref_df = dm.get(table)
        ref_ids = ref_df[col] if ref_df is not None and col in ref_df.columns else None
        missing = missing_values(base_ids, ref_ids) if ref_ids is not None else '⚠️'
        missing_list.append({'Category': cat, 'Missing Values': missing})

        dup = duplicates_values(ref_df[col]) if ref_df is not None and col in ref_df.columns else '⚠️'
        duplicates_list.append({'Category': cat, 'Duplicate Count': dup})

    print(pd.DataFrame(missing_list))
    print(pd.DataFrame(duplicates_list))

    print("\n=== Phase 1.4: Frequency distribution of constructors by nationality ===")
    constructors = dm.get('constructors')
    if constructors is not None and 'nationality' in constructors.columns:
        nationality_counts = constructors.groupby('nationality').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
        print(nationality_counts)
        plot_constructor_nationality(constructors)

    print("\n=== Phase 1.5: Merge all data sets and create masterdata ===")
    races = dm.get("races")
    drivers = dm.get("drivers")
    qualifying = dm.get("qualifying")

    datasets_ready = all(isinstance(df, pd.DataFrame) for df in [results, races, drivers, constructors, qualifying])
    if not datasets_ready:
        print("⚠️ Missing or invalid datasets. Cannot proceed with merging.")
        

    merge_plan = [
        (races, "raceId"),
        (drivers, "driverId"),
        (constructors, "constructorId"),
        (qualifying, ["raceId", "driverId"])
    ]
    masterdata = merge_datasets(results, merge_plan)
    masterdata = clean_masterdata(masterdata)
    masterdata = add_top3_column(masterdata, position_col='race_position', new_col='top3')

    print(f"✅ Masterdata created → shape: {masterdata.shape}")
    print(f"✅ 'top3' column added: {masterdata['top3'].sum()} top-3 entries")

    masterdata.to_csv("datasets/processed/masterdata.csv", index=False)
    print("📦 Masterdata saved to datasets/processed/masterdata.csv")

    print("\n=== Phase 1.6-1.8: Analyse drivers and constructors by top-3 ===")
    analyser = Top3Analyser(masterdata)
    drivers_top3 = analyser.analyse('driverRef', 'full_name')
    constructors_top3 = analyser.analyse('constructorRef', 'constructor_name')

    plot_bar(drivers_top3.head(15), 'full_name', 'top3_count', 'Top 15 Drivers', 'Driver', 'Top-3 Count')
    plot_bar(constructors_top3.head(15), 'constructor_name', 'top3_count', 'Top 15 Constructors', 'Constructor', 'Top-3 Count')

    print("\n=== Phase 2: Subset data from 2010 onward ===")
    modelling_data = masterdata[masterdata['year'] >= 2010].copy()
    print(f"✅ Modelling subset shape: {modelling_data.shape}")

    print("\n=== Phase 2.1–2.4: Top-3 % and Avg Position (last 3 years) ===")
    d_top3 = top3_percent_last3years(modelling_data, 'driverRef')
    c_top3 = top3_percent_last3years(modelling_data, 'constructorRef')
    d_avg = avg_position_last3years(modelling_data, 'driverRef')
    c_avg = avg_position_last3years(modelling_data, 'constructorRef')

    print("✅ Drivers Top-3 %:\n", d_top3.head(3))
    print("✅ Constructors Top-3 %:\n", c_top3.head(3))
    print("✅ Drivers Avg Pos:\n", d_avg.head(3))
    print("✅ Constructors Avg Pos:\n", c_avg.head(3))

    print("\n=== Phase 2.5: Merge weather data ===")
    masterdata = merge_weather()
    masterdata = masterdata[masterdata.year >=2010]
    print(f"✅ Final masterdata with weather → shape: {masterdata.shape}")

    print("\n=== Phase 2.6–2.8: Add average positions ===")
    masterdata = add_phase_2_6_7_8_features(masterdata)
    print(masterdata[['raceId', 'year', 'driverRef', 'avg_position_ytd_driver', 'avg_position_ytd_constructor', 'avg_position_weather_last3yrs']].head(10))
    masterdata.to_csv("datasets/processed/masterdata.csv", index=False)
    print("Final masterdata saved with all features.")

    ########################
    print("Phase 3 is about developing a model using Binary Logistic Regression(using final master data created in phase 2)")
    print("Phase 3.1: Create data partition into train and test data sets-recent data can be test data")
    
    train = masterdata[masterdata['year'] <= 2023].copy()
    test = masterdata[masterdata['year'] == 2024].copy()
    
    print("\n=== Phase 3.2: Run Binary Logistic Regression ===")
    model = run_logistic_regression(masterdata)
    X, y = preprocess_model_data(masterdata)
    vif_table = calculate_vif(X)
    
    print("\n=== Phase 3.3: VIF check for multicollinearity ===")
    print(vif_table)


    print("\n=== Phase 3.4: Check variable significance ===")
    check_logistic_significance(X, y)
    

    print("\n=== Phase 3.5: ROC и AUC ===")
    plot_roc_curve(model, X, y, title="Train ROC Curve")

    
    print("\n=== Phase 3.6: Classification Table and Accuracy ===") 
    evaluate_model_classification(model, X, y)

    print("\n=== Phase 3.7: Find best threshold ===")
    y_probs = model.predict_proba(X)[:, 1]
    best_threshold = find_best_threshold(y, y_probs)

    print("\n=== Phase 3.8: ROC and AUC for Test Data ===")
    X, y = preprocess_model_data(masterdata)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = run_logistic_regression_from_data(X_train, y_train)
    evaluate_roc_auc(model, X_test, y_test, plot_title="Test ROC Curve")

    print("\n=== Phase 3.9: Final evaluation on test set using best threshold ===")
    
    y_probs_test = model.predict_proba(X_test)[:, 1]
    evaluate_threshold_metrics(y_test, y_probs_test, threshold=best_threshold)    
    
    print("\n=== Phase 3.10: Perform steps 1-9 for different versions of variables.  ===")
 
    variant_results_df = run_all_variants(masterdata)
    print("\n=== Model Comparison by AUC ===")
    print(variant_results_df.sort_values("AUC", ascending=False))

    print("\n=== Phase 3.11: Finalize Logistic Regression Model ===")
    
    
    best_features = [
        'driverId', 'constructorId', 'year', 'circuitId',
        'avg_position_ytd_driver', 'avg_position_ytd_constructor',
        'avg_position_weather_last3yrs', 'weather_numeric'
    ]
    
    finalize_logistic_model(masterdata, best_features)
    
    
    print("\n=== Phase 4.1–4.2: Train Naive Bayes model ===")
    train_naive_bayes_model(masterdata, best_features)

    print("\n=== Phase 4.3: ROC Curve and AUC on Train Data (Naive Bayes) ===")
    X, y = preprocess_model_data(masterdata)
    X = X[best_features]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)
    evaluate_naive_bayes_train_roc(model, X_train_scaled, y_train)

    print("\n=== Phase 4.4: Confusion Matrix on Train Data (Naive Bayes) ===")
    evaluate_naive_bayes_train_confusion(model, X_train_scaled, y_train, threshold=0.5)



    print("\n=== Phase 4.5: ROC Curve on Test Data (Naive Bayes) ===")
    nb_model = joblib.load("results/models/nb_model.joblib")
    scaler = joblib.load("results/models/nb_scaler.joblib")
        
    with open("results/models/nb_features.json") as f:
        features = json.load(f)
    
    X, y = preprocess_model_data(masterdata)
    X = X[features]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_test_scaled = scaler.transform(X_test)
    evaluate_naive_bayes_test_roc(nb_model, X_test_scaled, y_test)

    print("\n=== Phase 4.6: Confusion Matrix on Test Data (Naive Bayes) ===")
    evaluate_naive_bayes_test_confusion(nb_model, X_test_scaled, y_test, threshold=0.5)

    print("\n=== Phase 4.7.1–2: Train Decision Tree model ===")
    train_decision_tree_model(masterdata, best_features)
    
    X, y = preprocess_model_data(masterdata)
    X = X[best_features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    
    # Phase 4.7.3
    print("\n=== Phase 4.7.3: ROC Curve on Train Data (Decision Tree) ===")
    evaluate_naive_bayes_train_roc(dt_model, X_train_scaled, y_train)
    
    # Phase 4.7.4
    print("\n=== Phase 4.7.4: Confusion Matrix on Train Data (Decision Tree) ===")
    evaluate_naive_bayes_train_confusion(dt_model, X_train_scaled, y_train, threshold=0.5)
    
    # Phase 4.7.5
    print("\n=== Phase 4.7.5: ROC Curve on Test Data (Decision Tree) ===")
    evaluate_naive_bayes_test_roc(dt_model, X_test_scaled, y_test)
    
    # Phase 4.7.6
    print("\n=== Phase 4.7.6: Confusion Matrix on Test Data (Decision Tree) ===")
    evaluate_naive_bayes_test_confusion(dt_model, X_test_scaled, y_test, threshold=0.5)



    print("\n=== Phase 4.8: Train Random Forest ===")
    rf_model, rf_scaler = train_random_forest_model(masterdata, best_features)
    
    X, y = preprocess_model_data(masterdata)
    X = X[best_features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    X_train_scaled = rf_scaler.transform(X_train)
    X_test_scaled = rf_scaler.transform(X_test)
    
    # 4.3 — ROC & AUC train
    print("\n=== Phase 4.8.1: Train ROC Curve and AUC (RF) ===")
    evaluate_roc_auc(rf_model, X_train_scaled, y_train, "Train ROC Curve - RF")
    
    # 4.4 — Confusion matrix train
    print("\n=== Phase 4.8.2: Train Confusion Matrix (RF) ===")
    y_probs_train = rf_model.predict_proba(X_train_scaled)[:, 1]
    evaluate_threshold_metrics(y_train, y_probs_train, threshold=0.5)
    
    # 4.5 — ROC & AUC test
    print("\n=== Phase 4.8.3: Test ROC Curve and AUC (RF) ===")
    evaluate_roc_auc(rf_model, X_test_scaled, y_test, "Test ROC Curve - RF")
    
    # 4.6 — Confusion matrix test
    print("\n=== Phase 4.8.4: Test Confusion Matrix (RF) ===")
    y_probs_test = rf_model.predict_proba(X_test_scaled)[:, 1]
    evaluate_threshold_metrics(y_test, y_probs_test, threshold=0.5)

if __name__ == "__main__":
    main()
